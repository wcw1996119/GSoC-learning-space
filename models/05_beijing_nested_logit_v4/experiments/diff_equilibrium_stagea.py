# -*- coding: utf-8 -*-
"""Stage B with the REAL Stage-A demand: wire the trained 21-class nested-logit forward
into the differentiable congestion equilibrium (the toy logit of diff_equilibrium.py is
replaced by beijing_model.BeijingNestedHead).

demand(t_link):  link times --Delta--> OD car times --override t_car--> Stage-A forward
                 --> car flow per edge --aggregate--> OD node-pairs --Delta^T--> link volumes --BPR-->

The GNN residual is frozen (computed once). Run a smoke forward equilibrium:
    python experiments/diff_equilibrium_stagea.py --pt evaluation_outputs/v4_full_anchored_s0.pt --smoke-cells 400
"""
import argparse, pathlib, sys
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
import train_beijing as TB
from beijing_model import BeijingNestedHead, BeijingPairEncoder
from diff_equilibrium import bpr, scipy_to_torch_sparse, solve_fixed_point, PERIOD_HOURS, TARGET_VC


def build_delta_aligned(n_nodes, eu, ev, ff, pairs):
    """Free-flow shortest-path tree per origin -> Delta aligned to the given `pairs` order.
    Pairs with no path (self / unreachable) get an all-zero row (handled by a fixed-time fallback)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    G = sp.csr_matrix((ff, (eu, ev)), shape=(n_nodes, n_nodes))
    link_id = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(eu, ev))}
    by_o = {}
    for r, (o, d) in enumerate(pairs):
        by_o.setdefault(int(o), []).append((r, int(d)))
    rows, cols, has_path = [], [], np.zeros(len(pairs), bool)
    for o, rd in by_o.items():
        dist, pred = dijkstra(G, indices=o, return_predecessors=True)
        for r, d in rd:
            if d == o or not np.isfinite(dist[d]):
                continue
            nd = d
            while pred[nd] >= 0:
                p = int(pred[nd]); lid = link_id.get((p, nd))
                if lid is not None:
                    rows.append(r); cols.append(lid)
                nd = p
            has_path[r] = True
    Delta = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pairs), len(eu)))
    return Delta, has_path


def load_stage_a(pt_path, smoke_cells, device="cpu"):
    pt = torch.load(pt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**pt["args"]); args.device = device; args.smoke_cells = smoke_cells
    # anchors only affect the training loss, not the forward -> disable so build_data needs no anchor files
    for k in ("anchor_transit", "anchor_share", "anchor_mode_dist", "anchor_income_time"):
        setattr(args, k, False)
    data = TB.build_data(args)
    g = lambda k, dv: getattr(args, k, dv)
    head = BeijingNestedHead(n_districts=16, use_self_loop=not g("no_self_loop", False),
                             use_consideration=g("use_consideration", False),
                             use_soc_mixture=g("use_soc_mixture", False), gnn_mode=g("gnn_mode", "residual"),
                             residual_scale_init=g("residual_scale_init", 0.1),
                             use_typed_mass=g("typed_mass_occ", False),
                             use_match_filter=not g("no_match_filter", False),
                             use_frozen_occ_mask=g("use_frozen_occ_mask", False),
                             soft_lex_match=g("soft_lex_match", False),
                             k_match_min=g("k_match_min", 20.0)).to(device)
    if g("soft_lex_match", False) and g("use_soc_mixture", False):
        head.set_tau_match_floor(g("tau_match_floor_mult", 0.5) * data["demand_share"].mean(0))
    head.load_state_dict(pt["head_state"]); head.eval()
    enc = None
    if args.use_nn and not getattr(args, "use_dynamic", False):
        enc = BeijingPairEncoder(in_dim=data["Xnode"].shape[1]).to(device)
        enc.load_state_dict(pt["enc_state"]); enc.eval()
    return args, data, head, enc


def run(pt_path, period=0, smoke_cells=400, n_iter=400):
    dev = "cpu"
    args, data, head, enc = load_stage_a(pt_path, smoke_cells, dev)
    use_soc = args.use_soc_mixture
    o = data["o"]; d = data["d"]; seg_id = data["seg_id"]; O = data["num_seg"]
    flow = data["flow"]; otot = torch.zeros(O, dtype=flow.dtype).index_add(0, seg_id, flow[:, period])

    # frozen GNN residual per edge (computed once)
    with torch.no_grad():
        if enc is not None:
            e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"],
                                      edge_weight=data.get("edge_weight"))
            v_nn = enc.edge_vnn(e_o, e_d, o, d).detach()
        else:
            v_nn = None

    def stage_a_car_flow(t_car_edge):
        t_min = torch.minimum(torch.minimum(t_car_edge, data["t_transit"]), data["t_walk"])
        b = dict(seg_id=seg_id, num_seg=O, t_car=t_car_edge, t_min=t_min,
                 t_transit=data["t_transit"], t_walk=data["t_walk"], log_d=data["log_d"],
                 log_M_d=data["log_M"][d], log_W_d=data["log_W"][d], log_D_d=data["log_D"][d],
                 match=data["match"], income_o=data["income"][o], tier_props_o=data["tier_props"][o],
                 district_o=data["district"][o], is_self=(o == d), period=period)
        if use_soc:
            b["soc_props_o"] = data["soc_props"][o]; b["demand_share_d"] = data["demand_share"][d]
            if "occ_mask" in data: b["occ_mask_d"] = data["occ_mask"][d]
        logP = head(b, v_nn=v_nn)                       # mixed log P(d|o)
        p_car = torch.softmax(head.mode_logits(b), 0)[0]  # P(car|edge)
        return torch.exp(logP) * otot[seg_id] * p_car   # car flow per edge

    # ---- node-pair structure + Delta on the smoke subnetwork ----
    net = np.load(HERE.parent / "data" / "processed" / "beijing_road_network.npz", allow_pickle=True)
    Nn = int(net["n_nodes"]); eu = net["edge_u"].astype(np.int64); ev = net["edge_v"].astype(np.int64)
    ff = net["edge_ff_min"].astype(np.float64); cap = net["edge_cap"].astype(np.float64)
    cell_node = net["cell_node"].astype(np.int64); hours = PERIOD_HOURS[period]
    onode = cell_node[o.cpu().numpy()]; dnode = cell_node[d.cpu().numpy()]
    key = onode.astype(np.int64) * Nn + dnode
    uk, inv = np.unique(key, return_inverse=True)
    pairs = [(int(k // Nn), int(k % Nn)) for k in uk]
    nodepair_of_edge = torch.tensor(inv)
    Delta_sp, has_path = build_delta_aligned(Nn, eu, ev, ff, pairs)
    print(f"[stageA] smoke {smoke_cells} cells -> {len(o):,} edges, {O} origins, "
          f"{len(pairs)} OD node-pairs ({int(has_path.sum())} with a road path); Delta nnz={Delta_sp.nnz}")
    Delta = scipy_to_torch_sparse(Delta_sp); DeltaT = Delta.t().coalesce()
    ffT = torch.tensor(ff); capT = torch.tensor(cap)
    has_path_t = torch.tensor(has_path)
    # fixed fallback car time for pairs with no road path (self/unreachable): flow-weighted obs car time
    tcar0 = data["t_car"][period].detach()
    fix_num = torch.zeros(len(pairs), dtype=tcar0.dtype).index_add(0, nodepair_of_edge, tcar0 * flow[:, period])
    fix_den = torch.zeros(len(pairs), dtype=tcar0.dtype).index_add(0, nodepair_of_edge, flow[:, period]).clamp_min(1e-9)
    fixed_tt = fix_num / fix_den

    # calibrate congestion scale at free-flow demand
    with torch.no_grad():
        tt0_np = torch.where(has_path_t, torch.sparse.mm(Delta, ffT.unsqueeze(1)).squeeze(1), fixed_tt)
        f0_edge = stage_a_car_flow(tt0_np[nodepair_of_edge].to(tcar0.dtype))
        f0_np = torch.zeros(len(pairs), dtype=torch.float64).index_add(0, nodepair_of_edge, f0_edge.double())
        x0 = torch.sparse.mm(DeltaT, f0_np.unsqueeze(1)).squeeze(1)
        vc0 = (x0 / (capT * hours + 1e-9)).numpy()
        scale = float(TARGET_VC / max(np.median(vc0[vc0 > 0]), 1e-6))
    print(f"[stageA] congestion scale = {scale:.2f}")

    def F(t_link):
        tt_np = torch.where(has_path_t, torch.sparse.mm(Delta, t_link.unsqueeze(1)).squeeze(1), fixed_tt)
        f_edge = stage_a_car_flow(tt_np[nodepair_of_edge].to(tcar0.dtype)).double()
        f_np = torch.zeros(len(pairs), dtype=torch.float64).index_add(0, nodepair_of_edge, f_edge)
        x = torch.sparse.mm(DeltaT, f_np.unsqueeze(1)).squeeze(1)
        return bpr(x, ffT, capT, hours, scale=scale)

    with torch.no_grad():
        t_star, it = solve_fixed_point(F, ffT.clone(), n_iter=n_iter, tol=1e-6, method="msa", verbose=True)
        rel = (F(t_star) - t_star).norm().item() / (t_star.norm().item() + 1e-9)
    used = t_star > ffT + 1e-9
    print(f"[stageA] REAL-demand equilibrium: {it} iters, residual={rel:.1e}; congested links {int(used.sum())}; "
          f"mean delay x{(t_star[used]/ffT[used]).mean().item():.3f}  max x{(t_star/ffT).max().item():.2f}")
    print("[stageA] real Stage-A demand is wired into the differentiable equilibrium. "
          "Gradient/estimation through the 21-class forward = AutoDL (reuse equilibrium_times implicit+GMRES).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="evaluation_outputs/v4_full_anchored_s0.pt")
    ap.add_argument("--period", type=int, default=0)
    ap.add_argument("--smoke-cells", type=int, default=400)
    ap.add_argument("--n-iter", type=int, default=400)
    a = ap.parse_args()
    run(str(HERE.parent / a.pt) if not pathlib.Path(a.pt).is_absolute() else a.pt,
        a.period, a.smoke_cells, a.n_iter)


if __name__ == "__main__":
    main()
