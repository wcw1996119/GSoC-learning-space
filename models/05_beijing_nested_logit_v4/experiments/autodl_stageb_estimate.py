# -*- coding: utf-8 -*-
"""Stage B mechanistic-preference estimation (AutoDL / GPU).

Recovers the time-sensitivity coefficients (value of time) under ENDOGENOUS congestion
by Adam through the differentiable fixed-route equilibrium, with the real Stage-A
21-class nested-logit forward as demand. Compares the recovered (mechanistic) beta_t
against the Stage-A (empirical) beta_t -> the §5.8.4 simultaneity-bias test.

  loss = flow-weighted cross-entropy @ equilibrium  +  lambda_tt * || tt_eq - tt_obs ||^2
  gradient: implicit-function theorem through t* = F(t*), adjoint solved by GMRES.

Everything else (gravity, consideration thresholds, GNN residual) is FROZEN at Stage-A
values; only raw_beta_t0/raw_beta_t1 are updated, initialised from Stage A.

Run (AutoDL, single GPU):
  python experiments/autodl_stageb_estimate.py --pt evaluation_outputs/v4_full_anchored_s0.pt \
      --device cuda --smoke-cells 6000 --steps 40 --lr 0.02 --lambda-tt 1.0 \
      --msa-iter 200 --gmres-iter 40 --out evaluation_outputs/v4_mech_beta_s0.pt

Scale note: --smoke-cells bounds the congestion equilibrium to the top-N cells by
employment (a representative core subset) so the path-incidence Delta fits in memory.
Full-city (all 30.4M edges) needs the tree-cascade assignment (no Delta materialisation);
this script is the Delta-based v1. Validate cost at --smoke-cells 4000 first.
"""
import argparse, pathlib, sys, time
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
from diff_equilibrium_stagea import load_stage_a, build_delta_aligned
from diff_equilibrium import bpr, scipy_to_torch_sparse, solve_fixed_point, equilibrium_times, PERIOD_HOURS, TARGET_VC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="evaluation_outputs/v4_full_anchored_s0.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--period", type=int, default=0)
    ap.add_argument("--smoke-cells", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lambda-tt", type=float, default=1.0)
    ap.add_argument("--msa-iter", type=int, default=200)
    ap.add_argument("--gmres-iter", type=int, default=40)
    ap.add_argument("--also-beta-t1", action="store_true", help="also re-estimate the distance-VOT slope")
    ap.add_argument("--out", default="evaluation_outputs/v4_mech_beta.pt")
    a = ap.parse_args()
    dev = a.device
    pt_path = str(HERE.parent / a.pt) if not pathlib.Path(a.pt).is_absolute() else a.pt
    t0 = time.time()

    args, data, head, enc = load_stage_a(pt_path, a.smoke_cells, dev)
    use_soc = getattr(args, "use_soc_mixture", False)
    o = data["o"]; d = data["d"]; seg_id = data["seg_id"]; O = data["num_seg"]
    flow = data["flow"]; flow_p = flow[:, a.period]
    otot = torch.zeros(O, dtype=flow.dtype, device=dev).index_add(0, seg_id, flow_p)

    # ---- freeze everything; train only the value-of-time coefficients ----
    for p in head.parameters():
        p.requires_grad_(False)
    if enc is not None:
        for p in enc.parameters():
            p.requires_grad_(False)
    head.raw_beta_t0.requires_grad_(True)
    train_params = [head.raw_beta_t0]
    if a.also_beta_t1:
        head.raw_beta_t1.requires_grad_(True); train_params.append(head.raw_beta_t1)
    empirical_b0 = head.beta_t0.detach().clone()

    # frozen GNN residual per edge
    with torch.no_grad():
        if enc is not None:
            e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"],
                                      edge_weight=data.get("edge_weight"))
            v_nn = enc.edge_vnn(e_o, e_d, o, d).detach()
        else:
            v_nn = None

    def head_logP(t_car_edge):
        t_min = torch.minimum(torch.minimum(t_car_edge, data["t_transit"]), data["t_walk"])
        b = dict(seg_id=seg_id, num_seg=O, t_car=t_car_edge, t_min=t_min,
                 t_transit=data["t_transit"], t_walk=data["t_walk"], log_d=data["log_d"],
                 log_M_d=data["log_M"][d], log_W_d=data["log_W"][d], log_D_d=data["log_D"][d],
                 match=data["match"], income_o=data["income"][o], tier_props_o=data["tier_props"][o],
                 district_o=data["district"][o], is_self=(o == d), period=a.period)
        if use_soc:
            b["soc_props_o"] = data["soc_props"][o]; b["demand_share_d"] = data["demand_share"][d]
            if "occ_mask" in data: b["occ_mask_d"] = data["occ_mask"][d]
        logP = head(b, v_nn=v_nn)
        p_car = torch.softmax(head.mode_logits(b), 0)[0]
        return logP, p_car

    def car_flow(t_car_edge):
        logP, p_car = head_logP(t_car_edge)
        return torch.exp(logP) * otot[seg_id] * p_car

    # ---- node-pair structure + Delta on the road network ----
    net = np.load(HERE.parent / "data" / "processed" / "beijing_road_network.npz", allow_pickle=True)
    Nn = int(net["n_nodes"]); eu = net["edge_u"].astype(np.int64); ev = net["edge_v"].astype(np.int64)
    ff = net["edge_ff_min"].astype(np.float64); cap = net["edge_cap"].astype(np.float64)
    cell_node = net["cell_node"].astype(np.int64); hours = PERIOD_HOURS[a.period]
    onode = cell_node[o.cpu().numpy()]; dnode = cell_node[d.cpu().numpy()]
    uk, inv = np.unique(onode.astype(np.int64) * Nn + dnode, return_inverse=True)
    pairs = [(int(k // Nn), int(k % Nn)) for k in uk]
    nodepair_of_edge = torch.tensor(inv, device=dev)
    print(f"[est] building Delta on the road network for {len(pairs):,} OD node-pairs ...", flush=True)
    Delta_sp, has_path = build_delta_aligned(Nn, eu, ev, ff, pairs)
    Delta = scipy_to_torch_sparse(Delta_sp).to(dev); DeltaT = Delta.t().coalesce().to(dev)
    ffT = torch.tensor(ff, device=dev); capT = torch.tensor(cap, device=dev)
    has_path_t = torch.tensor(has_path, device=dev)
    tcar0 = data["t_car"][a.period].detach()
    fix_num = torch.zeros(len(pairs), dtype=tcar0.dtype, device=dev).index_add(0, nodepair_of_edge, tcar0 * flow_p)
    fix_den = torch.zeros(len(pairs), dtype=tcar0.dtype, device=dev).index_add(0, nodepair_of_edge, flow_p).clamp_min(1e-9)
    fixed_tt = (fix_num / fix_den).double()
    print(f"[est] {len(o):,} edges, {O} origins, {len(pairs):,} node-pairs "
          f"({int(has_path.sum()):,} with a road path); Delta nnz={Delta_sp.nnz:,}", flush=True)

    def edge_car_time(t_link):
        tt_np = torch.where(has_path_t, torch.sparse.mm(Delta, t_link.unsqueeze(1)).squeeze(1), fixed_tt)
        return tt_np, tt_np[nodepair_of_edge].to(tcar0.dtype)

    def F(t_link):
        _, tce = edge_car_time(t_link)
        f_edge = car_flow(tce).double()
        f_np = torch.zeros(len(pairs), dtype=torch.float64, device=dev).index_add(0, nodepair_of_edge, f_edge)
        x = torch.sparse.mm(DeltaT, f_np.unsqueeze(1)).squeeze(1)
        return bpr(x, ffT, capT, hours, scale=scale)

    with torch.no_grad():                                # fixed congestion scale (free-flow median V/C -> target)
        _, tce0 = edge_car_time(ffT.clone())
        f0 = car_flow(tce0).double()
        x0 = torch.sparse.mm(DeltaT, torch.zeros(len(pairs), dtype=torch.float64, device=dev)
                             .index_add(0, nodepair_of_edge, f0).unsqueeze(1)).squeeze(1)
        vc0 = (x0 / (capT * hours + 1e-9)).cpu().numpy()
        scale = float(TARGET_VC / max(np.median(vc0[vc0 > 0]), 1e-6))
    print(f"[est] congestion scale = {scale:.2f}; lambda_tt={a.lambda_tt}, lr={a.lr}, "
          f"msa={a.msa_iter}, gmres={a.gmres_iter}", flush=True)

    opt = torch.optim.Adam(train_params, lr=a.lr)
    warm = ffT.clone()
    traj = []
    for s in range(a.steps):
        opt.zero_grad()
        t_g = equilibrium_times(F, warm, grad_mode="implicit", method="msa",
                                n_iter=a.msa_iter, tol=1e-6, neumann=a.gmres_iter)
        tt_np, tce = edge_car_time(t_g)
        logP, _ = head_logP(tce)
        loss_flow = -(flow_p * logP).sum() / flow_p.sum().clamp_min(1.0)
        loss_tt = ((tt_np[has_path_t] - fixed_tt[has_path_t]) ** 2).mean()
        loss = loss_flow + a.lambda_tt * loss_tt
        loss.backward(); opt.step()
        warm = t_g.detach()
        lf, lt = loss_flow.item(), loss_tt.item()
        b0 = head.beta_t0.detach()
        traj.append((lf, lt, b0.cpu().tolist()))
        print(f"  step {s:2d}  loss_flow={lf:.4f}  loss_tt={lt:.3e}  "
              f"beta_t0(car/tr/walk)={[round(x,4) for x in b0.cpu().tolist()]}", flush=True)

    mech_b0 = head.beta_t0.detach().cpu().tolist()
    emp_b0 = empirical_b0.cpu().tolist()
    print("[est] === mechanistic vs empirical value-of-time (beta_t0) ===")
    for m, lab in enumerate(["car", "transit", "walk"]):
        print(f"   {lab:8s} empirical {emp_b0[m]:+.4f} -> mechanistic {mech_b0[m]:+.4f}  "
              f"(Δ {100*(mech_b0[m]-emp_b0[m])/abs(emp_b0[m]+1e-9):+.1f}%)")
    out = str(HERE.parent / a.out) if not pathlib.Path(a.out).is_absolute() else a.out
    torch.save(dict(mechanistic_beta_t0=mech_b0, empirical_beta_t0=emp_b0,
                    mechanistic_beta_t1=head.beta_t1.detach().cpu().tolist(),
                    traj=traj, args=vars(a), smoke_cells=a.smoke_cells), out)
    print(f"[est] saved {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
