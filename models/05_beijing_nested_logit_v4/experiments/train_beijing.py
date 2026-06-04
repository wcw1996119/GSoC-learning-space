"""北京 v4 稀疏 trainer (Phase 3 + 2b 完整版)

数据: data/processed/{beijing_edges,beijing_modes,beijing_aux,beijing_grid}.npz
模型: beijing_model.{BeijingNestedHead, BeijingPairEncoder}

损失: 4 时段加权 CE, 逐时段 backward (省显存)。CPC: edge 级 Sørensen。
功能 flag: --use-nn --use-consideration --use-soc-mixture --gnn-mode --no-self-loop
smoke: --smoke-cells K 取 jobs 前 K 格子子图 (本地 CPU 验证语义)。
"""
import argparse, time, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, BeijingPairEncoder

PROC = ROOT / "data" / "processed"
PERIOD_NAMES = ["ampeak", "pmpeak", "midday", "night"]


def segment_sum(values, seg_id, num_seg):
    return torch.zeros(num_seg, device=values.device, dtype=values.dtype).scatter_add(0, seg_id, values)


def zlog(x):
    x = np.log(np.asarray(x, np.float64) + 1.0)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def build_data(args):
    edges = np.load(PROC / "beijing_edges.npz")
    modes = np.load(PROC / "beijing_modes.npz")
    aux = np.load(PROC / "beijing_aux.npz")
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)

    o = edges["o_idx"].astype(np.int64); d = edges["d_idx"].astype(np.int64)
    N = int(edges["N"])
    t_obs = edges["t_obs"].astype(np.float32); t_ff = edges["t_ff"].astype(np.float32)
    log_d = edges["log_d"].astype(np.float32); flow = edges["flow"].astype(np.float32)
    t_walk = modes["t_walk"].astype(np.float32); t_transit = modes["t_transit"].astype(np.float32)

    log_M = aux["log_M_z"].astype(np.float32); log_W = aux["log_W_z"].astype(np.float32)
    log_D = aux["log_D_z"].astype(np.float32); income = aux["income_z"].astype(np.float32)
    tier_props = aux["income_tier_props"].astype(np.float32)
    match = aux["match"].astype(np.float32)
    soc_props = aux["soc_props"].astype(np.float32)
    demand_share = aux["demand_share"].astype(np.float32)
    district = grid["district_idx"].astype(np.int64)
    xy = grid["xy_m"].astype(np.float32)
    jobs = grid["jobs"]; residents = grid["residents"]

    if args.smoke_cells > 0:
        keep = np.zeros(N, bool); keep[np.argsort(-jobs)[:args.smoke_cells]] = True
        em = keep[o] & keep[d]
        o, d, t_obs, t_ff, log_d, match = o[em], d[em], t_obs[em], t_ff[em], log_d[em], match[em]
        t_walk, t_transit, flow = t_walk[em], t_transit[em], flow[em]
        print(f"  smoke: {args.smoke_cells} cells -> {em.sum():,} edges (of {len(em):,})")

    origin_ids, inv = np.unique(o, return_inverse=True)
    seg_id = inv.astype(np.int64); O = len(origin_ids)
    t_car = {0: t_obs, 1: t_obs, 2: (0.5*(t_obs+t_ff)).astype(np.float32), 3: t_ff}

    # NN 节点特征 (全 N 格) + kNN 图
    Xnode = np.stack([log_M, log_D, income, log_W, zlog(jobs), zlog(residents),
                      (xy[:,0]/1e5), (xy[:,1]/1e5)], axis=1).astype(np.float32)
    edge_index = None
    if args.use_nn:
        from scipy.spatial import cKDTree
        k = 8
        _, nn = cKDTree(xy).query(xy, k=k+1)   # 含自身
        src = np.repeat(np.arange(N), k); dst = nn[:, 1:].reshape(-1)
        ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)  # 双向
        edge_index = ei.astype(np.int64)

    dev = args.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    data = dict(
        o=to(o), d=to(d), seg_id=to(seg_id), num_seg=O, N=N,
        t_obs=to(t_obs), t_ff=to(t_ff), t_walk=to(t_walk), t_transit=to(t_transit),
        t_car={p: to(v) for p, v in t_car.items()}, log_d=to(log_d), match=to(match), flow=to(flow),
        log_M=to(log_M), log_W=to(log_W), log_D=to(log_D), income=to(income),
        tier_props=to(tier_props), district=to(district),
        soc_props=to(soc_props), demand_share=to(demand_share),
        Xnode=to(Xnode), edge_index=to(edge_index) if edge_index is not None else None,
    )
    rng = np.random.RandomState(args.seed)
    val_seg = np.zeros(O, bool); val_seg[rng.permutation(O)[:int(O*args.val_frac)]] = True
    data["edge_is_val"] = to(val_seg[seg_id]); data["edge_is_train"] = to(~val_seg[seg_id])
    return data


def make_batch(data, period, use_soc):
    o, d = data["o"], data["d"]
    t_car = data["t_car"][period]
    t_min = torch.minimum(torch.minimum(t_car, data["t_transit"]), data["t_walk"])
    b = dict(
        seg_id=data["seg_id"], num_seg=data["num_seg"],
        t_car=t_car, t_transit=data["t_transit"], t_walk=data["t_walk"],
        log_d=data["log_d"], t_min=t_min,
        log_M_d=data["log_M"][d], log_W_d=data["log_W"][d], log_D_d=data["log_D"][d],
        match=data["match"], income_o=data["income"][o], tier_props_o=data["tier_props"][o],
        district_o=data["district"][o], is_self=(o == d), period=period,
    )
    if use_soc:
        b["soc_props_o"] = data["soc_props"][o]
        b["demand_share_d"] = data["demand_share"][d]
    return b


def cpc_metric(logP, flow_p, seg_id, num_seg, mask):
    with torch.no_grad():
        origin_tot = segment_sum(flow_p, seg_id, num_seg)
        pred = logP.exp() * origin_tot[seg_id]
        num = 2.0 * torch.minimum(pred[mask], flow_p[mask]).sum()
        den = (pred[mask].sum() + flow_p[mask].sum()).clamp_min(1.0)
        return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--smoke-cells", type=int, default=0)
    ap.add_argument("--no-self-loop", action="store_true")
    ap.add_argument("--use-nn", action="store_true")
    ap.add_argument("--use-consideration", action="store_true")
    ap.add_argument("--use-soc-mixture", action="store_true")
    ap.add_argument("--gnn-mode", default="residual")
    ap.add_argument("--residual-scale-init", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "evaluation_outputs" / "v4_run.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    flags = [f for f, on in [("nn", args.use_nn), ("filter", args.use_consideration),
                             ("soc", args.use_soc_mixture), ("self_loop", not args.no_self_loop)] if on]
    print(f"device={args.device} seed={args.seed} epochs={args.epochs} feat=[{','.join(flags)}] gnn={args.gnn_mode}")
    t0 = time.time()
    data = build_data(args)
    print(f"  data built {time.time()-t0:.1f}s  edges={data['o'].shape[0]:,} origins={data['num_seg']:,}")

    head = BeijingNestedHead(n_districts=16, use_self_loop=not args.no_self_loop,
                             use_consideration=args.use_consideration,
                             use_soc_mixture=args.use_soc_mixture,
                             gnn_mode=args.gnn_mode,
                             residual_scale_init=args.residual_scale_init).to(args.device)
    params = list(head.parameters())
    enc = None
    if args.use_nn:
        enc = BeijingPairEncoder(in_dim=data["Xnode"].shape[1]).to(args.device)
        params += list(enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    flow_all = data["flow"]; train_m, val_m = data["edge_is_train"], data["edge_is_val"]
    flow_train_sum = (flow_all * train_m.unsqueeze(1)).sum().clamp_min(1.0)
    best_cpc = -1.0

    for ep in range(args.epochs):
        head.train()
        if enc: enc.train()
        opt.zero_grad()
        # NN 节点嵌入 (时段无关, 每 epoch 一次)
        e_o = e_d = None
        if enc is not None:
            e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"])
        loss_val = 0.0; logP_periods = []
        for p in range(4):
            batch = make_batch(data, p, args.use_soc_mixture)
            v_nn = enc.edge_vnn(e_o, e_d, data["o"], data["d"]) if enc is not None else None
            logP = head(batch, v_nn=v_nn)
            loss_p = -(flow_all[:, p] * train_m * logP).sum() / flow_train_sum
            loss_p.backward(retain_graph=(p < 3) and enc is not None)
            loss_val += loss_p.item()
            logP_periods.append(logP.detach())
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            head.eval()
            cpcs = [cpc_metric(logP_periods[p], flow_all[:, p], data["seg_id"], data["num_seg"], val_m)
                    for p in range(4)]
            w = [float((flow_all[:, p] * val_m).sum()) for p in range(4)]
            cpc_tot = sum(c*wi for c, wi in zip(cpcs, w)) / max(sum(w), 1)
            best_cpc = max(best_cpc, cpc_tot)
            pr = head.param_report()
            extra = f" w_nn={pr.get('w_nn')}" if 'w_nn' in pr else ""
            extra += f" Tmax={pr.get('T_max')}" if 'T_max' in pr else ""
            print(f"ep {ep:3d}  loss {loss_val:.4f}  CPC {cpc_tot:.4f}  "
                  f"[{','.join(f'{c:.3f}' for c in cpcs)}]  λ={pr['lambda']:.3f} "
                  f"γ={pr['gamma_M']} ν={pr['nu_D']} δ={pr['delta']}{extra}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"head_state": head.state_dict(), "args": vars(args),
               "param_report": head.param_report(), "best_cpc": best_cpc}
    if enc: payload["enc_state"] = enc.state_dict()
    torch.save(payload, args.out)
    print(f"[OK] saved {args.out}  best CPC {best_cpc:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
