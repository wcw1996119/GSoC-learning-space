"""北京 v4 稀疏 trainer (Phase 3 + 2b 完整版, 支持 origin-chunking)

数据: data/processed/{beijing_edges,beijing_modes,beijing_aux,beijing_grid}.npz
模型: beijing_model.{BeijingNestedHead, BeijingPairEncoder}

损失: 4 时段加权 CE, 逐时段+逐chunk backward (省显存)。CPC: edge 级 Sørensen。
--origin-chunks G: 出发地切 G 块, 每块单独前向/backward (segment-softmax per-origin,
  CSR 边连续 -> 分块精确无近似)。显存降到 1/G, soc 混合(21类)必用。
功能 flag: --use-nn --use-consideration --use-soc-mixture --gnn-mode --no-self-loop
smoke: --smoke-cells K 取 jobs 前 K 格子子图。
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

    # 刷卡公交 OD -> 每 edge 观测 transit 量 (Ben-Akiva-Morikawa 联合估计锚)
    obs_transit = None
    if getattr(args, "anchor_transit", False):
        tod = np.load(PROC / "beijing_transit_od.npz")
        tk = tod["to_o"].astype(np.int64) * N + tod["to_d"].astype(np.int64)
        tv = tod["to_n"].astype(np.float64)
        order = np.argsort(tk); tk = tk[order]; tv = tv[order]
        ek = o.astype(np.int64) * N + d.astype(np.int64)
        pos = np.searchsorted(tk, ek)
        pos = np.clip(pos, 0, len(tk) - 1)
        hit = tk[pos] == ek
        obs_transit = np.where(hit, tv[pos], 0.0).astype(np.float32)
        print(f"  anchor: 刷卡公交命中 {int((obs_transit>0).sum()):,}/{len(o):,} edges, "
              f"覆盖公交出行 {obs_transit.sum()/tv.sum()*100:.0f}%")

    # CSR: 边已按 o 排序; seg_id + seg_starts (各出发地首边)
    origin_ids, seg_starts = np.unique(o, return_index=True)
    O = len(origin_ids)
    seg_id = np.repeat(np.arange(O), np.diff(np.append(seg_starts, len(o)))).astype(np.int64)
    seg_starts = np.append(seg_starts, len(o)).astype(np.int64)   # (O+1,)
    t_car = {0: t_obs, 1: t_obs, 2: (0.5*(t_obs+t_ff)).astype(np.float32), 3: t_ff}

    Xnode = np.stack([log_M, log_D, income, log_W, zlog(jobs), zlog(residents),
                      (xy[:,0]/1e5), (xy[:,1]/1e5)], axis=1).astype(np.float32)
    edge_index = None
    if args.use_nn:
        from scipy.spatial import cKDTree
        k = 8
        _, nn = cKDTree(xy).query(xy, k=k+1)
        src = np.repeat(np.arange(N), k); dst = nn[:, 1:].reshape(-1)
        edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0).astype(np.int64)

    dev = args.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    data = dict(
        o=to(o), d=to(d), seg_id=to(seg_id), seg_starts=seg_starts, num_seg=O, N=N,
        t_obs=to(t_obs), t_ff=to(t_ff), t_walk=to(t_walk), t_transit=to(t_transit),
        t_car={p: to(v) for p, v in t_car.items()}, log_d=to(log_d), match=to(match), flow=to(flow),
        log_M=to(log_M), log_W=to(log_W), log_D=to(log_D), income=to(income),
        tier_props=to(tier_props), district=to(district),
        soc_props=to(soc_props), demand_share=to(demand_share),
        Xnode=to(Xnode), edge_index=to(edge_index) if edge_index is not None else None,
        flow_tot=to(flow.sum(1)),
        obs_transit=to(obs_transit) if obs_transit is not None else None,
    )
    rng = np.random.RandomState(args.seed)
    val_seg = np.zeros(O, bool); val_seg[rng.permutation(O)[:int(O*args.val_frac)]] = True
    data["edge_is_val"] = to(val_seg[seg_id]); data["edge_is_train"] = to(~val_seg[seg_id])
    return data


def make_batch(data, period, use_soc, e0, e1, seg_base, n_seg):
    o = data["o"][e0:e1]; d = data["d"][e0:e1]
    t_car = data["t_car"][period][e0:e1]
    t_transit = data["t_transit"][e0:e1]; t_walk = data["t_walk"][e0:e1]
    t_min = torch.minimum(torch.minimum(t_car, t_transit), t_walk)
    b = dict(
        seg_id=data["seg_id"][e0:e1] - seg_base, num_seg=n_seg,
        t_car=t_car, t_transit=t_transit, t_walk=t_walk,
        log_d=data["log_d"][e0:e1], t_min=t_min,
        log_M_d=data["log_M"][d], log_W_d=data["log_W"][d], log_D_d=data["log_D"][d],
        match=data["match"][e0:e1], income_o=data["income"][o], tier_props_o=data["tier_props"][o],
        district_o=data["district"][o], is_self=(o == d), period=period,
    )
    if use_soc:
        b["soc_props_o"] = data["soc_props"][o]
        b["demand_share_d"] = data["demand_share"][d]
    return b


def chunk_ranges(seg_starts, O, G):
    """G 块出发地 -> [(e0,e1,seg_base,n_seg), ...]"""
    bounds = np.linspace(0, O, G + 1).astype(int)
    out = []
    for g in range(G):
        a, bb = bounds[g], bounds[g + 1]
        if bb <= a: continue
        out.append((int(seg_starts[a]), int(seg_starts[bb]), int(a), int(bb - a)))
    return out


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
    ap.add_argument("--typed-mass-occ", action="store_true",
                    help="引力作用在本职业岗位 M_j^o(不混总数, 需 soc-mixture + cell级demand)")
    ap.add_argument("--gnn-mode", default="residual")
    ap.add_argument("--residual-scale-init", type=float, default=0.1)
    ap.add_argument("--origin-chunks", type=int, default=1)
    ap.add_argument("--anchor-transit", action="store_true",
                    help="刷卡公交OD矩匹配(Ben-Akiva-Morikawa), 识别mode去向模式; 需 chunks=1")
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--anchor-share", action="store_true",
                    help="整体方式份额矩(北京交通发展年报通勤结构), 钉mode份额水平")
    ap.add_argument("--share-target", default="0.21,0.35,0.44",
                    help="目标 [车,公交,步行+自行车], 默认=2015 1%微观北京通勤 ground truth (车21/公交35/慢行44)")
    ap.add_argument("--share-weight", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "evaluation_outputs" / "v4_run.pt"))
    args = ap.parse_args()
    if args.anchor_transit or args.anchor_share:
        assert args.origin_chunks == 1, "--anchor-* 需 --origin-chunks 1 (全局归一化)"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    flags = [f for f, on in [("nn", args.use_nn), ("filter", args.use_consideration),
                             ("soc", args.use_soc_mixture), ("self_loop", not args.no_self_loop)] if on]
    print(f"device={args.device} seed={args.seed} epochs={args.epochs} feat=[{','.join(flags)}] "
          f"gnn={args.gnn_mode} chunks={args.origin_chunks}")
    t0 = time.time()
    data = build_data(args)
    O = data["num_seg"]
    chunks = chunk_ranges(data["seg_starts"], O, args.origin_chunks)
    print(f"  data built {time.time()-t0:.1f}s  edges={data['o'].shape[0]:,} origins={O:,} chunks={len(chunks)}")

    head = BeijingNestedHead(n_districts=16, use_self_loop=not args.no_self_loop,
                             use_consideration=args.use_consideration,
                             use_soc_mixture=args.use_soc_mixture,
                             gnn_mode=args.gnn_mode,
                             residual_scale_init=args.residual_scale_init,
                             use_typed_mass=args.typed_mass_occ).to(args.device)
    params = list(head.parameters())
    enc = None
    if args.use_nn:
        enc = BeijingPairEncoder(in_dim=data["Xnode"].shape[1]).to(args.device)
        params += list(enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    flow_all = data["flow"]; train_m = data["edge_is_train"]
    flow_train_sum = (flow_all * train_m.unsqueeze(1)).sum().clamp_min(1.0)
    best_cpc = -1.0
    n_back = len(chunks) * 4   # 总 backward 次数 (retain encoder 图到最后一次)

    for ep in range(args.epochs):
        head.train()
        if enc: enc.train()
        opt.zero_grad()
        e_o = e_d = None
        if enc is not None:
            e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"])
        loss_val = 0.0; bi = 0
        for (e0, e1, sb, ns) in chunks:
            for p in range(4):
                batch = make_batch(data, p, args.use_soc_mixture, e0, e1, sb, ns)
                v_nn = enc.edge_vnn(e_o, e_d, data["o"][e0:e1], data["d"][e0:e1]) if enc is not None else None
                logP = head(batch, v_nn=v_nn)
                loss_p = -(flow_all[e0:e1, p] * train_m[e0:e1] * logP).sum() / flow_train_sum
                # 刷卡公交锚 (Ben-Akiva-Morikawa): 模型预测公交分布 对齐 刷卡公交OD分布
                if (args.anchor_transit or args.anchor_share) and p == 0:
                    Pm = torch.softmax(head.mode_logits(batch), 0)            # (3,E)
                    ftot = data["flow_tot"]
                    if args.anchor_transit:
                        pred_tr = (Pm[1] * ftot).clamp_min(1e-12)
                        pred_dist = pred_tr / pred_tr.sum()
                        obs = data["obs_transit"]; obs_dist = obs / obs.sum()
                        loss_p = loss_p + args.anchor_weight * (-(obs_dist * torch.log(pred_dist)).sum())
                    if args.anchor_share:
                        ms = (Pm * ftot).sum(1) / ftot.sum()                  # (3,) 流量加权方式份额
                        tgt = torch.tensor([float(x) for x in args.share_target.split(",")],
                                           device=ms.device)
                        loss_p = loss_p + args.share_weight * (-(tgt * torch.log(ms.clamp_min(1e-9))).sum())
                bi += 1
                loss_p.backward(retain_graph=(enc is not None) and (bi < n_back))
                loss_val += loss_p.item()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            head.eval()
            if enc: enc.eval()
            num = [0.0]*4; den = [0.0]*4
            with torch.no_grad():
                e_o2 = e_d2 = None
                if enc is not None:
                    e_o2, e_d2 = enc.node_embed(data["Xnode"], data["edge_index"], data["N"])
                for (e0, e1, sb, ns) in chunks:
                    vm = data["edge_is_val"][e0:e1]
                    for p in range(4):
                        batch = make_batch(data, p, args.use_soc_mixture, e0, e1, sb, ns)
                        v_nn = enc.edge_vnn(e_o2, e_d2, data["o"][e0:e1], data["d"][e0:e1]) if enc is not None else None
                        logP = head(batch, v_nn=v_nn)
                        fp = flow_all[e0:e1, p]
                        otot = segment_sum(fp, batch["seg_id"], ns)
                        pred = logP.exp() * otot[batch["seg_id"]]
                        num[p] += float(torch.minimum(pred[vm], fp[vm]).sum())
                        den[p] += float((pred[vm].sum() + fp[vm].sum()))
            cpcs = [2*num[p]/max(den[p], 1.0) for p in range(4)]
            w = [float((flow_all[:, p] * data["edge_is_val"]).sum()) for p in range(4)]
            cpc_tot = sum(c*wi for c, wi in zip(cpcs, w)) / max(sum(w), 1)
            best_cpc = max(best_cpc, cpc_tot)
            pr = head.param_report()
            # 方式分担 (流量加权, ampeak)
            with torch.no_grad():
                b0 = make_batch(data, 0, args.use_soc_mixture, 0, len(data["o"]), 0, data["num_seg"])
                Pm = torch.softmax(head.mode_logits(b0), 0)            # (3,E)
                w_m = data["flow_tot"]; ms = (Pm * w_m).sum(1) / w_m.sum()
            extra = f" w_nn={pr.get('w_nn')}" if 'w_nn' in pr else ""
            extra += f" Tmax={pr.get('T_max')}" if 'T_max' in pr else ""
            print(f"ep {ep:3d}  loss {loss_val:.4f}  CPC {cpc_tot:.4f}  "
                  f"[{','.join(f'{c:.3f}' for c in cpcs)}]  mode[车{ms[0]:.2f}公交{ms[1]:.2f}步{ms[2]:.2f}] "
                  f"λ={pr['lambda']:.3f} γ={pr['gamma_M']} ν={pr['nu_D']}{extra}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"head_state": head.state_dict(), "args": vars(args),
               "param_report": head.param_report(), "best_cpc": best_cpc}
    if enc: payload["enc_state"] = enc.state_dict()
    torch.save(payload, args.out)
    print(f"[OK] saved {args.out}  best CPC {best_cpc:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
