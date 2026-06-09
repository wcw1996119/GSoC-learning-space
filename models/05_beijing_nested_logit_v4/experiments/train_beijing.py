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
    if getattr(args, "use_education", False):   # 教育替收入: 异质维度换成教育档(信号干净可锚)
        tier_props = np.load(PROC / "beijing_edu_hukou.npz")["education_tier_props"].astype(np.float32)
    match = aux["match"].astype(np.float32)
    soc_props = aux["soc_props"].astype(np.float32)
    demand_share = aux["demand_share"].astype(np.float32)
    occ_mask = (np.load(PROC / "beijing_occ_mask.npz")["occ_mask"].astype(np.float32)
                if getattr(args, "use_frozen_occ_mask", False) else None)
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

    # 微观 方式×距离 矩 (2015 1% 个体 mode+工作区 -> 区中心距离档分担; 钉 β_t 距离梯度)
    modedist_band = None; modedist_target = None
    if getattr(args, "anchor_mode_dist", False):
        mm = np.load(PROC / "beijing_micro_moments.npz")
        cent = mm["district_centroids"].astype(np.float64)        # (16,2) m, 跟 grid 16 区同序
        dist_ub = mm["dist_bands"].astype(np.float64)             # 上界 km, 如 [1,5,15,100]
        modedist_target = mm["mode_by_dist"].astype(np.float32)   # (nbands,3) [车,公交,步]
        dd = np.linalg.norm(cent[district[o]] - cent[district[d]], axis=1) / 1000.0
        modedist_band = np.clip(np.searchsorted(dist_ub, dd, side="right"),
                                0, len(dist_ub) - 1).astype(np.int64)
        t = modedist_target
        print(f"  anchor mode×dist: {len(dist_ub)}档 target "
              f"近[车{t[0,0]:.2f}公交{t[0,1]:.2f}步{t[0,2]:.2f}] 远[车{t[-1,0]:.2f}公交{t[-1,1]:.2f}步{t[-1,2]:.2f}]")

    # CSR: 边已按 o 排序; seg_id + seg_starts (各出发地首边)
    origin_ids, seg_starts = np.unique(o, return_index=True)
    O = len(origin_ids)
    seg_id = np.repeat(np.arange(O), np.diff(np.append(seg_starts, len(o)))).astype(np.int64)
    seg_starts = np.append(seg_starts, len(o)).astype(np.int64)   # (O+1,)
    t_car = {0: t_obs, 1: t_obs, 2: (0.5*(t_obs+t_ff)).astype(np.float32), 3: t_ff}

    # NN 静态特征; --nn-exclude-jobs: 去掉岗位(log_M + zlog jobs), 让 NN 不见政策杠杆 -> 反事实不外推
    if getattr(args, "nn_exclude_jobs", False):
        nn_feats = [log_D, income, log_W, zlog(residents), (xy[:,0]/1e5), (xy[:,1]/1e5)]
    else:
        nn_feats = [log_M, log_D, income, log_W, zlog(jobs), zlog(residents), (xy[:,0]/1e5), (xy[:,1]/1e5)]
    Xnode = np.stack(nn_feats, axis=1).astype(np.float32)
    edge_index = None; edge_weight = None
    if args.use_nn:
        if getattr(args, "road_graph", False):
            rg = np.load(PROC / "beijing_road_graph.npz")
            edge_index = rg["edge_index"].astype(np.int64)
            edge_weight = np.log1p(rg["edge_weight"].astype(np.float32))   # log 压缩边权(跨界路段数)
            print(f"  road-graph: {edge_index.shape[1]:,} 有向边, 加权 log1p(跨界路段数)")
        else:
            from scipy.spatial import cKDTree
            k = 8
            _, nn = cKDTree(xy).query(xy, k=k+1)
            src = np.repeat(np.arange(N), k); dst = nn[:, 1:].reshape(-1)
            edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0).astype(np.int64)

    # 动态时空特征 (双路 ST-GNN 动态腿): 逐小时 pop + 拥堵 -> Xdyn (24,N,Fd)
    Xdyn = None; hour2period = None
    if getattr(args, "use_dynamic", False):
        pop = np.load(PROC / "beijing_dynamic_pop.npz")["pop_hourly"].astype(np.float64)   # (N,24)
        cg = np.load(PROC / "beijing_dynamic_cong.npz")["cong_frac_hourly"].astype(np.float32)  # (N,24)
        pop_z = ((np.log(pop + 1.0) - np.log(pop + 1.0).mean()) /
                 (np.log(pop + 1.0).std() + 1e-9)).astype(np.float32)   # 全局 z(留时序变化)
        Xdyn = np.stack([pop_z.T, cg.T], axis=-1).astype(np.float32)    # (24,N,2)
        h2p = np.array([0 if 7 <= h <= 9 else 1 if 17 <= h <= 19 else 2 if 10 <= h <= 16 else 3
                        for h in range(24)], dtype=np.int64)
        hour2period = h2p
        print(f"  dynamic: Xdyn {Xdyn.shape} (pop+拥堵逐小时), hour->period {h2p.tolist()}")

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
        edge_weight=to(edge_weight) if edge_weight is not None else None,
        Xdyn=to(Xdyn) if Xdyn is not None else None,
        hour2period=to(hour2period) if hour2period is not None else None,
        flow_tot=to(flow.sum(1)),
        obs_transit=to(obs_transit) if obs_transit is not None else None,
        modedist_band=to(modedist_band) if modedist_band is not None else None,
        modedist_target=(torch.from_numpy(modedist_target).to(dev)
                         if modedist_target is not None else None),
    )
    if occ_mask is not None:
        data["occ_mask"] = to(occ_mask)            # (N,7) 外生硬职业 mask
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
        if "occ_mask" in data:
            b["occ_mask_d"] = data["occ_mask"][d]
    return b


def make_batch_idx(data, period, use_soc, idx, seg_id, n_seg):
    """像 make_batch 但用任意边索引 idx (非连续, 给 origin-sampled 矩用); seg_id 已 0..K-1。"""
    o = data["o"][idx]; d = data["d"][idx]
    t_car = data["t_car"][period][idx]
    t_transit = data["t_transit"][idx]; t_walk = data["t_walk"][idx]
    t_min = torch.minimum(torch.minimum(t_car, t_transit), t_walk)
    b = dict(
        seg_id=seg_id, num_seg=n_seg, t_car=t_car, t_transit=t_transit, t_walk=t_walk,
        log_d=data["log_d"][idx], t_min=t_min,
        log_M_d=data["log_M"][d], log_W_d=data["log_W"][d], log_D_d=data["log_D"][d],
        match=data["match"][idx], income_o=data["income"][o], tier_props_o=data["tier_props"][o],
        district_o=data["district"][o], is_self=(o == d), period=period,
    )
    if use_soc:
        b["soc_props_o"] = data["soc_props"][o]
        b["demand_share_d"] = data["demand_share"][d]
        if "occ_mask" in data:
            b["occ_mask_d"] = data["occ_mask"][d]
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
    ap.add_argument("--use-dynamic", action="store_true",
                    help="双路 ST-GNN 动态腿(逐小时 pop+拥堵 GRU/TCN); 需 beijing_dynamic_{pop,cong}.npz + --use-nn")
    ap.add_argument("--road-graph", action="store_true",
                    help="GNN 图用路网连通图(加权)替代空间kNN; 需 beijing_road_graph.npz")
    ap.add_argument("--nn-exclude-jobs", action="store_true",
                    help="NN 输入去掉岗位特征(log_M+zlog jobs), RUM 仍用岗位 -> 反事实时 NN 不外推政策杠杆")
    ap.add_argument("--use-consideration", action="store_true")
    ap.add_argument("--use-soc-mixture", action="store_true")
    ap.add_argument("--typed-mass-occ", action="store_true",
                    help="引力作用在本职业岗位 M_j^o(不混总数, 需 soc-mixture + cell级demand)")
    ap.add_argument("--no-match-filter", action="store_true",
                    help="ablation: 关掉 L1 词典筛选职业匹配门(保留 L2 成本/L3 时间门), 测匹配门单独贡献")
    ap.add_argument("--use-frozen-occ-mask", action="store_true",
                    help="外生硬职业 mask(非补偿考虑集筛选, 需 beijing_occ_mask.npz + soc-mixture); 建议配 --no-match-filter")
    ap.add_argument("--soft-lex-match", action="store_true",
                    help="soft-lexicographic 职业门(伦敦 winning 设计): τ_s≥floor + k_s≥k_min, 防塌陷防糊, 自动学每职业筛多少")
    ap.add_argument("--tau-match-floor-mult", type=float, default=0.5,
                    help="τ_s ≥ mult × mean(demand_share[:,s]) (soft-lex floor)")
    ap.add_argument("--k-match-min", type=float, default=20.0,
                    help="k_s 锐度下限 (soft-lex, 防 sigmoid 糊成不筛)")
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
    ap.add_argument("--anchor-mode-dist", action="store_true",
                    help="微观 方式×距离 矩(钉 β_t 距离梯度: 短途步行/长途公交); 需 beijing_micro_moments.npz")
    ap.add_argument("--mode-dist-weight", type=float, default=2.0)
    ap.add_argument("--use-education", action="store_true",
                    help="教育替收入: 异质档=教育(信号干净可锚), 锚=教育通勤[20.8,33.6,44.4]; 需 beijing_edu_hukou.npz")
    ap.add_argument("--anchor-income-time", action="store_true",
                    help="CFPS 真收入×通勤时间矩(钉收入档拆分 γ_M/T_max); 需 beijing_cfps_income_moment.npz")
    ap.add_argument("--income-time-weight", type=float, default=2.0)
    ap.add_argument("--income-sample-origins", type=int, default=2500)
    ap.add_argument("--out", default=str(ROOT / "evaluation_outputs" / "v4_run.pt"))
    ap.add_argument("--anchor-sample", type=int, default=2000000,
                    help="锚在多少条边的采样上算(脱离chunks=1, 兼容chunking/24GB)")
    args = ap.parse_args()

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

    # 锚采样 (脱离 chunks=1: 在固定边采样上算 mode 锚, 兼容 chunking)
    anchor_on = (args.anchor_transit or args.anchor_share or args.anchor_mode_dist
                 or args.anchor_income_time)
    asb = None
    if anchor_on:
        E = data["o"].shape[0]
        si = torch.randperm(E, device="cpu")[:min(args.anchor_sample, E)].to(args.device)
        o_s, d_s = data["o"][si], data["d"][si]
        asb = dict(
            si=si, log_d=data["log_d"][si], income_o=data["income"][o_s],
            t_car=data["t_car"][0][si], t_transit=data["t_transit"][si], t_walk=data["t_walk"][si],
            flow_tot=data["flow_tot"][si],
            obs_transit=(data["obs_transit"][si] if data.get("obs_transit") is not None else None),
            tgt=torch.tensor([float(x) for x in args.share_target.split(",")], device=args.device),
        )
        if data.get("modedist_band") is not None:
            asb["modedist_band"] = data["modedist_band"][si]
            asb["modedist_target"] = data["modedist_target"]
        print(f"  anchor 采样 {len(si):,} 边 (transit={args.anchor_transit} share={args.anchor_share} "
              f"mode_dist={args.anchor_mode_dist})")

    # CFPS 收入×通勤时间矩: 采样 K 个出发地(全候选集), origin-sampled forward
    inc = None
    if args.anchor_income_time:
        ss = data["seg_starts"]   # numpy (O+1,)
        rng2 = np.random.RandomState(args.seed + 7)
        Ko = min(args.income_sample_origins, O)
        oids = rng2.choice(O, Ko, replace=False)
        idx_l, seg_l = [], []
        for k, oi in enumerate(oids):
            a, bb = int(ss[oi]), int(ss[oi + 1])
            idx_l.append(np.arange(a, bb)); seg_l.append(np.full(bb - a, k, np.int64))
        inc_idx = torch.from_numpy(np.concatenate(idx_l)).to(args.device)
        inc_seg = torch.from_numpy(np.concatenate(seg_l)).to(args.device)
        if getattr(args, "use_education", False):   # 锚换成教育通勤目标[20.8,33.6,44.4]
            tgt = np.load(PROC / "beijing_edu_hukou.npz")["ct_by_edu"]
        else:
            tgt = np.load(PROC / "beijing_cfps_income_moment.npz")["target_ct"]
        inc = dict(idx=inc_idx, seg=inc_seg, nseg=Ko,
                   target=torch.tensor(tgt, device=args.device),
                   oi=data["o"][inc_idx], di=data["d"][inc_idx])
        print(f"  {'edu' if getattr(args,'use_education',False) else 'income'}-time 矩: 采样 {Ko} 出发地 → {len(inc_idx):,} 边, "
              f"目标通勤[低,中,高]={[round(float(x),1) for x in tgt]}min")

    head = BeijingNestedHead(n_districts=16, use_self_loop=not args.no_self_loop,
                             use_consideration=args.use_consideration,
                             use_soc_mixture=args.use_soc_mixture,
                             gnn_mode=args.gnn_mode,
                             residual_scale_init=args.residual_scale_init,
                             use_typed_mass=args.typed_mass_occ,
                             use_match_filter=not args.no_match_filter,
                             use_frozen_occ_mask=args.use_frozen_occ_mask,
                             soft_lex_match=args.soft_lex_match,
                             k_match_min=args.k_match_min).to(args.device)
    if args.soft_lex_match and args.use_soc_mixture:
        floor = args.tau_match_floor_mult * data["demand_share"].mean(0)   # (S,) floor = mult×mean
        head.set_tau_match_floor(floor)
        print(f"  soft-lex match: k_min={args.k_match_min}, "
              f"τ floor = {[round(float(x),4) for x in floor]}")
    params = list(head.parameters())
    enc = None; dyn = args.use_dynamic
    if args.use_nn:
        if dyn:
            from beijing_model import BeijingDualBranchEncoder
            enc = BeijingDualBranchEncoder(static_dim=data["Xnode"].shape[1],
                                           dyn_dim=data["Xdyn"].shape[2]).to(args.device)
        else:
            enc = BeijingPairEncoder(in_dim=data["Xnode"].shape[1]).to(args.device)
        params += list(enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    def embed():
        if enc is None: return None, None
        ew = data.get("edge_weight")
        if dyn:
            return enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"],
                                  data["N"], data["hour2period"], edge_weight=ew)
        return enc.node_embed(data["Xnode"], data["edge_index"], data["N"], edge_weight=ew)

    def vnn(e_o, e_d, p, e0, e1):
        if enc is None: return None
        o_c, d_c = data["o"][e0:e1], data["d"][e0:e1]
        return enc.edge_vnn(e_o, e_d, p, o_c, d_c) if dyn else enc.edge_vnn(e_o, e_d, o_c, d_c)

    flow_all = data["flow"]; train_m = data["edge_is_train"]
    flow_train_sum = (flow_all * train_m.unsqueeze(1)).sum().clamp_min(1.0)
    best_cpc = -1.0
    n_back = len(chunks) * 4   # 总 backward 次数 (retain encoder 图到最后一次)

    for ep in range(args.epochs):
        head.train()
        if enc: enc.train()
        opt.zero_grad()
        e_o, e_d = embed()
        loss_val = 0.0; bi = 0
        for (e0, e1, sb, ns) in chunks:
            for p in range(4):
                batch = make_batch(data, p, args.use_soc_mixture, e0, e1, sb, ns)
                v_nn = vnn(e_o, e_d, p, e0, e1)
                logP = head(batch, v_nn=v_nn)
                loss_p = -(flow_all[e0:e1, p] * train_m[e0:e1] * logP).sum() / flow_train_sum
                bi += 1
                # 锚是独立 forward(asb)独立图, 不需保留 chunk 图; 只为 encoder(跨chunk共享)retain
                loss_p.backward(retain_graph=(enc is not None) and (bi < n_back))
                loss_val += loss_p.item()
        # 锚 (采样, Ben-Akiva-Morikawa): mode_logits 不依赖 NN/segment, 单独 backward
        if anchor_on:
            ml = head.mode_logits(asb)                       # (3, n_sample)
            Pm = torch.softmax(ml, 0)
            a_loss = 0.0
            if args.anchor_transit and asb["obs_transit"] is not None:
                pred = (Pm[1] * asb["flow_tot"]).clamp_min(1e-12); pred = pred / pred.sum()
                obs = asb["obs_transit"]; obs = obs / obs.sum().clamp_min(1e-9)
                a_loss = a_loss + args.anchor_weight * (-(obs * torch.log(pred)).sum())
            if args.anchor_share:
                ms = (Pm * asb["flow_tot"]).sum(1) / asb["flow_tot"].sum()
                a_loss = a_loss + args.share_weight * (-(asb["tgt"] * torch.log(ms.clamp_min(1e-9))).sum())
            if args.anchor_mode_dist and asb.get("modedist_band") is not None:
                band = asb["modedist_band"]; tgt = asb["modedist_target"]; w = asb["flow_tot"]
                md_loss = 0.0
                for b in range(tgt.shape[0]):
                    m = (band == b)
                    if m.any():
                        wm = w[m]
                        sh = (Pm[:, m] * wm).sum(1) / wm.sum().clamp_min(1e-9)   # (3,) 流量加权方式分担
                        md_loss = md_loss - (tgt[b] * torch.log(sh.clamp_min(1e-9))).sum()
                a_loss = a_loss + args.mode_dist_weight * md_loss
            # CFPS 收入×通勤时间矩 (origin-sampled, 需 segment softmax -> 单独 forward)
            if args.anchor_income_time and inc is not None:
                bi = make_batch_idx(data, 0, args.use_soc_mixture, inc["idx"], inc["seg"], inc["nseg"])
                if enc is not None:
                    vnn_i = (enc.edge_vnn(e_o, e_d, 0, inc["oi"], inc["di"]) if dyn
                             else enc.edge_vnn(e_o, e_d, inc["oi"], inc["di"])).detach()
                else:
                    vnn_i = None
                logP_i = head(bi, v_nn=vnn_i)
                otot_i = segment_sum(flow_all[inc["idx"], 0], bi["seg_id"], inc["nseg"])
                pred_i = logP_i.exp() * otot_i[bi["seg_id"]]      # 预测流 (period 0)
                tmin_i = bi["t_min"]; tpr_i = bi["tier_props_o"]
                it_loss = 0.0
                for T in range(3):
                    wT = pred_i * tpr_i[:, T]                     # 收入档 T 权重的预测流
                    mct = (wT * tmin_i).sum() / wT.sum().clamp_min(1e-6)   # 档 T 预测平均通勤时间
                    it_loss = it_loss + ((mct - inc["target"][T]) / inc["target"][T]) ** 2
                a_loss = a_loss + args.income_time_weight * it_loss
            a_loss.backward()
            loss_val += float(a_loss)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            head.eval()
            if enc: enc.eval()
            num = [0.0]*4; den = [0.0]*4
            with torch.no_grad():
                e_o2, e_d2 = embed()
                for (e0, e1, sb, ns) in chunks:
                    vm = data["edge_is_val"][e0:e1]
                    for p in range(4):
                        batch = make_batch(data, p, args.use_soc_mixture, e0, e1, sb, ns)
                        v_nn = vnn(e_o2, e_d2, p, e0, e1)
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
