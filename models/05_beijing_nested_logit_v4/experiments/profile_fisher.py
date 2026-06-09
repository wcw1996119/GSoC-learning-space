"""Fisher 信息谱 = 模型级可识别性 / 认知置信指标 (trustworthy ML).

加载训练好的 .pt, 对【结构行为参数】(β_t/asc/θ_inc/λ/α_W/γ_M/ν_D/δ/T_max,
排除 NN encoder 与 self_loop) 算观测 Fisher(OD-NLL 的 Hessian), 转到【有效系数空间】
(消 softplus/sigmoid 重参数化的尺度扭曲), 报:
  - 特征谱 (log10): 大=硬(被数据钉住) / 近零=软(不可识别方向)  [sloppy models, Transtrum-Sethna 2015]
  - 条件数 λ_max/λ_min, 有效可识别维数 (阈值秩 + 熵维数)
  - 逐参数条件 Fisher (对角, 越大越识别)
  - 最软的几个特征向量 = "哪些参数组合识别不出"
NN 残差 (若有) 作为固定 offset 进 forward, 即"给定 NN 下结构参数的可识别性"。

用法: python experiments/profile_fisher.py --pt evaluation_outputs/v4_full_anchored_s0.pt --cells 1500
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, BeijingPairEncoder
from train_beijing import build_data, make_batch, make_batch_idx, chunk_ranges, segment_sum
PROC = ROOT / "data" / "processed"

# 选中的结构参数: (head 属性名, 有效变换类型, 标签前缀)
PARAM_SPEC = [
    ("raw_beta_t0", "negsoftplus", "β_t0"),
    ("raw_beta_t1", "negsoftplus", "β_t1"),
    ("asc",         "identity",    "asc"),
    ("theta_inc",   "identity",    "θ_inc"),
    ("raw_lambda",  "lambda",      "λ"),
    ("raw_alpha_W", "softplus",    "α_W"),
    ("raw_gamma_M", "softplus",    "γ_M"),
    ("raw_nu_D",    "negsoftplus", "ν_D"),
    ("raw_delta",   "softplus",    "δ"),
    ("raw_T_max",   "softplus",    "T_max"),   # 仅 consideration 时存在
]


def abs_jacobian(kind, r):
    """|dθ_eff/dθ_raw| 在 raw 值 r 处 (转到有效系数空间用)。"""
    s = torch.sigmoid(r)
    if kind == "identity":    return torch.ones_like(r)
    if kind == "softplus":    return s                       # d softplus = sigmoid
    if kind == "negsoftplus": return s                       # 取绝对值
    if kind == "lambda":      return (0.95 * s * (1 - s)).abs()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "evaluation_outputs" / "v4_full_anchored_s0.pt"))
    ap.add_argument("--cells", type=int, default=1500, help="0=全量(慎,内存大)")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--with-moments", action="store_true",
                    help="Fisher 算完整目标(OD-NLL + 训练用的所有矩), 看矩给参数加的曲率(真识别性)")
    args = ap.parse_args()
    torch.manual_seed(0)
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    a = ck["args"]
    print(f"=== Fisher 可识别性谱: {Path(args.pt).name} (CPC {ck['best_cpc']:.4f}) ===")

    dyn = a.get("use_dynamic", False)
    wm = args.with_moments
    da = SimpleNamespace(smoke_cells=args.cells, use_nn=a.get("use_nn", False),
                         seed=a["seed"], val_frac=0.0, device="cpu",
                         use_dynamic=dyn, road_graph=a.get("road_graph", False),
                         anchor_transit=wm and a.get("anchor_transit", False),
                         anchor_mode_dist=wm and a.get("anchor_mode_dist", False))
    data = build_data(da)
    use_soc = a.get("use_soc_mixture", False)
    head = BeijingNestedHead(n_districts=16, use_self_loop=not a.get("no_self_loop", False),
                             use_consideration=a.get("use_consideration", False),
                             use_soc_mixture=use_soc, gnn_mode=a.get("gnn_mode", "residual"),
                             use_typed_mass=a.get("typed_mass_occ", False))
    head.load_state_dict(ck["head_state"]); head.eval()
    enc = None
    if a.get("use_nn", False):
        if dyn:
            from beijing_model import BeijingDualBranchEncoder
            enc = BeijingDualBranchEncoder(static_dim=data["Xnode"].shape[1],
                                           dyn_dim=data["Xdyn"].shape[2])
        else:
            enc = BeijingPairEncoder(in_dim=data["Xnode"].shape[1])
        enc.load_state_dict(ck["enc_state"]); enc.eval()

    # NN 残差固定 offset (给定 NN 下结构参数识别性); 双路 -> 逐时段
    vnn_p = None   # [4] 各 (E,) 或 None
    if enc is not None:
        with torch.no_grad():
            ew = data.get("edge_weight")
            if dyn:
                e_o, e_d = enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"],
                                          data["N"], data["hour2period"], edge_weight=ew)
                vnn_p = [enc.edge_vnn(e_o, e_d, p, data["o"], data["d"]).detach() for p in range(4)]
            else:
                e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"], edge_weight=ew)
                vf = enc.edge_vnn(e_o, e_d, data["o"], data["d"]).detach()
                vnn_p = [vf, vf, vf, vf]

    # 选中的 leaf 参数 (存在的), 按 tensor 分组保留标签/变换
    groups = []   # (tensor, kind, [per-elem labels])
    for name, kind, pre in PARAM_SPEC:
        if not hasattr(head, name): continue
        p = getattr(head, name); n = p.numel()
        labs = [pre] if n == 1 else [f"{pre}[{i}]" for i in range(n)]
        groups.append((p, kind, labs))

    O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, args.chunks)
    flow = data["flow"]; fs = flow.sum().clamp_min(1.0)

    # 含矩: 重建训练用的矩 (mode 锚 on 采样边 + income 矩 on 采样出发地)
    mom = None
    if args.with_moments:
        E = len(data["o"]); rng = np.random.RandomState(a["seed"])
        si = torch.from_numpy(rng.permutation(E)[:min(a.get("anchor_sample", 2000000), E)])
        o_s = data["o"][si]
        asb = dict(log_d=data["log_d"][si], income_o=data["income"][o_s],
                   t_car=data["t_car"][0][si], t_transit=data["t_transit"][si], t_walk=data["t_walk"][si],
                   flow_tot=data["flow_tot"][si],
                   obs_transit=(data["obs_transit"][si] if data.get("obs_transit") is not None else None),
                   tgt=torch.tensor([float(x) for x in a.get("share_target", "0.21,0.35,0.44").split(",")]))
        if data.get("modedist_band") is not None:
            asb["modedist_band"] = data["modedist_band"][si]; asb["modedist_target"] = data["modedist_target"]
        inc = None
        if a.get("anchor_income_time", False):
            ss = data["seg_starts"]; rng2 = np.random.RandomState(a["seed"] + 7)
            Ko = min(a.get("income_sample_origins", 2500), O); oids = rng2.choice(O, Ko, replace=False)
            idx_l, seg_l = [], []
            for k, oi in enumerate(oids):
                aa, bb = int(ss[oi]), int(ss[oi + 1])
                idx_l.append(np.arange(aa, bb)); seg_l.append(np.full(bb - aa, k, np.int64))
            ii = torch.from_numpy(np.concatenate(idx_l)); sg = torch.from_numpy(np.concatenate(seg_l))
            mm = np.load(PROC / "beijing_cfps_income_moment.npz")
            inc = dict(idx=ii, seg=sg, nseg=Ko, target=torch.tensor(mm["target_ct"]),
                       oi=data["o"][ii], di=data["d"][ii])
        mom = dict(asb=asb, inc=inc)
        print(f"  含矩: transit={a.get('anchor_transit')} share={a.get('anchor_share')} "
              f"mode_dist={a.get('anchor_mode_dist')} income={a.get('anchor_income_time')}")

    def moment_loss():
        asb = mom["asb"]; Pm = torch.softmax(head.mode_logits(asb), 0); L = 0.0
        if a.get("anchor_transit", False) and asb["obs_transit"] is not None:
            pred = (Pm[1] * asb["flow_tot"]).clamp_min(1e-12); pred = pred / pred.sum()
            obs = asb["obs_transit"]; obs = obs / obs.sum().clamp_min(1e-9)
            L = L + a.get("anchor_weight", 1.0) * (-(obs * torch.log(pred)).sum())
        if a.get("anchor_share", False):
            ms = (Pm * asb["flow_tot"]).sum(1) / asb["flow_tot"].sum()
            L = L + a.get("share_weight", 2.0) * (-(asb["tgt"] * torch.log(ms.clamp_min(1e-9))).sum())
        if a.get("anchor_mode_dist", False) and asb.get("modedist_band") is not None:
            band = asb["modedist_band"]; tgt = asb["modedist_target"]; w = asb["flow_tot"]
            for bb in range(tgt.shape[0]):
                m = (band == bb)
                if m.any():
                    wm = w[m]; sh = (Pm[:, m] * wm).sum(1) / wm.sum().clamp_min(1e-9)
                    L = L + a.get("mode_dist_weight", 2.0) * (-(tgt[bb] * torch.log(sh.clamp_min(1e-9))).sum())
        if mom["inc"] is not None:
            inc = mom["inc"]; bi = make_batch_idx(data, 0, use_soc, inc["idx"], inc["seg"], inc["nseg"])
            vi = vnn_p[0][inc["idx"]] if vnn_p is not None else None
            logP_i = head(bi, v_nn=vi); otot = segment_sum(flow[inc["idx"], 0], bi["seg_id"], inc["nseg"])
            pred = logP_i.exp() * otot[bi["seg_id"]]; tmin = bi["t_min"]; tpr = bi["tier_props_o"]
            for T in range(3):
                wT = pred * tpr[:, T]; mct = (wT * tmin).sum() / wT.sum().clamp_min(1e-6)
                L = L + a.get("income_time_weight", 2.0) * ((mct - inc["target"][T]) / inc["target"][T]) ** 2
        return L

    def nll():
        tot = 0.0
        for (e0, e1, sb, ns) in chunks:
            for p in range(4):
                vnn = vnn_p[p][e0:e1] if vnn_p is not None else None
                b = make_batch(data, p, use_soc, e0, e1, sb, ns)
                logP = head(b, v_nn=vnn)
                tot = tot - (flow[e0:e1, p] * logP).sum() / fs
        if mom is not None:
            tot = tot + moment_loss()       # 含矩: 加训练用的矩项 -> Hessian 含矩曲率
        return tot

    # 先探测哪些参数没进 OD-NLL 图 (如 typed-mass 下 δ 走 M_j^o 不走 δ·match), 剔除
    L0 = nll()
    probe = torch.autograd.grad(L0, [g[0] for g in groups], allow_unused=True, retain_graph=False)
    used, dropped = [], []
    for (p, kind, labs), gi in zip(groups, probe):
        (used if gi is not None else dropped).append((p, kind, labs))
    if dropped:
        print(f"  ⚠ 剔除未进 OD 图的参数(此 formulation 下不存在/不可识别): "
              f"{', '.join(l for _,_,ls in dropped for l in ls)}\n")
    sel = [p for p, _, _ in used]
    labels = [l for _, _, ls in used for l in ls]
    jac_kinds = [k for _, k, ls in used for _ in ls]
    P = sum(p.numel() for p in sel)
    print(f"  结构参数 {P} 个进 Fisher (排除 NN/self_loop + 上面剔除项)\n")

    # 观测 Fisher = NLL 的 Hessian (raw 空间), 手写 double-backward 省内存
    L = nll()
    g = torch.autograd.grad(L, sel, create_graph=True)
    g_flat = torch.cat([gi.reshape(-1) for gi in g])
    gnorm = float(g_flat.detach().norm())
    H = torch.zeros(P, P)
    for i in range(P):
        gi = torch.autograd.grad(g_flat[i], sel, retain_graph=True, allow_unused=True)
        H[i] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                          for x, p in zip(gi, sel)])
    H = 0.5 * (H + H.T)

    # 转有效系数空间: F_eff = D H D, D=diag(1/|J|)
    rvals = torch.cat([p.detach().reshape(-1) for p in sel])
    Jabs = torch.cat([abs_jacobian(k, r.reshape(1)) for k, r in zip(jac_kinds, rvals)])
    D = (1.0 / Jabs.clamp_min(1e-6))
    F = D[:, None] * H * D[None, :]

    evals, evecs = torch.linalg.eigh(F)
    ev = evals.numpy()
    ev_pos = np.clip(ev, 0, None)
    lam_max = ev_pos.max(); lam_min_pos = ev_pos[ev_pos > 0].min() if (ev_pos > 0).any() else 0.0

    print(f"  (在训练最优处, |grad|={gnorm:.2e} —— 越小越接近驻点, Fisher≈Hessian 越可靠)\n")
    print("--- 特征谱 (log10, 降序; 大=硬/可识别, 小=软/不可识别方向) ---")
    for i, e in enumerate(sorted(ev, reverse=True)):
        bar = "█" * max(0, int((np.log10(abs(e)+1e-30) - np.log10(lam_max+1e-30) + 8) * 2)) if e > 0 else ""
        print(f"  λ{i:2d} = {e:+.3e}  {bar}")
    neg = int((ev < -1e-9).sum())
    if neg: print(f"  ⚠ {neg} 个负特征值(未完全收敛/非凸), 取绝对值看尺度")

    # 有效可识别维数
    tol = 1e-3
    rank_tol = int((ev_pos > lam_max * tol).sum())
    p_i = ev_pos / ev_pos.sum() if ev_pos.sum() > 0 else ev_pos
    ent = -np.sum([x * np.log(x) for x in p_i if x > 0])
    D_eff = float(np.exp(ent))
    cond = lam_max / lam_min_pos if lam_min_pos > 0 else float("inf")
    print(f"\n--- 模型级识别性指标 ---")
    print(f"  参数总数 P = {P}")
    print(f"  有效可识别维数(阈值秩 λ>λmax·1e-3) = {rank_tol} / {P}")
    print(f"  有效可识别维数(熵维数 exp(H))       = {D_eff:.1f} / {P}")
    print(f"  条件数 λmax/λmin+                    = {cond:.2e}  (大=sloppy, 跨数量级)")

    # 逐参数条件 Fisher (对角)
    diagF = torch.diag(F).numpy()
    order = np.argsort(-diagF)
    print(f"\n--- 逐参数条件刚度 (diag F_eff, 大=被数据钉得紧) top/bottom ---")
    for i in order[:6]:  print(f"  硬 {labels[i]:10s} {diagF[i]:.3e}")
    print("   ...")
    for i in order[-6:]: print(f"  软 {labels[i]:10s} {diagF[i]:.3e}")

    # 最软的 3 个特征向量 = 哪些参数组合识别不出
    print(f"\n--- 最软方向 (识别不出的参数组合) ---")
    idx_soft = np.argsort(ev)[:3]
    for k in idx_soft:
        vec = evecs[:, k].numpy()
        top = np.argsort(-np.abs(vec))[:4]
        combo = " + ".join(f"{vec[j]:+.2f}·{labels[j]}" for j in top)
        print(f"  λ={ev[k]:+.2e}: {combo}")


if __name__ == "__main__":
    main()
