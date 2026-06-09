"""北京 v4 行为诊断 (本地 CPU, 不需 GPU)

加载训练好的 .pt + 数据, 算模型学到的行为是否合常识 (CPC + behavioral fidelity 双轴):
  1. 参数解读 (各项符号/量级/收入档异质性)
  2. 模型隐含方式分担 P(m|i,j) (流量加权) vs 北京常识
  3. 预测 vs 观测 出行距离分布
  4. 预测 vs 观测 自连边(同格通勤)占比
用法: python experiments/analyze_beijing.py --pt evaluation_outputs/v4_full_s0.pt
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, BeijingPairEncoder
from train_beijing import build_data, make_batch, chunk_ranges, segment_sum

PERIODS = ["ampeak", "pmpeak", "midday", "night"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(ROOT / "evaluation_outputs" / "v4_full_s0.pt"))
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args()
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    a = ck["args"]
    print(f"=== {Path(args.pt).name}  (best CPC {ck['best_cpc']:.4f}) ===\n")

    # 重建数据 (复用 trainer build_data)
    dyn = a.get("use_dynamic", False)
    da = SimpleNamespace(smoke_cells=0, use_nn=a.get("use_nn", False), seed=a["seed"],
                         val_frac=a["val_frac"], device="cpu",
                         use_dynamic=dyn, road_graph=a.get("road_graph", False))
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

    def embed():
        if enc is None: return None, None
        ew = data.get("edge_weight")
        if dyn:
            return enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"],
                                  data["N"], data["hour2period"], edge_weight=ew)
        return enc.node_embed(data["Xnode"], data["edge_index"], data["N"], edge_weight=ew)

    # ---- 1. 参数解读 ----
    pr = head.param_report()
    print("--- 1. 参数解读 (行为含义) ---")
    print(f"  λ(mode嵌套) = {pr['lambda']:.3f}  (→1 趋近独立 logit; <1 mode 内相关)")
    bt = pr['beta_t0']
    print(f"  β_t [车,公交,步行] = {bt} 元/分钟效用  → 步行每分钟痛 {bt[2]/min(bt[0],-1e-6):.0f}× 车")
    print(f"  γ_M(引力,收入档低/中/高) = {pr['gamma_M']}  (>0 岗位多更吸引)")
    print(f"  ν_D(竞争) = {pr['nu_D']}  (<0 竞争者多更不吸引)")
    print(f"  α_W(工资敏感) = {pr['alpha_W']}  (按收入档)")
    if 'T_max' in pr:
        print(f"  T_max(通勤忍受度,分钟) = {pr['T_max']}  (Bhat 1995 风格三档)")
    if 'w_nn' in pr:
        print(f"  w_nn(NN残差权重) = {pr['w_nn']}")

    # ---- 2-4. 流量加权行为 (origin-chunk, no_grad) ----
    O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, args.chunks)
    lam = head.lam.detach()
    b0, b1 = head.beta_t0.detach(), head.beta_t1.detach()
    asc, th = head.asc.detach(), head.theta_inc.detach()
    mode_flow = np.zeros(3); tot_pred = 0.0
    sum_pred_km = sum_obs_km = sum_pred = sum_obs = 0.0
    self_pred = self_obs = 0.0
    # 距离档 / 收入档 拆分 (CPC = 2Σmin/(Σpred+Σobs), val 边)
    BANDS = [(0,2),(2,5),(5,10),(10,20),(20,40)]
    band_min = np.zeros(len(BANDS)); band_pd = np.zeros(len(BANDS)); band_ob = np.zeros(len(BANDS))
    band_mode = np.zeros((len(BANDS),3)); band_w = np.zeros(len(BANDS))
    tier_min = np.zeros(3); tier_pd = np.zeros(3); tier_ob = np.zeros(3)
    tier_props_all = data["tier_props"]
    with torch.no_grad():
        e_o, e_d = embed()
        for (e0, e1, sb, ns) in chunks:
            o = data["o"][e0:e1]; d = data["d"][e0:e1]
            hav = (data["hav_km"][e0:e1] if "hav_km" in data else torch.exp(data["log_d"][e0:e1]))
            is_self = (o == d)
            inc_o = data["income"][o]
            for p in range(4):
                batch = make_batch(data, p, use_soc, e0, e1, sb, ns)
                v_nn = (enc.edge_vnn(e_o, e_d, p, o, d) if dyn else enc.edge_vnn(e_o, e_d, o, d)) \
                       if enc is not None else None
                logP = head(batch, v_nn=v_nn)
                fp = data["flow"][e0:e1, p]
                otot = segment_sum(fp, batch["seg_id"], ns)
                pred = logP.exp() * otot[batch["seg_id"]]          # 预测流
                # 方式分担 P(m|edge)
                tms = [batch["t_car"], batch["t_transit"], batch["t_walk"]]
                Vlm = torch.stack([(b0[m]+b1[m]*batch["log_d"])*tms[m]+asc[m]+th[m]*inc_o
                                   for m in range(3)], 0) / lam     # (3,E)
                Pm = torch.softmax(Vlm, 0)                          # (3,E)
                w = pred
                for m in range(3): mode_flow[m] += float((Pm[m]*w).sum())
                tot_pred += float(w.sum())
                sum_pred_km += float((pred*hav).sum()); sum_pred += float(pred.sum())
                sum_obs_km += float((fp*hav).sum()); sum_obs += float(fp.sum())
                self_pred += float(pred[is_self].sum()); self_obs += float(fp[is_self].sum())
                # 拆分 (只在 val 边算 CPC)
                vm = data["edge_is_val"][e0:e1]
                havn = hav.numpy(); predn = pred.numpy(); fpn = fp.numpy(); vmn = vm.numpy()
                mn = np.minimum(predn, fpn)
                for bi,(lo,hi) in enumerate(BANDS):
                    bmask = (havn>=lo)&(havn<hi)
                    vb = bmask & vmn
                    band_min[bi]+=mn[vb].sum(); band_pd[bi]+=predn[vb].sum(); band_ob[bi]+=fpn[vb].sum()
                    Pmn = Pm.numpy()
                    for m in range(3): band_mode[bi,m]+=float((Pmn[m][bmask]*predn[bmask]).sum())
                    band_w[bi]+=float(predn[bmask].sum())
                tier_o = tier_props_all[o].argmax(1).numpy()
                for ti in range(3):
                    vt = (tier_o==ti) & vmn
                    tier_min[ti]+=mn[vt].sum(); tier_pd[ti]+=predn[vt].sum(); tier_ob[ti]+=fpn[vt].sum()

    ms = mode_flow / max(tot_pred, 1)
    print("\n--- 2. 模型隐含方式分担 (流量加权) ---")
    print(f"  [车, 公交, 步行] = {ms.round(3).tolist()}")
    print(f"  [!] 公交时间是无地铁的代理 + mode_share 先验主导(OD无方式label) → 方式分担弱识别, 仅看结构")
    print("\n--- 3. 出行距离 (km, 流量加权均值) ---")
    print(f"  观测 {sum_obs_km/max(sum_obs,1):.2f}  vs  预测 {sum_pred_km/max(sum_pred,1):.2f}")
    print("\n--- 4. 自连边(同格通勤)占比 ---")
    print(f"  观测 {self_obs/max(sum_obs,1)*100:.1f}%  vs  预测 {self_pred/max(sum_pred,1)*100:.1f}%")

    print("\n--- 5. CPC + 方式分担 按距离档 (val 边) ---")
    for bi,(lo,hi) in enumerate(BANDS):
        cpc = 2*band_min[bi]/max(band_pd[bi]+band_ob[bi],1)
        ms_b = band_mode[bi]/max(band_w[bi],1)
        print(f"  {lo:2d}-{hi:2d}km: CPC {cpc:.3f}  方式[车{ms_b[0]:.2f} 公交{ms_b[1]:.2f} 步行{ms_b[2]:.2f}]")

    print("\n--- 6. CPC 按收入档 (val 边, 出发地主档) ---")
    for ti,nm in enumerate(["低","中","高"]):
        cpc = 2*tier_min[ti]/max(tier_pd[ti]+tier_ob[ti],1)
        print(f"  {nm}收入: CPC {cpc:.3f}")


if __name__ == "__main__":
    main()
