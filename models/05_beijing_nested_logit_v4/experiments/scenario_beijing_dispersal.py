"""反事实: 北京总规超大城市疏解 (通州副中心 +岗位 / 核心区 -岗位) + 跨seed敏感性。
测聚合/相对量(比单格稳): 通州涌入 / 核心区涌入(缓解) / 总通勤距离。
判据: 这些量在等价seed间的CV. 小=对总规疏解可用, 大=不可用。
用法: python experiments/scenario_beijing_dispersal.py --pts s0.pt s1.pt s2.pt [--add-tongzhou 300000 --core-cut 0.2]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "experiments"))
from train_beijing import make_batch, chunk_ranges, segment_sum
from scenario_beijing_A import load_model
PROC = ROOT / "data" / "processed"
TONGZHOU, CORE = 13, [0, 12]   # 通州 / 东城+西城


def measure(data, head, enc, use_soc, dyn, chunks, tong_m, core_m, mono=None):
    """全量前向 -> (通州涌入, 核心涌入, 流量加权平均通勤距离km)。"""
    with torch.no_grad():
        if enc is None: e_o = e_d = None
        elif dyn: e_o, e_d = enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"], data["N"],
                                            data["hour2period"], edge_weight=data.get("edge_weight"))
        else: e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"],
                                        edge_weight=data.get("edge_weight"))
        def vnn(eo, ed, p, o, d): return enc.edge_vnn(eo, ed, p, o, d) if dyn else enc.edge_vnn(eo, ed, o, d)
        tong = core = dist_num = dist_den = 0.0
        for (e0, e1, sb, ns) in chunks:
            o = data["o"][e0:e1]; d = data["d"][e0:e1]
            hav = data["hav_km"][e0:e1] if "hav_km" in data else torch.exp(data["log_d"][e0:e1])
            td = tong_m[d]; cd = core_m[d]
            for p in range(4):
                batch = make_batch(data, p, use_soc, e0, e1, sb, ns)
                if enc is None: v = None
                else:
                    v = vnn(e_o, e_d, p, o, d)
                    if mono is not None:  # NN 对岗位响应掰单调; 不变格保留 α·溢出(α=狠度旋钮)
                        vb = vnn(mono["eo_b"], mono["ed_b"], p, o, d)
                        dv = v - vb; sgn = mono["jobsign"][d]; al = mono["alpha"]
                        dv = torch.where(sgn > 0, torch.relu(dv),
                                         torch.where(sgn < 0, -torch.relu(-dv), al * dv))
                        v = vb + dv
                logP = head(batch, v_nn=v)
                fp = data["flow"][e0:e1, p]; otot = segment_sum(fp, batch["seg_id"], ns)
                pred = logP.exp() * otot[batch["seg_id"]]
                tong += float(pred[td].sum()); core += float(pred[cd].sum())
                dist_num += float((pred * hav).sum()); dist_den += float(pred.sum())
    return tong, core, dist_num / max(dist_den, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", nargs="+", required=True)
    ap.add_argument("--add-tongzhou", type=float, default=300000.0)
    ap.add_argument("--core-cut", type=float, default=0.2)
    ap.add_argument("--freeze-nn", action="store_true",
                    help="干预只走 RUM 引力(log_M), 不动 NN 输入(NN 不外推) -> 结构性反事实")
    ap.add_argument("--nn-monotonic", action="store_true",
                    help="NN active 但把它对岗位的响应掰单调(涨格≥0/跌格≤0) -> 验 B 地基")
    ap.add_argument("--mono-spillover", type=float, default=0.0,
                    help="狠度旋钮: 不变格保留多少 GNN 溢出(0=严格最稳/1=全放最不稳)")
    ap.add_argument("--chunks", type=int, default=16)
    args = ap.parse_args()

    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    jobs0 = grid["jobs"].astype(np.float64); didx = grid["district_idx"]
    lz0 = np.log(jobs0 + 1.0); MU, SD = lz0.mean(), lz0.std() + 1e-9   # 基线归一化(固定, 干预不重拟合)
    tong_cells = np.where(didx == TONGZHOU)[0]; core_cells = np.where(np.isin(didx, CORE))[0]
    print(f"通州 {len(tong_cells)}格(现{jobs0[tong_cells].sum():.0f}岗位) +{args.add_tongzhou:.0f}; "
          f"核心区 {len(core_cells)}格(现{jobs0[core_cells].sum():.0f}) ×{1-args.core_cut:.2f}\n")

    rows = []
    for pt in args.pts:
        ck, a, data, head, enc, use_soc, dyn = load_model(pt)
        O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, args.chunks)
        tong_m = torch.zeros(data["N"], dtype=torch.bool); tong_m[tong_cells] = True
        core_m = torch.zeros(data["N"], dtype=torch.bool); core_m[core_cells] = True
        b_tong, b_core, b_dist = measure(data, head, enc, use_soc, dyn, chunks, tong_m, core_m)
        # 干预: 通州加岗位(按现岗位+1比例分配) + 核心区减
        jobs = jobs0.copy()
        w = jobs0[tong_cells] + 1.0; jobs[tong_cells] += args.add_tongzhou * w / w.sum()
        jobs[core_cells] *= (1 - args.core_cut)
        # 单调: 先存基线 NN 嵌入 + 岗位变化方向(干预前算)
        mono = None
        if args.nn_monotonic and enc is not None:
            with torch.no_grad():
                if dyn: eo_b, ed_b = enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"], data["N"],
                                                    data["hour2period"], edge_weight=data.get("edge_weight"))
                else: eo_b, ed_b = enc.node_embed(data["Xnode"], data["edge_index"], data["N"], edge_weight=data.get("edge_weight"))
            sgn = np.zeros(data["N"], np.float32); sgn[tong_cells] = 1.0; sgn[core_cells] = -1.0
            mono = dict(eo_b=eo_b, ed_b=ed_b, jobsign=torch.from_numpy(sgn), alpha=args.mono_spillover)
        lmz = ((np.log(jobs + 1.0) - MU) / SD).astype(np.float32)   # 固定基线归一化
        data["log_M"] = torch.from_numpy(lmz)                       # RUM 引力(总是更新)
        # NN 输入: 冻住/分工时不动; 否则更新(单调模式也更新, 让NN看新岗位再掰单调)
        if not args.freeze_nn and not a.get("nn_exclude_jobs", False):
            data["Xnode"][:, 0] = torch.from_numpy(lmz)
        n_tong, n_core, n_dist = measure(data, head, enc, use_soc, dyn, chunks, tong_m, core_m, mono=mono)
        rows.append((n_tong - b_tong, (n_tong/b_tong-1)*100, n_core - b_core, (n_core/b_core-1)*100, n_dist - b_dist))
        print(f"  {Path(pt).name} (CPC {ck['best_cpc']:.4f}):")
        print(f"     通州涌入 {b_tong:.0f}->{n_tong:.0f} ({(n_tong/b_tong-1)*100:+.1f}%)  "
              f"核心区涌入 {b_core:.0f}->{n_core:.0f} ({(n_core/b_core-1)*100:+.1f}%)  "
              f"平均通勤 {b_dist:.2f}->{n_dist:.2f}km")

    r = np.array(rows)
    def cv(x): return x.std() / max(abs(x.mean()), 1e-9) * 100
    print(f"\n=== 跨 {len(r)} seed 敏感性 (CV小=对总规疏解可用) ===")
    print(f"  通州净吸引:   {r[:,0].mean():+.0f} ± {r[:,0].std():.0f}  (CV={cv(r[:,0]):.1f}%)")
    print(f"  通州相对增幅: {r[:,1].mean():+.1f}% ± {r[:,1].std():.1f}  (CV={cv(r[:,1]):.1f}%)")
    print(f"  核心区疏解:   {r[:,2].mean():+.0f} ± {r[:,2].std():.0f}  (CV={cv(r[:,2]):.1f}%)")
    print(f"  核心区相对:   {r[:,3].mean():+.1f}% ± {r[:,3].std():.1f}  (CV={cv(r[:,3]):.1f}%)")
    print(f"  总通勤距离Δ:  {r[:,4].mean():+.3f} ± {r[:,4].std():.3f}km  (CV={cv(r[:,4]):.1f}%)")


if __name__ == "__main__":
    main()
