"""反事实场景 A (新就业中心) + 跨 seed 敏感性: 给目标格加岗位, 看预测涌入变化。
判据: 同一干预下, 多个等价 seed(都 CPC~0.69) 给的"吸引通勤者数"散布多大。
  散布小 -> 反事实稳/可用(哪怕单参数软); 大 -> 不可用。
用法: python experiments/scenario_beijing_A.py --pts evaluation_outputs/v4_stgnn_s0.pt ... --add-jobs 50000
"""
import argparse, sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "experiments"))
from beijing_model import BeijingNestedHead, BeijingPairEncoder
from train_beijing import build_data, make_batch, chunk_ranges, segment_sum
PROC = ROOT / "data" / "processed"


def load_model(pt):
    ck = torch.load(pt, map_location="cpu", weights_only=False); a = ck["args"]
    dyn = a.get("use_dynamic", False)
    da = SimpleNamespace(smoke_cells=0, use_nn=a.get("use_nn", False), seed=a["seed"],
                         val_frac=a["val_frac"], device="cpu",
                         use_dynamic=dyn, road_graph=a.get("road_graph", False))
    data = build_data(da); use_soc = a.get("use_soc_mixture", False)
    head = BeijingNestedHead(n_districts=16, use_self_loop=not a.get("no_self_loop", False),
                             use_consideration=a.get("use_consideration", False),
                             use_soc_mixture=use_soc, gnn_mode=a.get("gnn_mode", "residual"),
                             use_typed_mass=a.get("typed_mass_occ", False))
    head.load_state_dict(ck["head_state"]); head.eval()
    enc = None
    if a.get("use_nn", False):
        if dyn:
            from beijing_model import BeijingDualBranchEncoder
            enc = BeijingDualBranchEncoder(static_dim=data["Xnode"].shape[1], dyn_dim=data["Xdyn"].shape[2])
        else:
            enc = BeijingPairEncoder(in_dim=data["Xnode"].shape[1])
        enc.load_state_dict(ck["enc_state"]); enc.eval()
    return ck, a, data, head, enc, use_soc, dyn


def inflow_to(target, data, head, enc, use_soc, dyn, chunks):
    """全量前向, 返回预测涌入 target 格的总通勤量(4 时段合)。"""
    with torch.no_grad():
        if enc is None: e_o = e_d = None
        elif dyn: e_o, e_d = enc.node_embed(data["Xnode"], data["Xdyn"], data["edge_index"], data["N"],
                                            data["hour2period"], edge_weight=data.get("edge_weight"))
        else: e_o, e_d = enc.node_embed(data["Xnode"], data["edge_index"], data["N"],
                                        edge_weight=data.get("edge_weight"))
        inflow = 0.0
        for (e0, e1, sb, ns) in chunks:
            o = data["o"][e0:e1]; d = data["d"][e0:e1]; to_tgt = (d == target)
            if not bool(to_tgt.any()): continue
            for p in range(4):
                batch = make_batch(data, p, use_soc, e0, e1, sb, ns)
                if enc is None: v = None
                elif dyn: v = enc.edge_vnn(e_o, e_d, p, o, d)
                else: v = enc.edge_vnn(e_o, e_d, o, d)
                logP = head(batch, v_nn=v)
                fp = data["flow"][e0:e1, p]; otot = segment_sum(fp, batch["seg_id"], ns)
                pred = logP.exp() * otot[batch["seg_id"]]
                inflow += float(pred[to_tgt].sum())
    return inflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", nargs="+", required=True)
    ap.add_argument("--add-jobs", type=float, default=50000.0)
    ap.add_argument("--chunks", type=int, default=16)
    args = ap.parse_args()

    # 目标格: 多居民、少岗位的格(住宅区引入新就业中心), 确定性(跨seed同一格)
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    jobs0 = grid["jobs"].astype(np.float64); res = grid["residents"].astype(np.float64)
    cand = np.where((jobs0 <= np.quantile(jobs0, 0.5)) & (res > 0))[0]
    target = int(cand[np.argmax(res[cand])])
    print(f"目标新就业中心: cell {target}  (现岗位 {jobs0[target]:.0f}, 居民 {res[target]:.0f}) + {args.add_jobs:.0f} 岗位\n")

    deltas = []
    for pt in args.pts:
        ck, a, data, head, enc, use_soc, dyn = load_model(pt)
        O = data["num_seg"]; chunks = chunk_ranges(data["seg_starts"], O, args.chunks)
        base = inflow_to(target, data, head, enc, use_soc, dyn, chunks)
        # 干预: 改 target 的 log_M (引力) + Xnode 第0列(NN看到的log_M)
        jobs = jobs0.copy(); jobs[target] += args.add_jobs
        lz0 = np.log(jobs0 + 1.0); mu, sd = lz0.mean(), lz0.std() + 1e-9   # 固定基线归一化
        log_M_z = ((np.log(jobs + 1.0) - mu) / sd).astype(np.float32)
        data["log_M"] = torch.from_numpy(log_M_z)
        data["Xnode"][:, 0] = torch.from_numpy(log_M_z)
        new = inflow_to(target, data, head, enc, use_soc, dyn, chunks)
        d_sw = new - base; deltas.append(d_sw)
        print(f"  {Path(pt).name} (CPC {ck['best_cpc']:.4f}): 涌入 {base:.0f} -> {new:.0f}  "
              f"吸引 {d_sw:+.0f} 通勤者 ({d_sw/max(base,1)*100:+.1f}%)")

    d = np.array(deltas)
    print(f"\n=== 跨 {len(d)} seed 敏感性 ===")
    print(f"  吸引通勤者: 均值 {d.mean():.0f} ± {d.std():.0f}  (CV={d.std()/max(abs(d.mean()),1)*100:.1f}%)")
    print(f"  CV 小(<~15%)=反事实稳/可用(哪怕单参数软); 大=不可用")


if __name__ == "__main__":
    main()
