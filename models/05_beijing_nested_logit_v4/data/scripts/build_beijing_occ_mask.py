"""外生 frozen 职业 mask -> beijing_occ_mask.npz

动机: 把职业匹配从"软门(跟 typed-mass 引力同信号, 冗余)"升级成"硬的、外生的考虑集筛选"——
非补偿性: 占本职业岗位太少的格子, 不管总岗位多大, 直接踢出考虑集(typed-mass 的连续 log 做不到这一点)。

口径: 对每个职业 o, 可去格 = 覆盖该职业 95% 岗位 M_j^o(=总岗位×demand_share) 的格子。
来源: beijing_aux.npz 的 demand_share(2008 经普街道行业 + 2023 AOI, 外生于 OD) × grid jobs。
空职业(occ6 不便分类, 0 岗位): 全 True(无 mask, 它在 soc_props 里权重也≈0)。

应用: trainer --use-frozen-occ-mask, 在 consideration filter 加 (elig-1)*30 硬惩罚;
      建议同时 --no-match-filter 关掉软门(同信号冗余)。孤儿自愈: 全 mask 段的 -30 在 segment-softmax 抵消 -> 退化无 mask, 不 NaN。
"""
import sys
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "processed"
COVERAGE = 0.95   # 每职业保留覆盖其岗位这一比例的格子


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    aux = np.load(PROC / "beijing_aux.npz", allow_pickle=True)
    g = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    ds = aux["demand_share"].astype(np.float64)       # (N,7) 每格职业占比
    jobs = g["jobs"].astype(np.float64)               # (N,)
    N, S = ds.shape
    Mo = ds * jobs[:, None]                            # (N,7) 本职业岗位

    occ_mask = np.zeros((N, S), np.float32)
    counts = []
    for o in range(S):
        col = Mo[:, o]; tot = col.sum()
        if tot <= 0:                                   # 空职业 -> 全 True (无 mask)
            occ_mask[:, o] = 1.0; counts.append((o, N, 0.0)); continue
        order = np.argsort(col)[::-1]                  # 岗位多->少
        cum = np.cumsum(col[order]) / tot
        keep_n = int((cum < COVERAGE).sum()) + 1       # 覆盖 COVERAGE 的最少格数
        occ_mask[order[:keep_n], o] = 1.0
        counts.append((o, keep_n, float(col[order[keep_n - 1]])))

    # 孤儿诊断: 每出发地每职业, 候选集里还剩几个可去格 (用边的候选集)
    e = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    o_idx = e["o_idx"].astype(np.int64); d_idx = e["d_idx"].astype(np.int64)
    seg_ptr = e["seg_ptr"].astype(np.int64); O = len(seg_ptr) - 1
    orphan = np.zeros(S, np.int64)
    cand_left = np.zeros(S, np.float64)
    for oi in range(O):
        a, b = seg_ptr[oi], seg_ptr[oi + 1]
        dd = d_idx[a:b]
        elig = occ_mask[dd]                            # (cand, S)
        ncand = elig.sum(0)                            # (S,) 每职业剩几个候选
        orphan += (ncand == 0).astype(np.int64)
        cand_left += ncand
    cand_left /= max(O, 1)

    np.savez(PROC / "beijing_occ_mask.npz",
             occ_mask=occ_mask, coverage=np.float32(COVERAGE),
             keep_counts=np.array([c[1] for c in counts], np.int64))

    print(f"[OK] beijing_occ_mask.npz  (N={N}, S={S}, coverage={COVERAGE})")
    print("  各职业: 可去格数 / 占全图 / 平均每出发地剩候选 / 孤儿出发地数(自愈,无惩罚)")
    for o in range(S):
        print(f"    occ{o}: 可去 {counts[o][1]:5d} 格 ({counts[o][1]/N*100:4.1f}%)  "
              f"均候选剩 {cand_left[o]:6.0f}  孤儿 {orphan[o]:5d}/{O} ({orphan[o]/O*100:.1f}%)")
    print(f"  孤儿=候选全被筛(softmax 自愈退化无 mask, 不 NaN); occ6 空职业全 True")


if __name__ == "__main__":
    main()
