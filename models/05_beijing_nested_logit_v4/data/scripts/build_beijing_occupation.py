"""Phase 1c: 职业 (区级广播) -> beijing_occupation.npz

源 (只读):
  occ_share_by_district.parquet (16 区 ×7 职业)  出发地职业构成
  xj_industry_2023.parquet      (19 行业 ×16 区)  目的地行业就业
  P_occ_given_ind.parquet       (19 行业 ×7 职业) P(职业|行业)
依赖: beijing_grid.npz (district_idx, district_names)
出: data/processed/beijing_occupation.npz (gitignored)

产出 (伦敦对应 aux soc_props / grid_industry_prop / epsilon):
  soc_props          (N,7)  出发地 7 职业构成 (区级广播)
  grid_industry_prop (N,19) 目的地 19 行业占比 (区级广播)
  epsilon            (7,19) = P(职业|行业)^T
  demand_share       (N,7)  目的地 7 职业需求占比 = norm(grid_industry @ epsilon^T)

⚠ 区级数据 -> 同区内 demand_share 恒定 -> delta_match 必然识别不出 (预期 finding,
  伦敦/北京三次撞墙一致, 见 CLAUDE.md / memory)。保留机器, 不调参压它。
"""
import re
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

PAPERC = Path(r"D:/UoM/PaperC_local")
PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
OUT = PROC / "beijing_occupation.npz"

def main():
    grid = np.load(GRID, allow_pickle=True)
    district_idx = grid["district_idx"]
    district_names = grid["district_names"].astype(str)   # (16,) 与 district_idx 对应
    N = len(district_idx)

    occ = pq.read_table(PAPERC / "occ_share_by_district.parquet").to_pandas()
    # index 是 __index_level_0__ 区名; 取 7 列 occ_share_*
    occ_cols = [c for c in occ.columns if c.startswith("occ_share")]
    occ.index = occ["__index_level_0__"].astype(str).values if "__index_level_0__" in occ.columns else occ.index.astype(str)
    occ_by_dist = {d: occ.loc[d, occ_cols].values.astype(np.float64) for d in occ.index}
    n_occ = len(occ_cols)

    xj = pq.read_table(PAPERC / "xj_industry_2023.parquet").to_pandas()  # index 行业, columns 区
    ind_names = xj.index.astype(str).tolist()
    # 行业代码 (括号内 ASCII 字母, GBK 不破坏 ASCII)
    ind_codes = [re.search(r"\(([A-Z])\)", nm).group(1) if re.search(r"\(([A-Z])\)", nm) else nm
                 for nm in ind_names]
    n_ind = len(ind_codes)

    poi = pq.read_table(PAPERC / "P_occ_given_ind.parquet").to_pandas()
    poi.index = poi["industry_2017"].astype(str).values if "industry_2017" in poi.columns else poi.index.astype(str)
    poi_cols = [c for c in poi.columns if c.startswith("P_occ")]
    # 对齐 epsilon[occ, ind] = P(occ|ind)
    epsilon = np.zeros((n_occ, n_ind), dtype=np.float64)
    for j, code in enumerate(ind_codes):
        if code in poi.index:
            epsilon[:, j] = poi.loc[code, poi_cols].values.astype(np.float64)
        # 缺失行业 -> 0 列 (该行业对任何职业无需求贡献)

    # 区级行业占比 (每区 19 行业归一)
    xj_cols = xj.columns.astype(str).tolist()
    ind_prop_by_dist = {}
    for d in xj_cols:
        v = xj[d].values.astype(np.float64)
        s = v.sum()
        ind_prop_by_dist[d] = v / s if s > 0 else np.ones(n_ind) / n_ind

    # 区名对齐校验: grid district_names 必须能在 occ / xj 里找到
    occ_keys = set(occ.index); xj_keys = set(xj_cols)
    miss_occ = [d for d in district_names if d not in occ_keys]
    miss_xj = [d for d in district_names if d not in xj_keys]
    if miss_occ or miss_xj:
        print(f"  [!] 区名未对齐 occ:{miss_occ} xj:{miss_xj}")
        print(f"      grid: {list(district_names)}")
        print(f"      occ : {list(occ.index)}")
        print(f"      xj  : {xj_cols}")
        raise SystemExit("区名对齐失败, 终止")

    # 广播到格子
    soc_props = np.zeros((N, n_occ), dtype=np.float64)
    grid_industry = np.zeros((N, n_ind), dtype=np.float64)
    for i in range(N):
        d = district_names[district_idx[i]]
        soc_props[i] = occ_by_dist[d]
        grid_industry[i] = ind_prop_by_dist[d]
    # 归一 soc_props
    soc_props = soc_props / soc_props.sum(axis=1, keepdims=True).clip(min=1e-9)

    # demand_share = norm(grid_industry @ epsilon^T)
    expected_occ = grid_industry @ epsilon.T          # (N,7)
    demand_share = expected_occ / expected_occ.sum(axis=1, keepdims=True).clip(min=1e-9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        soc_props=soc_props.astype(np.float32),
        grid_industry_prop=grid_industry.astype(np.float32),
        epsilon=epsilon.astype(np.float32),
        demand_share=demand_share.astype(np.float32),
    )

    print(f"[OK] {OUT}")
    print(f"  n_occ={n_occ} n_ind={n_ind}  16 区全对齐")
    print(f"  ind_codes: {ind_codes}")
    print(f"  soc_props 全市均值: {soc_props.mean(0).round(3).tolist()}")
    print(f"  demand_share 全市均值: {demand_share.mean(0).round(3).tolist()}")
    print(f"  demand_share 跨格子 std (区级广播应很小): "
          f"{demand_share.std(0).round(4).tolist()}")
    # 区级广播 -> demand_share 只有 16 个不同值
    nuniq = len(np.unique(demand_share.round(6), axis=0))
    print(f"  demand_share 唯一行数 = {nuniq} (= 区数, 证实区级广播 -> delta_match 预期平)")

if __name__ == "__main__":
    main()
