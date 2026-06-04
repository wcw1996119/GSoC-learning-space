"""Phase 1c: 收入代理 (小区房价) -> beijing_income.npz

源: D:\\UoM\\congestion_project\\202501北京小区\\202501北京小区84.csv (安居客 ~14597 小区, GBK)
    列: [1]小区 [2]区 [5]经度 [6]纬度 [8]每平米均价 ...
依赖: data/processed/beijing_grid.npz (coords_latlon 投影一致)
出: data/processed/beijing_income.npz (gitignored)

产出 (伦敦对应 aux income_z / income_tier_props):
  income_z          (N,)   z(log 房价/㎡), 格子无小区 -> 区中位 -> 全市中位 回填
  income_tier_props (N,3)  低/中/高 软档 (按价格 z 到三分位中心的高斯权重)
  price_per_cell    (N,)   原始均价 (诊断 + log_W 代理用)
"""
import numpy as np
import pandas as pd
from pathlib import Path

CSV = Path(r"D:/UoM/congestion_project/202501北京小区/202501北京小区84.csv")
PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
OUT = PROC / "beijing_income.npz"
N_TIERS = 3

def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)

def main():
    grid = np.load(GRID, allow_pickle=True)
    coords = grid["coords_latlon"].astype(np.float64)   # (N,2) lat,lon
    district_idx = grid["district_idx"]
    N = len(coords)
    lat0, lon0 = float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))

    def proj(lat, lon):
        x = (lon - lon0) * np.cos(np.deg2rad(lat0)) * 111_320.0
        y = (lat - lat0) * 110_540.0
        return np.stack([x, y], axis=1)

    cell_xy = proj(coords[:, 0], coords[:, 1])

    # 房价 CSV (GBK, 按位置取列, 列名乱码)
    df = pd.read_csv(CSV, encoding="gbk", header=0)
    cols = list(df.columns)
    lng = pd.to_numeric(df[cols[5]], errors="coerce").values
    lat = pd.to_numeric(df[cols[6]], errors="coerce").values
    price = pd.to_numeric(df[cols[8]], errors="coerce").values
    ok = np.isfinite(lng) & np.isfinite(lat) & np.isfinite(price) & (price > 0)
    lng, lat, price = lng[ok], lat[ok], price[ok]
    n_comm = len(price)

    # 小区 -> 最近格子 (KDTree)
    from scipy.spatial import cKDTree
    comm_xy = proj(lat, lng)
    tree = cKDTree(cell_xy)
    dist_m, nn_cell = tree.query(comm_xy, k=1)

    # 每格均价 (有小区的格子)
    price_sum = np.zeros(N); price_cnt = np.zeros(N)
    np.add.at(price_sum, nn_cell, price)
    np.add.at(price_cnt, nn_cell, 1.0)
    has = price_cnt > 0
    price_cell = np.full(N, np.nan)
    price_cell[has] = price_sum[has] / price_cnt[has]

    # 回填: 区中位 -> 全市中位
    global_med = np.nanmedian(price_cell)
    for d in np.unique(district_idx):
        m = district_idx == d
        dm = np.nanmedian(price_cell[m])
        fill = dm if np.isfinite(dm) else global_med
        price_cell[m & ~np.isfinite(price_cell)] = fill
    price_cell[~np.isfinite(price_cell)] = global_med

    income_z = zscore(np.log(price_cell))

    # income_tier_props: z 到三分位中心的高斯软权重
    z = income_z.astype(np.float64)
    centers = np.quantile(z, [1/6, 0.5, 5/6])   # 低/中/高 档中心
    bw = (centers[2] - centers[0]) / 2 + 1e-6
    w = np.exp(-((z[:, None] - centers[None, :]) / bw) ** 2)
    tier_props = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        income_z=income_z,
        income_tier_props=tier_props,
        price_per_cell=price_cell.astype(np.float32),
    )

    print(f"[OK] {OUT}")
    print(f"  小区 {n_comm:,} 个 (有效); 命中格子 {int(has.sum()):,}/{N} "
          f"({has.sum()/N*100:.1f}%); 最近匹配中位距 {np.median(dist_m):.0f}m")
    print(f"  房价/㎡ range [{np.nanmin(price_cell):.0f}, {np.nanmax(price_cell):.0f}] "
          f"median {np.nanmedian(price_cell):.0f}")
    print(f"  income_z mean {income_z.mean():.3f} std {income_z.std():.3f}")
    print(f"  tier-marginal [低,中,高] = {tier_props.mean(0).round(3).tolist()}")

if __name__ == "__main__":
    main()
