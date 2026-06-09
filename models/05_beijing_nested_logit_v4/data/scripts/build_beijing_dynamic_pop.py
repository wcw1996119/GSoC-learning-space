"""动态特征 A: 百度热力逐小时 -> cell × 24 小时人口序列 -> beijing_dynamic_pop.npz

源: D:/UoM/congestion_project/北京市百度热力/beijing20250303/北京市_20250303{HH}.csv
    (24 个文件 = 24 小时; 字段 wgs84_LNG, wgs84_LAT, value=相对人口密度指数)
依赖: cell_index.parquet (16907 格中心 lat/lon, 同 grid builder 投影)
出: data/processed/beijing_dynamic_pop.npz (gitignored)

产出 (保留细粒度, 不提前压成 4 时段 -> 喂 GRU/TCN):
  pop_hourly (N,24)   每 cell 每小时人口 (热力 value 落格求和)
  pop_period (N,4)    按时段池化 (早7-9 / 晚17-19 / 平10-16 / 夜20-6) 仅作 sanity
  hours      (24,)    小时标签
⚠ 热力 value 是相对密度指数非绝对人数; 跨年 2025-03 vs OD 2023 (典型日代理, limitation)。
"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

PAPERC = Path(r"D:/UoM/PaperC_local")
HEAT = Path(r"D:/UoM/congestion_project/北京市百度热力/beijing20250303")
OUT = Path(__file__).resolve().parents[1] / "processed" / "beijing_dynamic_pop.npz"
CUTOFF_M = 1000.0   # 点离最近 cell 中心 > 1km 丢弃 (北京网格外)

# 时段池化映射 (小时 -> period idx): 0=早高峰 1=晚高峰 2=平峰 3=夜
def hour_to_period(h):
    if 7 <= h <= 9:   return 0
    if 17 <= h <= 19: return 1
    if 10 <= h <= 16: return 2
    return 3          # 20-23, 0-6

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ci = pq.read_table(PAPERC / "cell_index.parquet").to_pandas().sort_values("idx").reset_index(drop=True)
    N = len(ci)
    lat = ci.lat.values.astype(np.float64); lon = ci.lon.values.astype(np.float64)
    lat0, lon0 = float(np.median(lat)), float(np.median(lon))   # 同 build_beijing_grid 投影
    def proj(la, lo):
        x = (lo - lon0) * np.cos(np.deg2rad(lat0)) * 111_320.0
        y = (la - lat0) * 110_540.0
        return np.stack([x, y], axis=1)
    cell_xy = proj(lat, lon)
    tree = cKDTree(cell_xy)

    files = sorted(glob.glob(str(HEAT / "*.csv")))
    assert len(files) == 24, f"期望 24 小时, 实得 {len(files)}"
    pop_hourly = np.zeros((N, 24), np.float64)
    kept_frac = []
    for f in files:
        hh = int(re.search(r"(\d{2})\.csv$", f).group(1))
        df = pd.read_csv(f)
        pxy = proj(df["wgs84_LAT"].values.astype(np.float64), df["wgs84_LNG"].values.astype(np.float64))
        dist, idx = tree.query(pxy, k=1)
        keep = dist <= CUTOFF_M
        kept_frac.append(keep.mean())
        np.add.at(pop_hourly[:, hh], idx[keep], df["value"].values.astype(np.float64)[keep])

    # 时段池化 (sanity)
    pop_period = np.zeros((N, 4), np.float64)
    cnt = np.zeros(4)
    for h in range(24):
        p = hour_to_period(h); pop_period[:, p] += pop_hourly[:, h]; cnt[p] += 1
    pop_period /= np.maximum(cnt, 1)   # 每时段均值(每小时)

    np.savez_compressed(OUT,
        pop_hourly=pop_hourly.astype(np.float32),
        pop_period=pop_period.astype(np.float32),
        hours=np.arange(24))

    # --- 自检 ---
    print(f"[OK] {OUT}")
    print(f"  N={N}  小时={len(files)}  点落格保留率 mean {np.mean(kept_frac)*100:.1f}%")
    tot_h = pop_hourly.sum(0)
    print(f"  每小时总热力 (00..23): {[int(x) for x in tot_h]}")
    peak_h = int(tot_h.argmax()); low_h = int(tot_h.argmin())
    print(f"  峰值小时 {peak_h}点 ({tot_h[peak_h]:,.0f}); 谷值 {low_h}点 ({tot_h[low_h]:,.0f}) -> 比 {tot_h[peak_h]/max(tot_h[low_h],1):.2f}x")
    cov = (pop_hourly.sum(1) > 0).mean()
    print(f"  有热力覆盖的 cell 占比 {cov*100:.1f}%")
    print(f"  时段池化均值 [早,晚,平,夜] 总量: {[int(pop_period[:,p].sum()) for p in range(4)]}")

if __name__ == "__main__":
    main()
