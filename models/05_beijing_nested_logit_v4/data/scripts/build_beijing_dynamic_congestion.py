"""动态特征 B: 实时路况 30分钟 SHP -> cell × 24小时 拥堵速度因子 -> beijing_dynamic_cong.npz

源: D:/UoM/congestion_project/北京市实时路况.../北京市16级实时路况{YYYY-MM-DD-HH-MM}.shp
    (97 帧, 每30分钟, 2天; 字段 status + type[畅通/轻度拥堵/缓行/严重拥堵/未知] + length + geometry)
依赖: cell_index.parquet (16907 格中心)
出: data/processed/beijing_dynamic_cong.npz (gitignored)

产出 (典型日, 跟热力对齐到 24 小时):
  speed_hourly (N,24)  每 cell 每小时 长度加权速度因子 (1.0=畅通, 越低越堵; 无路网格=1.0)
  speed_period (N,4)   时段池化 [早,晚,平,夜]
  cover_mask   (N,)    该 cell 有没有路网数据
⚠ status 是拥堵【等级】非连续车速 -> type->速度因子是假设映射(承重墙, limitation 要标);
  跨年 2025-02 vs OD 2023; 仅 2 天典型日。
"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
import geopandas as gpd

PAPERC = Path(r"D:/UoM/PaperC_local")
ROAD = Path(r"D:/UoM/congestion_project/北京市实时路况2025-02-11-23-59-00-2025-02-13-23-59-00实时路况")
OUT = Path(__file__).resolve().parents[1] / "processed" / "beijing_dynamic_cong.npz"
CUTOFF_M = 1500.0
# type -> 速度因子 (Amap 风格假设; 1.0=自由流)
SPEED_MAP = {"畅通": 1.0, "轻度拥堵": 0.75, "缓行": 0.55, "严重拥堵": 0.35, "未知": np.nan}

def hour_to_period(h):
    if 7 <= h <= 9:   return 0
    if 17 <= h <= 19: return 1
    if 10 <= h <= 16: return 2
    return 3

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ci = pq.read_table(PAPERC / "cell_index.parquet").to_pandas().sort_values("idx").reset_index(drop=True)
    N = len(ci)
    lat = ci.lat.values.astype(np.float64); lon = ci.lon.values.astype(np.float64)
    lat0, lon0 = float(np.median(lat)), float(np.median(lon))
    def proj(la, lo):
        return np.stack([(lo-lon0)*np.cos(np.deg2rad(lat0))*111_320.0, (la-lat0)*110_540.0], axis=1)
    tree = cKDTree(proj(lat, lon))

    CONG = {"轻度拥堵", "缓行", "严重拥堵"}   # 拥堵(分子); 分母=畅通+拥堵(排除未知)
    files = sorted(glob.glob(str(ROAD / "*.shp")))
    print(f"  路况帧 {len(files)} 个, 落格中...")
    # 累计 (跨2天同小时合并): 拥堵路长 / 有效路长 + 速度加权(留作备用)
    cong = np.zeros((N, 24)); road = np.zeros((N, 24))
    swsum = np.zeros((N, 24))
    type_seen = {}
    for fi, f in enumerate(files):
        # ⚠ 只从文件名取小时: 父文件夹名也含起始时间戳, 扫全路径会全匹配成 23
        hh = int(re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", Path(f).name).group(4))
        g = gpd.read_file(f)
        tp = g["type"].astype(str).values
        for t in np.unique(tp): type_seen[t] = type_seen.get(t, 0) + 1
        sp = pd.Series(tp).map(SPEED_MAP).values
        ln = g["length"].values.astype(np.float64)
        cen = g.geometry.centroid
        pxy = proj(cen.y.values.astype(np.float64), cen.x.values.astype(np.float64))
        dist, idx = tree.query(pxy, k=1)
        valid = (dist <= CUTOFF_M) & np.isfinite(sp)        # 排除未知/远点
        is_c = np.isin(tp, list(CONG)) & valid
        np.add.at(road[:, hh], idx[valid], ln[valid])
        np.add.at(cong[:, hh], idx[is_c], ln[is_c])
        np.add.at(swsum[:, hh], idx[valid], (ln * sp)[valid])
        if fi % 24 == 0: print(f"    {fi}/{len(files)} {Path(f).name[:30]}...")

    cover_mask = (road.sum(1) > 0)
    cong_frac_hourly = np.divide(cong, road, out=np.zeros_like(cong), where=road > 0)   # 拥堵路长占比
    speed_hourly = np.divide(swsum, road, out=np.full_like(swsum, 1.0), where=road > 0) # 速度因子(备用)

    def pool(arr):
        out = np.zeros((N, 4)); cnt = np.zeros(4)
        for h in range(24):
            p = hour_to_period(h); out[:, p] += arr[:, h]; cnt[p] += 1
        return out / np.maximum(cnt, 1)
    cong_frac_period = pool(cong_frac_hourly); speed_period = pool(speed_hourly)

    np.savez_compressed(OUT,
        cong_frac_hourly=cong_frac_hourly.astype(np.float32),
        cong_frac_period=cong_frac_period.astype(np.float32),
        speed_hourly=speed_hourly.astype(np.float32),
        speed_period=speed_period.astype(np.float32),
        cover_mask=cover_mask)

    print(f"[OK] {OUT}")
    print(f"  type 出现帧数: {type_seen}")
    print(f"  有路网的 cell 占比 {cover_mask.mean()*100:.1f}%")
    cov = cong_frac_hourly[cover_mask]
    ch = cov.mean(0)
    print(f"  覆盖格 每小时拥堵占比×100 (00..23): {[round(float(x)*100,2) for x in ch]}")
    pk = int(ch.argmax()); lo = int(ch.argmin())
    print(f"  最堵 {pk}点 ({ch[pk]*100:.2f}%); 最畅 {lo}点 ({ch[lo]*100:.2f}%) -> 比 {ch[pk]/max(ch[lo],1e-6):.1f}x")
    # 峰时空间集中度: 有多少格真有拥堵
    pkcol = cong_frac_hourly[:, pk]
    print(f"  晚峰{pk}点: 拥堵占比>5%的格 {int((pkcol>0.05).sum())} 个, >20%的格 {int((pkcol>0.2).sum())} 个, 最堵格 {pkcol.max()*100:.0f}%")
    print(f"  时段拥堵占比×100 [早,晚,平,夜]: {[round(float(cong_frac_period[cover_mask,p].mean())*100,2) for p in range(4)]}")

if __name__ == "__main__":
    main()
