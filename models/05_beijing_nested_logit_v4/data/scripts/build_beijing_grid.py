"""Phase 1a: 北京格子层基础特征 -> beijing_grid.npz

源: D:\\UoM\\PaperC_local\\cell_index.parquet  (16907 格, 只读, 不出库)
出: data/processed/beijing_grid.npz  (gitignored)

产出 (伦敦 v3 同构):
  coords_latlon (N,2)  lat,lon
  xy_m          (N,2)  本地等距投影米 (kNN 图 / 自距离用)
  district_idx  (N,)   16 区 factorize 0..15
  district_names(16,)  原始(GBK 乱码)区名, 仅占位; 解码后续
  jobs, residents (N,)
  log_M_z       (N,)   log(jobs+1) z-score  = Hansen 引力 M_j
  (log_D_z = Shen 竞争 在 choiceset builder 里算, 需要 edge)

伦敦对应: demo_cache coords_bng / static_features; aux log_M_z
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

PAPERC = Path(r"D:/UoM/PaperC_local")
OUT = Path(__file__).resolve().parents[1] / "processed" / "beijing_grid.npz"

def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)

def main():
    ci = pq.read_table(PAPERC / "cell_index.parquet").to_pandas()
    ci = ci.sort_values("idx").reset_index(drop=True)
    assert (ci.idx.values == np.arange(len(ci))).all(), "cell_index.idx 必须是 0..N-1 连续"
    N = len(ci)

    lat = ci.lat.values.astype(np.float64)
    lon = ci.lon.values.astype(np.float64)
    coords_latlon = np.stack([lat, lon], axis=1).astype(np.float32)

    # 本地等距投影 (北京中心) -> 米
    lat0, lon0 = float(np.median(lat)), float(np.median(lon))
    x_m = (lon - lon0) * np.cos(np.deg2rad(lat0)) * 111_320.0
    y_m = (lat - lat0) * 110_540.0
    xy_m = np.stack([x_m, y_m], axis=1).astype(np.float32)

    # 区 factorize (字符串可能 GBK 乱码, 用稳定 factorize)
    districts = ci.district.astype(str).values
    uniq, district_idx = np.unique(districts, return_inverse=True)
    district_idx = district_idx.astype(np.int64)
    assert len(uniq) == 16, f"期望 16 区, 实得 {len(uniq)}"

    jobs = ci.jobs.values.astype(np.float32)
    residents = ci.residents.values.astype(np.float32)
    log_M_z = zscore(np.log(jobs + 1.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        coords_latlon=coords_latlon,
        xy_m=xy_m,
        district_idx=district_idx,
        district_names=uniq.astype("U32"),
        jobs=jobs,
        residents=residents,
        log_M_z=log_M_z,
    )

    # --- 数据自检 (feedback_data_sanity_check) ---
    print(f"[OK] {OUT}")
    print(f"  N={N}  districts={len(uniq)}")
    print(f"  jobs      range [{jobs.min():.1f}, {jobs.max():.1f}] sum {jobs.sum():,.0f}")
    print(f"  residents range [{residents.min():.1f}, {residents.max():.1f}] sum {residents.sum():,.0f}")
    print(f"  log_M_z   mean {log_M_z.mean():.3f} std {log_M_z.std():.3f}")
    print(f"  xy_m span  x [{xy_m[:,0].min():.0f},{xy_m[:,0].max():.0f}]m "
          f"y [{xy_m[:,1].min():.0f},{xy_m[:,1].max():.0f}]m "
          f"(~{(xy_m[:,0].max()-xy_m[:,0].min())/1000:.0f}km × "
          f"{(xy_m[:,1].max()-xy_m[:,1].min())/1000:.0f}km)")
    print(f"  district cell counts: {np.bincount(district_idx).tolist()}")

if __name__ == "__main__":
    main()
