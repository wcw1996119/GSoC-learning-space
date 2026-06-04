"""Mode 识别关键: 刷卡 -> 格子级公交(+地铁)OD + 真实公交时间 -> beijing_transit_od.npz

源: D:\\UoM\\congestion_project\\Public Transit Smart Card Data\\20190506.csv (10.6M 行)
    每行 = 一次公交/地铁出行: 起点站经纬度+出发时间 -> 终点站经纬度+到达时间, 起点站方式(公交/地铁)
依赖: beijing_grid.npz (coords, 投影一致)
出: data/processed/beijing_transit_od.npz (gitignored)

产出 (Ben-Akiva-Morikawa 联合估计的第二份 mode-labeled 观测流量):
  to_o, to_d (E2,)    公交 OD 的格子对 (出现过的)
  to_n (E2,)          该对公交+地铁出行数
  to_min (E2,)        平均行程时间(分钟) -> 校准 t_transit
  to_frac_sub (E2,)   地铁占比
  + 距离档 transit 真实出行时间表 (校准 t_transit 速度模型)

⚠ 数据是 2019-05-06 单日; 坐标 datum 可能与 cell 略偏(~1格blur, 聚合OD稳健)。
⚠ 刷卡(持卡人) vs 信令(手机用户) 分母不同 -> 不取绝对share, 用 transit OD 空间/距离分布当矩。
"""
import numpy as np
import pandas as pd
from pathlib import Path

CSV = Path(r"D:/UoM/congestion_project/Public Transit Smart Card Data/20190506.csv")
PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
OUT = PROC / "beijing_transit_od.npz"

def main():
    grid = np.load(GRID, allow_pickle=True)
    coords = grid["coords_latlon"].astype(np.float64)
    lat0, lon0 = float(np.median(coords[:,0])), float(np.median(coords[:,1]))
    def proj(lat, lon):
        x=(lon-lon0)*np.cos(np.deg2rad(lat0))*111320.0; y=(lat-lat0)*110540.0
        return np.stack([x,y],axis=1)
    cell_xy = proj(coords[:,0], coords[:,1])
    from scipy.spatial import cKDTree
    tree = cKDTree(cell_xy)

    cols = ["起点站方式","起点站经度","起点站纬度","出发时间",
            "终点站经度","终点站纬度","到达时间"]
    print("loading smartcard (10.6M rows) ...")
    df = pd.read_csv(CSV, usecols=cols)
    n0 = len(df)
    for c in ["起点站经度","起点站纬度","终点站经度","终点站纬度"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["起点站经度","起点站纬度","终点站经度","终点站纬度"])
    # 投影 + 最近格子
    o_xy = proj(df["起点站纬度"].values, df["起点站经度"].values)
    d_xy = proj(df["终点站纬度"].values, df["终点站经度"].values)
    _, o_cell = tree.query(o_xy, k=1)
    _, d_cell = tree.query(d_xy, k=1)
    df["o_cell"] = o_cell; df["d_cell"] = d_cell
    df["hav_km"] = np.linalg.norm(o_xy - d_xy, axis=1) / 1000.0
    df["t_min"] = (pd.to_datetime(df["到达时间"]) - pd.to_datetime(df["出发时间"])).dt.total_seconds()/60
    df = df[(df.t_min > 0) & (df.t_min < 240)]
    df["is_sub"] = (df["起点站方式"] == "地铁").astype(float)

    # 聚合 (o_cell,d_cell)
    g = df.groupby(["o_cell","d_cell"]).agg(
        n=("t_min","size"), t_min=("t_min","mean"), frac_sub=("is_sub","mean")).reset_index()
    print(f"  聚合: {len(df):,} 有效出行 -> {len(g):,} 格子对")

    # 距离档真实公交时间 (校准速度模型)
    df["band"] = np.digitize(df["hav_km"], [2,5,10,20])
    band_t = df.groupby("band")["t_min"].agg(["median","mean","size"])

    np.savez_compressed(OUT,
        to_o=g.o_cell.values.astype(np.int32), to_d=g.d_cell.values.astype(np.int32),
        to_n=g.n.values.astype(np.float32), to_min=g.t_min.values.astype(np.float32),
        to_frac_sub=g.frac_sub.values.astype(np.float32))

    print(f"[OK] {OUT}")
    print(f"  原始 {n0:,} 行, 有效 {len(df):,}")
    print(f"  公交格子对 {len(g):,}, 总出行 {g.n.sum():,.0f}")
    print(f"  地铁占比(出行加权) {(df.is_sub).mean():.3f}")
    print(f"  行程时间(分钟) median {df.t_min.median():.1f}")
    print(f"  距离档真实公交时间:")
    bn = ["0-2","2-5","5-10","10-20","20+"]
    for b,r in band_t.iterrows():
        print(f"    {bn[int(b)]:>5}km: median {r['median']:.1f} mean {r['mean']:.1f} min (n={int(r['size']):,})")
    print(f"  公交出行距离(hav) median {df.hav_km.median():.1f}km mean {df.hav_km.mean():.1f}km")

if __name__ == "__main__":
    main()
