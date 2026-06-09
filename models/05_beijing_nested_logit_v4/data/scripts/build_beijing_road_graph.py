"""图构造升级: 路网中心线 -> cell 间路网连通图 -> beijing_road_graph.npz

源: D:/UoM/congestion_project/202306北京市道路中心线/北京市道路中心线.shp (114万段 LineString, EPSG:32650, ROAD_CLASS)
依赖: cell_index.parquet (16907 格中心 lat/lon -> 投到 32650)
出: data/processed/beijing_road_graph.npz (gitignored)

算法 (分段端点跨格连边): 每段路两端点落最近 cell, 两端在不同 cell -> 连边, 次数=边权。
  = 两格之间有路跨过边界才连(隔河/铁路的空间近邻不会连)。对称化。
孤立格(无路跨界)回退: 补该格的空间最近邻 1 条边, 保证图连通无孤点。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd
from scipy.spatial import cKDTree

PAPERC = Path(r"D:/UoM/PaperC_local")
ROAD = Path(r"D:/UoM/congestion_project/202306北京市道路中心线/北京市道路中心线.shp")
OUT = Path(__file__).resolve().parents[1] / "processed" / "beijing_road_graph.npz"
CUTOFF_M = 1200.0

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ci = pq.read_table(PAPERC / "cell_index.parquet").to_pandas().sort_values("idx").reset_index(drop=True)
    N = len(ci)
    cells = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ci.lon, ci.lat), crs=4326).to_crs(32650)
    cxy = np.stack([cells.geometry.x.values, cells.geometry.y.values], axis=1)
    tree = cKDTree(cxy)

    print("  读路网 114万段 (可能 1-2 分钟)...")
    import time; t0 = time.time()
    g = gpd.read_file(ROAD, columns=["geometry"])
    print(f"  读完 {len(g):,} 段 ({time.time()-t0:.0f}s), 取端点...")
    sp = g.geometry.interpolate(0.0, normalized=True)
    ep = g.geometry.interpolate(1.0, normalized=True)
    sxy = np.stack([sp.x.values, sp.y.values], axis=1)
    exy = np.stack([ep.x.values, ep.y.values], axis=1)
    ds, si = tree.query(sxy); de, ei = tree.query(exy)
    ok = (ds <= CUTOFF_M) & (de <= CUTOFF_M) & (si != ei)   # 两端有效且跨格
    a = si[ok]; b = ei[ok]
    print(f"  跨格路段 {ok.sum():,} / {len(g):,}")

    # 无向边权: (min,max) key 计数
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    key = lo.astype(np.int64) * N + hi.astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    eu = uk // N; ev = uk % N                                 # 无向边两端
    deg = np.bincount(np.concatenate([eu, ev]), minlength=N)

    # 孤立格回退: 补空间最近邻 1 条
    iso = np.where(deg == 0)[0]
    if len(iso):
        _, nn = tree.query(cxy[iso], k=2)                    # k=2: 自己+最近邻
        fb_u = iso; fb_v = nn[:, 1]
        eu = np.concatenate([eu, fb_u]); ev = np.concatenate([ev, fb_v])
        cnt = np.concatenate([cnt, np.ones(len(iso))])

    # 对称化 -> 有向 edge_index (2,2E)
    src = np.concatenate([eu, ev]); dst = np.concatenate([ev, eu])
    w = np.concatenate([cnt, cnt]).astype(np.float32)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    deg_final = np.bincount(edge_index[0], minlength=N)

    np.savez_compressed(OUT, edge_index=edge_index, edge_weight=w, N=np.int64(N))
    print(f"[OK] {OUT}")
    print(f"  无向边 {len(uk):,}, 有向 {edge_index.shape[1]:,}")
    print(f"  度: mean {deg_final.mean():.1f}, median {int(np.median(deg_final))}, max {deg_final.max()}")
    print(f"  孤立格回退补了 {len(iso)} 个; 现在 0 度格 {int((deg_final==0).sum())}")
    print(f"  边权(跨界路段数): mean {w.mean():.1f}, max {int(w.max())}")
    print(f"  对照: 旧空间kNN 固定度=8")

if __name__ == "__main__":
    main()
