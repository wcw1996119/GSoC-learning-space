"""B1: 从道路中心线 SHP 建可路由主干路网 (节点-链路, 带容量/自由流时间) -> beijing_road_network.npz

源: 202306北京市道路中心线.shp (114万段, ROAD_CLASS), 剔除四级街巷 -> 主干 ~9.4万段。
做法: 端点按 TOL 吸附成节点 -> 节点-链路图; 按等级配自由流速+容量; cell 接最近节点。
自检: 最大连通分量占比 + cell 接入率 + 抽样最短路时间 vs edges 的 t_ff。
环境: base anaconda (geopandas/pyogrio/networkx/scipy)。governance: 输出 gitignored。
"""
from pathlib import Path
import numpy as np, pyogrio, networkx as nx
from shapely import get_coordinates
from scipy.spatial import cKDTree

ROAD = Path(r"D:/UoM/congestion_project/202306北京市道路中心线/北京市道路中心线.shp")
PROC = Path(__file__).resolve().parents[1] / "processed"
OUT = PROC / "beijing_road_network.npz"
TOL = 25.0  # m 端点吸附

# 等级 -> 自由流速(km/h) / 单向通行能力(veh/h)
SPEED = {"高速公路": 100, "快速路": 80, "国道": 70, "省道": 60, "二级道路": 50, "三级道路": 40}
CAP = {"高速公路": 4000, "快速路": 3000, "国道": 1800, "省道": 1500, "二级道路": 1200, "三级道路": 800}
CLS = list(SPEED)  # 主干(剔四级)


def main():
    print("读 SHP (只主干)...")
    g = pyogrio.read_dataframe(ROAD, columns=["ROAD_CLASS", "geometry"])
    g = g[g["ROAD_CLASS"].isin(CLS)].reset_index(drop=True)
    print(f"  主干段 {len(g)}")

    nodes = {}
    def nid(x, y):
        k = (round(x / TOL), round(y / TOL))
        i = nodes.get(k)
        if i is None: i = nodes[k] = len(nodes)
        return i

    eu, ev, elen, eff, ecap, ecls = [], [], [], [], [], []
    geoms = g.geometry.values; classes = g["ROAD_CLASS"].values
    for geom, cls in zip(geoms, classes):
        if geom is None: continue
        cs = get_coordinates(geom)
        if len(cs) < 2: continue
        u = nid(cs[0, 0], cs[0, 1]); v = nid(cs[-1, 0], cs[-1, 1])
        if u == v: continue
        L = float(geom.length)                       # m (EPSG:32650)
        eu.append(u); ev.append(v); elen.append(L)
        eff.append(L / 1000.0 / SPEED[cls] * 60.0)   # min
        ecap.append(CAP[cls]); ecls.append(CLS.index(cls))
    nn = len(nodes)
    node_xy = np.zeros((nn, 2), np.float64)
    for k, i in nodes.items(): node_xy[i] = (k[0] * TOL, k[1] * TOL)
    eu = np.array(eu); ev = np.array(ev); elen = np.array(elen, np.float32)
    eff = np.array(eff, np.float32); ecap = np.array(ecap, np.float32); ecls = np.array(ecls, np.int8)
    print(f"  节点 {nn}, 链路 {len(eu)}")

    # 连通性
    G = nx.Graph(); G.add_edges_from(zip(eu.tolist(), ev.tolist()))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = comps[0]; print(f"  最大连通分量 {len(giant)}/{nn} ({len(giant)/nn*100:.1f}%)")

    # cell 接最近节点 (grid 经纬度 -> EPSG:32650, 跟 SHP 同 CRS)
    from pyproj import Transformer
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    ll = grid["coords_latlon"].astype(np.float64)   # (N,2) lat,lon
    tr = Transformer.from_crs(4326, 32650, always_xy=True)
    cx, cy = tr.transform(ll[:, 1], ll[:, 0])        # always_xy: (lon, lat)
    cell_xy = np.stack([cx, cy], axis=1)
    gnodes = np.array(sorted(giant)); tree = cKDTree(node_xy[gnodes])
    dist, gi = tree.query(cell_xy); cell_node = gnodes[gi]
    print(f"  cell 接入: 中位距 {np.median(dist):.0f}m, 90分位 {np.percentile(dist,90):.0f}m, "
          f">500m 的 {(dist>500).mean()*100:.1f}%")

    # 自检: 抽样最短路自由流时间 vs edges t_ff
    Gw = nx.DiGraph()
    for u, v, t in zip(eu.tolist(), ev.tolist(), eff.tolist()):
        Gw.add_edge(u, v, w=t); Gw.add_edge(v, u, w=t)   # 双向
    e = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    o = e["o_idx"]; d = e["d_idx"]; tff = e["t_ff"]
    rng = np.random.default_rng(0); idx = rng.choice(len(o), 60, replace=False)
    net_t, obs_t = [], []
    for ii in idx:
        a, b = cell_node[o[ii]], cell_node[d[ii]]
        try:
            tt = nx.shortest_path_length(Gw, a, b, weight="w")
            net_t.append(tt); obs_t.append(float(tff[ii]))
        except Exception: pass
    net_t, obs_t = np.array(net_t), np.array(obs_t)
    if len(net_t):
        corr = np.corrcoef(net_t, obs_t)[0, 1]
        print(f"  抽样 {len(net_t)} OD: 网络最短路自由流 vs edges t_ff  corr={corr:.3f}  "
              f"网络中位 {np.median(net_t):.1f}min / t_ff 中位 {np.median(obs_t):.1f}min")

    np.savez_compressed(OUT, node_xy=node_xy.astype(np.float32), edge_u=eu.astype(np.int32),
                        edge_v=ev.astype(np.int32), edge_len_m=elen, edge_ff_min=eff,
                        edge_cap=ecap, edge_cls=ecls, cell_node=cell_node.astype(np.int32),
                        n_nodes=np.int64(nn), giant_nodes=gnodes.astype(np.int32))
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
