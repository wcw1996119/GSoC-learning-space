"""B1: OSMnx 北京主干驾驶路网 -> 可路由节点-链路(自由流时间+容量) -> beijing_road_network.npz

osmnx 拉 bbox(=cell 范围)内主干路(motorway..tertiary + ramps), 加速度/自由流时间, 投影UTM。
节点-链路 + 容量 by highway type; cell centroid -> 最近节点。
自检: 弱连通占比 + cell 接入距 + 抽样最短路自由流 vs edges t_ff 相关性。
环境: base anaconda (osmnx 2.1)。governance: 输出 gitignored。
"""
from pathlib import Path
import numpy as np, osmnx as ox, networkx as nx
from pyproj import Transformer
from scipy.spatial import cKDTree

PROC = Path(__file__).resolve().parents[1] / "processed"
OUT = PROC / "beijing_road_network.npz"
MAJOR = ('["highway"~"motorway|trunk|primary|secondary|tertiary|'
         'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]')
# highway type -> 单向通行能力 veh/h
CAP = {"motorway": 4000, "trunk": 3000, "primary": 2000, "secondary": 1500, "tertiary": 1000,
       "motorway_link": 1500, "trunk_link": 1200, "primary_link": 1000, "secondary_link": 800,
       "tertiary_link": 600}


def first(x): return x[0] if isinstance(x, list) else x


def main():
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    ll = grid["coords_latlon"].astype(np.float64)   # (N,2) lat,lon
    m = 0.02
    n, s = ll[:, 0].max() + m, ll[:, 0].min() - m
    e, w = ll[:, 1].max() + m, ll[:, 1].min() - m
    print(f"bbox lat[{s:.2f},{n:.2f}] lon[{w:.2f},{e:.2f}], 拉 OSM 主干路...")
    G = ox.graph_from_bbox((w, s, e, n), network_type="drive", custom_filter=MAJOR, simplify=True)
    print(f"  raw: {G.number_of_nodes()} 节点 / {G.number_of_edges()} 链路")
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    Gp = ox.project_graph(G)
    crs = Gp.graph["crs"]

    nid = list(Gp.nodes); idx = {x: i for i, x in enumerate(nid)}
    node_xy = np.array([[Gp.nodes[x]["x"], Gp.nodes[x]["y"]] for x in nid], np.float64)
    eu, ev, elen, eff, ecap = [], [], [], [], []
    for u, v, dd in Gp.edges(data=True):
        eu.append(idx[u]); ev.append(idx[v])
        elen.append(float(dd.get("length", 0.0)))
        eff.append(float(dd.get("travel_time", 0.0)) / 60.0)        # min
        ecap.append(CAP.get(first(dd.get("highway", "tertiary")), 800))
    eu = np.array(eu); ev = np.array(ev); elen = np.array(elen, np.float32)
    eff = np.array(eff, np.float32); ecap = np.array(ecap, np.float32)
    nn = len(nid); print(f"  投影后: {nn} 节点 / {len(eu)} 链路 (CRS {crs})")

    # 弱连通(有向)
    DG = nx.DiGraph(); DG.add_edges_from(zip(eu.tolist(), ev.tolist()))
    wcc = sorted(nx.weakly_connected_components(DG), key=len, reverse=True)
    giant = np.array(sorted(wcc[0])); print(f"  最大弱连通 {len(giant)}/{nn} ({len(giant)/nn*100:.1f}%)")

    # cell -> 最近节点 (cell lat/lon -> 同 UTM)
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    cx, cy = tr.transform(ll[:, 1], ll[:, 0]); cell_xy = np.stack([cx, cy], 1)
    tree = cKDTree(node_xy[giant]); dist, gi = tree.query(cell_xy); cell_node = giant[gi]
    print(f"  cell 接入: 中位 {np.median(dist):.0f}m, 90分位 {np.percentile(dist,90):.0f}m, "
          f">1km {(dist>1000).mean()*100:.1f}%")

    # 自检: 抽样最短路自由流时间 vs edges t_ff
    Gt = nx.DiGraph()
    for u, v, t in zip(eu.tolist(), ev.tolist(), eff.tolist()): Gt.add_edge(u, v, w=t)
    ed = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    oo, ddi, tff = ed["o_idx"], ed["d_idx"], ed["t_ff"]
    rng = np.random.default_rng(0); samp = rng.choice(len(oo), 80, replace=False)
    nt, ot = [], []
    for ii in samp:
        try:
            tt = nx.shortest_path_length(Gt, cell_node[oo[ii]], cell_node[ddi[ii]], weight="w")
            if np.isfinite(tt): nt.append(tt); ot.append(float(tff[ii]))
        except Exception: pass
    nt, ot = np.array(nt), np.array(ot)
    if len(nt) > 3:
        print(f"  自检 {len(nt)}/80 OD 可路由: 网络自由流 vs t_ff  corr={np.corrcoef(nt,ot)[0,1]:.3f}  "
              f"网络中位 {np.median(nt):.1f} / t_ff 中位 {np.median(ot):.1f} min")

    np.savez_compressed(OUT, node_xy=node_xy.astype(np.float32), edge_u=eu.astype(np.int32),
                        edge_v=ev.astype(np.int32), edge_len_m=elen, edge_ff_min=eff, edge_cap=ecap,
                        cell_node=cell_node.astype(np.int32), n_nodes=np.int64(nn),
                        giant_nodes=giant.astype(np.int32), crs=str(crs))
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
