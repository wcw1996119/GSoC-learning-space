"""B2: 一阶拥堵模型 = 节点级自由流分配 + 封顶 BPR + 实测标定。

车 OD(人流×车分担) 聚合到网络节点 -> 自由流全有全无分配(每起点 dijkstra+子树级联) -> 链路流 V。
拥堵: V/C(需求自动缩放到实际中位 ~0.6), t_cong=t_ff(1+α·min(V/C,cap)^β)(封顶防爆)。
验证: 链路 V/C 聚 cell vs 实测 cong_frac_period 相关性。
诚实定位: 一阶(无拥堵重路由), 只主干路, 单日实测 -> corr~0.35 可接受。
用法: python experiments/traffic_assignment.py --period 0
"""
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
PERIOD_HOURS = [3.0, 3.0, 6.0, 8.0]
ALPHA, BETA, VC_CAP = 0.15, 4.0, 3.0
TARGET_VC = 0.6   # 需求缩放目标(实际拥堵区中位 V/C)


def aon(eu, ev, times, Nn, origins, dmap, zone_node, skey, sidx):
    graph = sp.csr_matrix((times, (eu, ev)), shape=(Nn, Nn))
    lv = np.zeros(len(eu)); nodes = np.arange(Nn)
    for oi in origins:
        dn, dv = dmap[oi]
        dist, pred = dijkstra(graph, indices=int(zone_node[oi]), return_predecessors=True)
        vol = np.zeros(Nn); fin = np.isfinite(dist[dn]); np.add.at(vol, dn[fin], dv[fin])
        finite = np.isfinite(dist)
        for nd in nodes[finite][np.argsort(-dist[finite])]:
            p = pred[nd]
            if p >= 0: vol[p] += vol[nd]
        valid = (pred >= 0) & finite
        ks = pred[valid] * Nn + nodes[valid]
        pos = np.searchsorted(skey, ks).clip(0, len(skey) - 1); ok = skey[pos] == ks
        np.add.at(lv, sidx[pos[ok]], vol[valid][ok])
    return lv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--period", type=int, default=0); a = ap.parse_args()
    net = np.load(PROC / "beijing_road_network.npz", allow_pickle=True)
    ed = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    mm = np.load(PROC / "beijing_micro_moments.npz", allow_pickle=True)
    cg = np.load(PROC / "beijing_dynamic_cong.npz", allow_pickle=True)
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)

    Nn = int(net["n_nodes"]); eu = net["edge_u"].astype(np.int64); ev = net["edge_v"].astype(np.int64)
    eff = net["edge_ff_min"].astype(np.float64); ecap = net["edge_cap"].astype(np.float64)
    nxy = net["node_xy"].astype(np.float64); cell_node = net["cell_node"].astype(np.int64)

    o = ed["o_idx"]; d = ed["d_idx"]; hav = ed["hav_km"]; flow = ed["flow"][:, a.period].astype(np.float64)
    band = np.clip(np.digitize(hav, [1, 5, 15]), 0, 3)
    car = flow * mm["mode_by_dist"][:, 0].astype(np.float64)[band]
    onode = cell_node[o]; dnode = cell_node[d]
    key = onode * Nn + dnode; uk, inv = np.unique(key, return_inverse=True); dem = np.bincount(inv, weights=car)
    o_of = uk // Nn; d_of = uk % Nn
    oo = np.unique(o_of); odem = np.bincount(np.searchsorted(oo, o_of), weights=dem)
    order = np.argsort(-odem); cum = np.cumsum(odem[order]) / odem.sum()
    keep = oo[order[:np.searchsorted(cum, 0.97) + 1]]
    dmap = {}
    for oi in keep:
        msk = o_of == oi; dmap[oi] = (d_of[msk].astype(np.int64), dem[msk])
    print(f"  period {a.period}: 车需求{dem.sum():.0f}, 起点{len(oo)}->{len(keep)}(97%)")

    ekey = eu * Nn + ev; sidx = np.argsort(ekey); skey = ekey[sidx]
    zone_node = np.arange(Nn)   # 节点级: 起点就是节点本身
    lv = aon(eu, ev, eff.copy(), Nn, keep, dmap, zone_node, skey, sidx)

    # 需求自动缩放 -> 实际拥堵区中位 V/C = TARGET_VC
    vc_raw = lv / (ecap * PERIOD_HOURS[a.period] + 1e-9)
    scale = TARGET_VC / max(np.median(vc_raw[vc_raw > 0]), 1e-6)
    vc = vc_raw * scale
    t_cong = eff * (1 + ALPHA * np.minimum(vc, VC_CAP) ** BETA)
    print(f"  V/C(缩放后 ×{scale:.2f}): 中位{np.median(vc[vc>0]):.2f} 90%{np.percentile(vc,90):.2f} "
          f">1:{(vc>1).mean()*100:.1f}%; 延误中位{np.median((t_cong/eff)[lv>0]):.2f} max{np.max(t_cong/eff):.1f}")

    # 验证 vs 实测
    mid = (nxy[eu] + nxy[ev]) / 2
    ll = grid["coords_latlon"].astype(np.float64)
    tr = Transformer.from_crs(4326, 32650, always_xy=True); cx, cy = tr.transform(ll[:, 1], ll[:, 0])
    cellxy = np.stack([cx, cy], 1); _, link_cell = cKDTree(cellxy).query(mid)
    N = len(ll); cnt = np.bincount(link_cell, minlength=N)
    cell_vc = np.bincount(link_cell, weights=vc, minlength=N) / (cnt + 1e-9)
    obs = cg["cong_frac_period"][:, a.period]; m = cg["cover_mask"] & (cnt > 0) & (obs >= 0)
    c = np.corrcoef(cell_vc[m], obs[m])[0, 1]
    cs = np.corrcoef(np.argsort(np.argsort(cell_vc[m])), np.argsort(np.argsort(obs[m])))[0, 1]
    print(f"  ⭐ 标定: 模型 cell V/C vs 实测拥堵 Pearson={c:.3f} Spearman={cs:.3f} (n={m.sum()})")
    np.savez_compressed(PROC / f"beijing_assign_p{a.period}.npz", vc=vc.astype(np.float32),
                        t_cong=t_cong.astype(np.float32), link_vol=lv.astype(np.float32),
                        link_cell=link_cell.astype(np.int32), scale=np.float32(scale))
    print(f"  saved beijing_assign_p{a.period}.npz")


if __name__ == "__main__":
    main()
