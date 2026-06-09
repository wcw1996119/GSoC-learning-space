"""B5: 疏解反事实 -> 拥堵变化(+ GE 可达性钩子)。

结构引力预测 OD 流量(跟可达性一致, 可外推): V_ij=γ·log(M_j)-decay·t_ij, segment-softmax -> P(j|i),
flow_ij = 出发地总量 × P。baseline M=岗位; 疏解 M=通州+核心-。
车需求(×车分担)聚节点 -> 自由流分配(B2 的 aon)-> 链路拥堵。对比 baseline vs 疏解: 哪片升/降。
用法: python experiments/scenario_congestion.py --period 0
"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from traffic_assignment import aon, PERIOD_HOURS, ALPHA, BETA, VC_CAP, TARGET_VC

PROC = ROOT / "data" / "processed"
TONGZHOU = 13; CORE = [0, 12]; GAMMA = 0.7; DECAY = 0.0947; DELAY_CAP = 2.9


def grav_flow(o, d, seg_ptr, logM, t, To):
    """结构引力 OD 流量: P=segment_softmax(γ·logM_d - decay·t), flow=To·P。"""
    V = GAMMA * logM[d] - DECAY * t
    nseg = len(seg_ptr) - 1; segl = np.diff(seg_ptr); seg_id = np.repeat(np.arange(nseg), segl)
    smax = np.maximum.reduceat(V, seg_ptr[:-1])
    expV = np.exp(V - smax[seg_id])
    ssum = np.add.reduceat(expV, seg_ptr[:-1])
    P = expV / ssum[seg_id]
    return To[seg_id] * P


def car_dmap(car, onode, dnode, Nn, keep_cov=0.97):
    key = onode * Nn + dnode; uk, inv = np.unique(key, return_inverse=True); dem = np.bincount(inv, weights=car)
    o_of = uk // Nn; d_of = uk % Nn
    oo = np.unique(o_of); odem = np.bincount(np.searchsorted(oo, o_of), weights=dem)
    order = np.argsort(-odem); cum = np.cumsum(odem[order]) / odem.sum()
    keep = oo[order[:np.searchsorted(cum, keep_cov) + 1]]
    dmap = {oi: (d_of[o_of == oi].astype(np.int64), dem[o_of == oi]) for oi in keep}
    return keep, dmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=0); ap.add_argument("--add-tongzhou", type=float, default=300000.0)
    ap.add_argument("--core-cut", type=float, default=0.2); a = ap.parse_args()
    net = np.load(PROC / "beijing_road_network.npz", allow_pickle=True)
    ed = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    mm = np.load(PROC / "beijing_micro_moments.npz", allow_pickle=True)
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)

    Nn = int(net["n_nodes"]); eu = net["edge_u"].astype(np.int64); ev = net["edge_v"].astype(np.int64)
    eff = net["edge_ff_min"].astype(np.float64); ecap = net["edge_cap"].astype(np.float64)
    nxy = net["node_xy"].astype(np.float64); cell_node = net["cell_node"].astype(np.int64)
    ekey = eu * Nn + ev; sidx = np.argsort(ekey); skey = ekey[sidx]; zn = np.arange(Nn)

    o = ed["o_idx"].astype(np.int64); d = ed["d_idx"].astype(np.int64); t = ed["t_obs"].astype(np.float64)
    hav = ed["hav_km"]; flow = ed["flow"][:, a.period].astype(np.float64); seg_ptr = ed["seg_ptr"].astype(np.int64)
    band = np.clip(np.digitize(hav, [1, 5, 15]), 0, 3); cs = mm["mode_by_dist"][:, 0].astype(np.float64)[band]
    To = np.add.reduceat(flow, seg_ptr[:-1])     # 出发地总量
    jobs = grid["jobs"].astype(np.float64); district = grid["district_idx"]
    onode = cell_node[o]; dnode = cell_node[d]

    def congest(logM):
        fl = grav_flow(o, d, seg_ptr, logM, t, To); car = fl * cs
        keep, dmap = car_dmap(car, onode, dnode, Nn)
        lv = aon(eu, ev, eff.copy(), Nn, keep, dmap, zn, skey, sidx)
        vc = lv / (ecap * PERIOD_HOURS[a.period] + 1e-9)
        vc *= TARGET_VC / max(np.median(vc[vc > 0]), 1e-6)
        dfac = np.minimum(1 + ALPHA * vc ** BETA, DELAY_CAP)   # 延误倍数(封顶实测2.9)
        tot_tt = float((lv * eff * dfac).sum())                # 总通勤时间(车·分)
        tot_delay = float((lv * eff * (dfac - 1)).sum())       # 其中拥堵延误部分
        return vc, tot_tt, tot_delay

    logM0 = np.log(jobs + 1.0)
    jobs1 = jobs.copy(); tc = np.where(district == TONGZHOU)[0]; cc = np.where(np.isin(district, CORE))[0]
    w = jobs[tc] + 1.0; jobs1[tc] += a.add_tongzhou * w / w.sum(); jobs1[cc] *= (1 - a.core_cut)
    logM1 = np.log(jobs1 + 1.0)
    print(f"=== B5 疏解拥堵 period {a.period} ===\n通州+{a.add_tongzhou:.0f} 核心×{1-a.core_cut:.1f}")
    vc0, tt0, dl0 = congest(logM0); vc1, tt1, dl1 = congest(logM1)
    print(f"  ⭐ 总通勤时间(车·分): {tt0:.3e} -> {tt1:.3e} ({(tt1/tt0-1)*100:+.2f}%)")
    print(f"  ⭐ 其中拥堵延误: {dl0:.3e} -> {dl1:.3e} ({(dl1/dl0-1)*100:+.2f}%)  "
          f"<- 这个降=疏解真减了系统拥堵, 升=只挪没减")

    # 链路 -> cell, 拥堵变化
    mid = (nxy[eu] + nxy[ev]) / 2
    ll = grid["coords_latlon"].astype(np.float64); tr = Transformer.from_crs(4326, 32650, always_xy=True)
    cx, cy = tr.transform(ll[:, 1], ll[:, 0]); cellxy = np.stack([cx, cy], 1)
    _, lc = cKDTree(cellxy).query(mid); N = len(ll); cnt = np.bincount(lc, minlength=N)
    c0 = np.bincount(lc, weights=vc0, minlength=N) / (cnt + 1e-9)
    c1 = np.bincount(lc, weights=vc1, minlength=N) / (cnt + 1e-9)
    dvc = c1 - c0
    def zone_mean(cells): m = np.isin(np.arange(N), cells) & (cnt > 0); return c0[m].mean(), c1[m].mean()
    t0, t1 = zone_mean(tc); k0, k1 = zone_mean(cc)
    print(f"  通州区 V/C: {t0:.2f} -> {t1:.2f} ({(t1/t0-1)*100:+.1f}%)")
    print(f"  核心区 V/C: {k0:.2f} -> {k1:.2f} ({(k1/k0-1)*100:+.1f}%)")
    valid = cnt > 0
    print(f"  全市 V/C 均值: {c0[valid].mean():.3f} -> {c1[valid].mean():.3f} ({(c1[valid].mean()/c0[valid].mean()-1)*100:+.1f}%)")
    up = (dvc > 0.02) & valid; dn = (dvc < -0.02) & valid
    print(f"  拥堵上升的格 {up.sum()}, 下降的格 {dn.sum()}; 最堵升幅 top3 区: ", end="")
    topcells = np.argsort(-dvc)[:200]; tud = np.bincount(district[topcells], minlength=16)
    names = grid["district_names"]
    print(", ".join(f"{names[i]}({tud[i]})" for i in np.argsort(-tud)[:3]))
    np.savez_compressed(PROC / f"beijing_congest_scenario_p{a.period}.npz",
                        cell_vc0=c0.astype(np.float32), cell_vc1=c1.astype(np.float32))
    print(f"  saved beijing_congest_scenario_p{a.period}.npz")


if __name__ == "__main__":
    main()
