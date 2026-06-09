"""Paper B option2 阶段1: 真实北京【需求→供给→拥堵→反馈】闭环 (三模块连通).

按用户 ABM 设计:
  ① 需求(选择模型, 去confound β_ff): agent 按【自由流】时间选目的地 -> OD 车流
  ② 供给(分配+BPR, 复用 aon): 车流压网络 -> 链路拥堵 -> 拥堵后 OD 时间
  ③ 反馈: 把拥堵时间喂回选择模型 -> 目的地【改变】= forced choice 在真实北京的量级
验证: 选择模型车流分配出的拥堵 vs 实测 (Spearman); 反馈后目的地分布漂移多少。
诚实: 一阶分配(无重路由); 完整可微/部分预判 pin β 是后续研究块。本地 CPU(分配较慢, 几分钟)。
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from pyproj import Transformer
from traffic_assignment import aon, PROC, PERIOD_HOURS, ALPHA, BETA, VC_CAP, TARGET_VC

ALPHA_PREF, BETA_FF = 0.535, 0.177      # 去confound 偏好 (见 beijing_deconfound_simple)
PERIOD = 0                               # 早高峰


def seg_softmax(V, seg_ptr):
    P = np.empty_like(V)
    for i in range(len(seg_ptr) - 1):
        s, e = seg_ptr[i], seg_ptr[i + 1]
        if e > s:
            v = V[s:e]; v = v - v.max(); ex = np.exp(v); P[s:e] = ex / ex.sum()
    return P


def main():
    ed = np.load(PROC / "beijing_edges.npz", allow_pickle=True)
    net = np.load(PROC / "beijing_road_network.npz", allow_pickle=True)
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    mm = np.load(PROC / "beijing_micro_moments.npz", allow_pickle=True)
    cg = np.load(PROC / "beijing_dynamic_cong.npz", allow_pickle=True)

    o = ed["o_idx"].astype(np.int64); d = ed["d_idx"].astype(np.int64)
    t_ff = ed["t_ff"].astype(np.float64); t_obs = ed["t_obs"].astype(np.float64)
    hav = ed["hav_km"].astype(np.float64); seg_ptr = ed["seg_ptr"].astype(np.int64)
    flow_tot = ed["flow"][:, PERIOD].astype(np.float64)
    jobs = grid["jobs"].astype(np.float64); logM_d = np.log(jobs[d] + 1.0)

    # ── ① 需求: 选择模型(去confound β_ff), agent 按自由流选 ──
    P_ff = seg_softmax(ALPHA_PREF * logM_d - BETA_FF * t_ff, seg_ptr)
    # 出发地总需求 = 观测段总流量(用真实规模), 模型只重新分配目的地
    seg_tot = np.add.reduceat(flow_tot, seg_ptr[:-1])
    model_flow = P_ff * np.repeat(seg_tot, np.diff(seg_ptr))      # 模型预测 OD 车流(自由流选择)

    # ── ② 供给: 车流压网络 -> 分配 -> 拥堵 (复用 aon) ──
    Nn = int(net["n_nodes"]); eu = net["edge_u"].astype(np.int64); ev = net["edge_v"].astype(np.int64)
    eff = net["edge_ff_min"].astype(np.float64); ecap = net["edge_cap"].astype(np.float64)
    nxy = net["node_xy"].astype(np.float64); cell_node = net["cell_node"].astype(np.int64)
    band = np.clip(np.digitize(hav, [1, 5, 15]), 0, 3)
    car = model_flow * mm["mode_by_dist"][:, 0].astype(np.float64)[band]   # 车分担
    onode = cell_node[o]; dnode = cell_node[d]
    key = onode * Nn + dnode; uk, inv = np.unique(key, return_inverse=True); dem = np.bincount(inv, weights=car)
    o_of = uk // Nn; d_of = uk % Nn
    oo = np.unique(o_of); odem = np.bincount(np.searchsorted(oo, o_of), weights=dem)
    order = np.argsort(-odem); cum = np.cumsum(odem[order]) / odem.sum()
    keep = oo[order[:np.searchsorted(cum, 0.97) + 1]]
    dmap = {oi: (d_of[o_of == oi].astype(np.int64), dem[o_of == oi]) for oi in keep}
    ekey = eu * Nn + ev; sidx = np.argsort(ekey); skey = ekey[sidx]
    print(f"  分配中(模型车流, {len(keep)} 起点)...")
    lv = aon(eu, ev, eff.copy(), Nn, keep, dmap, np.arange(Nn), skey, sidx)
    vc_raw = lv / (ecap * PERIOD_HOURS[PERIOD] + 1e-9)
    scale = TARGET_VC / max(np.median(vc_raw[vc_raw > 0]), 1e-6); vc = vc_raw * scale
    cong_factor_link = 1 + ALPHA * np.minimum(vc, VC_CAP) ** BETA          # t_cong/t_ff per link

    # 验证: 模型车流分配的拥堵 vs 实测
    mid = (nxy[eu] + nxy[ev]) / 2; ll = grid["coords_latlon"].astype(np.float64)
    tr = Transformer.from_crs(4326, 32650, always_xy=True); cx, cy = tr.transform(ll[:, 1], ll[:, 0])
    _, link_cell = cKDTree(np.stack([cx, cy], 1)).query(mid)
    N = len(ll); cnt = np.bincount(link_cell, minlength=N)
    cell_vc = np.bincount(link_cell, weights=vc, minlength=N) / (cnt + 1e-9)
    obs = cg["cong_frac_period"][:, PERIOD]; m = cg["cover_mask"] & (cnt > 0) & (obs >= 0)
    cs = np.corrcoef(np.argsort(np.argsort(cell_vc[m])), np.argsort(np.argsort(obs[m])))[0, 1]
    print(f"  ⭐ 模型车流→分配→拥堵 vs 实测: Spearman={cs:.3f} (n={m.sum()})")

    # ── ③ 反馈: 拥堵后 OD 时间喂回选择 -> 目的地漂移 (forced choice) ──
    # 每条边的拥堵因子 = 其目的地 cell 周边链路平均
    cong_cell = np.bincount(link_cell, weights=cong_factor_link, minlength=N) / (cnt + 1e-9)
    cong_cell[cnt == 0] = np.median(cong_factor_link)
    t_cong_edge = t_ff * cong_cell[d]                                      # 拥堵后 OD 时间(模型自洽)
    P_cong = seg_softmax(ALPHA_PREF * logM_d - BETA_FF * t_cong_edge, seg_ptr)

    # 目的地漂移: 每出发地 |P_ff − P_cong| / 2 (total variation)
    tv = np.add.reduceat(np.abs(P_ff - P_cong), seg_ptr[:-1]) / 2
    print(f"\n  ③ forced choice(拥堵反馈→目的地漂移): TV 中位 {np.median(tv)*100:.1f}% "
          f"均值 {tv.mean()*100:.1f}% (= 平均 {tv.mean()*100:.0f}% 的人因拥堵改去向)")
    print(f"\n判读: 三模块在真实北京连通 [选择(β_ff)→分配→拥堵→反馈]; 模型车流分配的拥堵跟实测 Spearman={cs:.2f} "
          f"(≈观测天花板 0.76); 拥堵反馈让 ~{tv.mean()*100:.0f}% 的人改目的地 = forced choice 真实量级。"
          " 这是 option2 阶段1(闭环连通); pin β 的部分预判均衡是后续研究块。")


if __name__ == "__main__":
    main()
