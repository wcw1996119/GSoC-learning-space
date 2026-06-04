"""Phase 1b: 三套 per-mode 出行时间 + mode_share 先验 -> beijing_modes.npz

依赖: data/processed/beijing_edges.npz (o_idx,d_idx,hav_km,t_obs,t_ff 已按 CSR 排好)
出:   data/processed/beijing_modes.npz (gitignored), 与 edge 顺序一一对应

伦敦对应: make_distance_aware_mode_share -> t_per_mode dict + pair_mode_share

设计 (CLAUDE.md "London BPR / 北京高德, 可换"):
  车 t_car   : trainer 里从 t_obs(拥堵)/t_ff(自由流) 按时段派生, 本 builder 不存
  步行 t_walk: hav_km / 步速(5km/h), 与时段无关
  公交 t_transit (v1, 标定型, 长杆任务的第一版):
     = 接驳步行 + 等待(发车间隔/2) + 车内(hav_km / v_bus) + 长途换乘缓冲
     v_bus = 13.4 km/h  <- bus_crawl/bus_status 244K 轨迹样本实测中位速度
     ⚠ Caveats: ① 只有公交无地铁(系统偏高) ② 单日 ③ 简化非网络路由; 后续精修
  mode_share 先验 (软锚, ce_mode 正则用, kl_weight 小): 距离感知, car/transit/walk

后续精修(标 TODO): 用 01北京市公交 线网 shp + timetable 做真网络路由 + 接入地铁。
"""
import numpy as np
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "processed"
EDGES = PROC / "beijing_edges.npz"
OUT = PROC / "beijing_modes.npz"

WALK_SPEED_KMH = 5.0
WALK_THRESHOLD_KM = 5.0          # 超过此距离 walk 先验 share -> 0 (伦敦同款)
V_BUS_KMH = 13.4                 # bus_crawl 实测中位运行速度 (含停站/拥堵)
BUS_ACCESS_EGRESS_MIN = 10.0     # 接驳+下车步行 (北京站点密度高, ~5min 各端)
BUS_WAIT_MIN = 5.0               # 等待 ≈ 发车间隔/2 (北京公交典型间隔 ~10min)
BUS_TRANSFER_PER_10KM = 5.0      # 长途换乘缓冲

def main():
    e = np.load(EDGES)
    hav = e["hav_km"].astype(np.float64)
    E = len(hav)

    # ---- 步行 ----
    t_walk = (hav / WALK_SPEED_KMH * 60.0).astype(np.float32)

    # ---- 公交 v2 (刷卡实测标定, 含地铁) ----
    # 刷卡 20190506 真实行程时间距离档中位 (build_beijing_transit_od.py 实测):
    #   0-2km 5.7 / 2-5km 14.4 / 5-10km 26.1 / 10-20km 41.4 / 20+km 65.3 min
    # 用档中位距离→时间插值 (远超我原代理: 原 5-10km=52min, 实测仅 26min)
    SC_KM = np.array([1.0, 3.5, 7.5, 15.0, 30.0])
    SC_MIN = np.array([5.7, 14.4, 26.1, 41.4, 65.3])
    t_transit = np.interp(hav, SC_KM, SC_MIN).astype(np.float32)  # 端点外用边界值

    # ---- mode_share 先验 (距离感知, 顺序 = [car, transit, walk]) ----
    # walk: 短程主导, 指数衰减; >阈值置 0
    raw_walk = np.exp(-hav / 1.5)
    raw_walk[hav > WALK_THRESHOLD_KM] = 0.0
    # car: 随距离上升
    raw_car = 0.20 + 0.50 / (1.0 + np.exp(-(hav - 8.0) / 4.0))
    # transit: 中距离峰
    raw_transit = 0.30 + 0.50 * np.exp(-((hav - 6.0) / 6.0) ** 2)
    stk = np.stack([raw_car, raw_transit, raw_walk], axis=1)
    mode_share = (stk / stk.sum(axis=1, keepdims=True)).astype(np.float32)  # (E,3)

    np.savez_compressed(
        OUT,
        t_walk=t_walk,
        t_transit=t_transit,
        mode_share=mode_share,
        v_bus_kmh=np.float32(V_BUS_KMH),
        walk_speed_kmh=np.float32(WALK_SPEED_KMH),
    )

    # ---- 数据自检 ----
    # 对照: 车时间用 t_obs (从 edges)
    t_car_obs = e["t_obs"].astype(np.float64)
    print(f"[OK] {OUT}  E={E:,}")
    print(f"  t_walk    range [{t_walk.min():.1f},{t_walk.max():.1f}] mean {t_walk.mean():.1f} min")
    print(f"  t_transit range [{t_transit.min():.1f},{t_transit.max():.1f}] mean {t_transit.mean():.1f} min")
    print(f"  t_car(obs)range [{t_car_obs.min():.1f},{t_car_obs.max():.1f}] mean {t_car_obs.mean():.1f} min")
    print(f"  时间排序检验 (按距离档, 取中位):")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40)]:
        m = (hav >= lo) & (hav < hi)
        if m.sum() == 0: continue
        print(f"    {lo:2d}-{hi:2d}km (n={m.sum():>9,}): "
              f"car {np.median(t_car_obs[m]):5.1f}  "
              f"transit {np.median(t_transit[m]):5.1f}  "
              f"walk {np.median(t_walk[m]):6.1f}  min")
    print(f"  mode_share 均值 [car,transit,walk] = {mode_share.mean(0).round(3).tolist()}")
    print(f"  按距离档 mode_share:")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40)]:
        m = (hav >= lo) & (hav < hi)
        if m.sum() == 0: continue
        print(f"    {lo:2d}-{hi:2d}km: {mode_share[m].mean(0).round(3).tolist()}")
    print(f"  [!] 公交时间只含公交无地铁 (系统偏高); v_bus={V_BUS_KMH}km/h 实测标定")

if __name__ == "__main__":
    main()
