"""Phase 1a: 稀疏候选集 + 4 时段观测流量 -> beijing_edges.npz

源 (只读, 不出库):
  D:\\UoM\\PaperC_local\\t_matrix.parquet   30.4M 行 (o_idx,d_idx,hav_km,t_obs_min,t_ff_min)
  D:\\UoM\\PaperC_local\\od_cell_flow.parquet 7.06M 行 (o_cell,d_cell,f_ampeak,f_pmpeak,f_midday,f_night)
  data/processed/beijing_grid.npz  (residents/jobs, 算 Shen 竞争用)

出: data/processed/beijing_edges.npz (gitignored)

候选集 = t_matrix 的 (o_idx,d_idx) 对 (有出行时间才能算效用)。
观测流量按 (o_idx,d_idx) 对齐到候选集; 候选集上没出现的观测对 -> 丢失 (报覆盖率)。

伦敦对应: demo_cache F_ij_t (这里稀疏 4 时段) / t_ij_t / log_d; aux log_D_z
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

PAPERC = Path(r"D:/UoM/PaperC_local")
PROC = Path(__file__).resolve().parents[1] / "processed"
GRID = PROC / "beijing_grid.npz"
OUT = PROC / "beijing_edges.npz"

GAMMA_DECAY = 0.0947  # /min, DCM 标定 (dcm_params.parquet), Shen 竞争 logsum 衰减
PERIODS = ["f_ampeak", "f_pmpeak", "f_midday", "f_night"]

def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)

def main():
    # ---- cell_id -> idx 映射 ----
    ci = pq.read_table(PAPERC / "cell_index.parquet", columns=["idx", "cell_id"]).to_pandas()
    N = len(ci)
    id2idx = dict(zip(ci.cell_id.values, ci.idx.values.astype(np.int64)))

    grid = np.load(GRID, allow_pickle=True)
    residents = grid["residents"].astype(np.float64)
    jobs = grid["jobs"].astype(np.float64)

    # ---- t_matrix = 候选集 ----
    print("loading t_matrix ...")
    tm = pq.read_table(PAPERC / "t_matrix.parquet").to_pandas()
    tm = tm.astype({"o_idx": np.int64, "d_idx": np.int64,
                    "hav_km": np.float32, "t_obs_min": np.float32, "t_ff_min": np.float32})
    # 按 (o_idx,d_idx) 排序 -> CSR 分段
    tm = tm.sort_values(["o_idx", "d_idx"], kind="stable").reset_index(drop=True)
    E = len(tm)
    o_idx = tm.o_idx.values
    d_idx = tm.d_idx.values
    hav_km = tm.hav_km.values
    t_obs = tm.t_obs_min.values
    t_ff = tm.t_ff_min.values

    self_loops = int((o_idx == d_idx).sum())

    # CSR seg_ptr over active origins
    origin_ids, seg_first = np.unique(o_idx, return_index=True)
    O = len(origin_ids)
    seg_ptr = np.append(seg_first, E).astype(np.int64)  # (O+1,)
    seg_sizes = np.diff(seg_ptr)

    # ---- od_cell_flow -> 4 时段, 映射到 idx ----
    print("loading od_cell_flow ...")
    od = pq.read_table(PAPERC / "od_cell_flow.parquet",
                       columns=["o_cell", "d_cell"] + PERIODS).to_pandas()
    od_o = od.o_cell.map(id2idx)
    od_d = od.d_cell.map(id2idx)
    n_unmapped = int(od_o.isna().sum() + od_d.isna().sum())
    keep = od_o.notna() & od_d.notna()
    od = od[keep].copy()
    od["o_idx"] = od_o[keep].astype(np.int64).values
    od["d_idx"] = od_d[keep].astype(np.int64).values
    flow_obs_total = od[PERIODS].values.sum()

    # ---- 对齐观测流量到候选集 (left join on (o_idx,d_idx)) ----
    print("aligning flows to choice set ...")
    edges_df = pd.DataFrame({"o_idx": o_idx, "d_idx": d_idx})
    od_small = od[["o_idx", "d_idx"] + PERIODS]
    merged = edges_df.merge(od_small, on=["o_idx", "d_idx"], how="left")
    flow = merged[PERIODS].fillna(0.0).values.astype(np.float32)  # (E,4)

    flow_captured = flow.sum()
    coverage = flow_captured / max(flow_obs_total, 1.0)

    # 自连边里的观测流量 (intra-cell), 看 self-loop boost 是否有目标
    self_mask = (o_idx == d_idx)
    flow_self = flow[self_mask].sum()

    # ---- Shen 竞争 log_D_z (用候选集 edge) ----
    # comp_j = Σ_{(i,j) edge} residents_i * exp(-γ t_obs_ij);  D_j = comp_j / (jobs_j+1)
    print("computing Shen competition log_D_z ...")
    decay = np.exp(-GAMMA_DECAY * t_obs).astype(np.float64)
    contrib = residents[o_idx] * decay
    comp = np.zeros(N, dtype=np.float64)
    np.add.at(comp, d_idx, contrib)
    D_j = comp / (jobs + 1.0)
    log_D_z = zscore(np.log(D_j + 1e-6))

    # log distance (伦敦 log_d), clamp
    log_d = np.log(np.clip(hav_km, 0.1, None)).astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        o_idx=o_idx.astype(np.int32),
        d_idx=d_idx.astype(np.int32),
        seg_ptr=seg_ptr.astype(np.int64),
        origin_ids=origin_ids.astype(np.int32),
        hav_km=hav_km,
        t_obs=t_obs,
        t_ff=t_ff,
        log_d=log_d,
        flow=flow,                       # (E,4) ampeak/pmpeak/midday/night
        log_D_z=log_D_z,                 # (N,) grid-level Shen 竞争
        N=np.int64(N),
        gamma_decay=np.float32(GAMMA_DECAY),
    )

    # ---- 数据自检 ----
    print(f"\n[OK] {OUT}")
    print(f"  edges E={E:,}  origins O={O:,} (cell {N})  self-loops in t_matrix={self_loops:,}")
    print(f"  cand/origin: mean {seg_sizes.mean():.0f} median {np.median(seg_sizes):.0f} "
          f"min {seg_sizes.min()} max {seg_sizes.max()}")
    print(f"  hav_km  range [{hav_km.min():.2f}, {hav_km.max():.2f}]")
    print(f"  t_obs   range [{t_obs.min():.1f}, {t_obs.max():.1f}] min  mean {t_obs.mean():.1f}")
    print(f"  t_ff    range [{t_ff.min():.1f}, {t_ff.max():.1f}] min  mean {t_ff.mean():.1f}")
    print(f"  t_obs/t_ff ratio mean {(t_obs/np.clip(t_ff,0.1,None)).mean():.2f} (拥堵倍数)")
    print(f"  od pairs mapped: dropped {n_unmapped} unmapped cell refs; kept {len(od):,} pairs")
    print(f"  观测流量总 {flow_obs_total:,.0f}; 候选集捕获 {flow_captured:,.0f} "
          f"=> 覆盖率 {coverage*100:.1f}%")
    print(f"  4 时段捕获流量: {flow.sum(0).astype(np.int64).tolist()} (ampeak/pmpeak/midday/night)")
    print(f"  intra-cell 自连边观测流量 {flow_self:,.0f} "
          f"({flow_self/max(flow_captured,1)*100:.1f}% of captured)")
    print(f"  log_D_z mean {log_D_z.mean():.3f} std {log_D_z.std():.3f}")
    if coverage < 0.8:
        print(f"  [!] 覆盖率 < 80%: 大量观测流量落在无出行时间的对上, 需检查 t_matrix 候选集是否够全")

if __name__ == "__main__":
    main()
