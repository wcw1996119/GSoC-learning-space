"""Phase 1c: 汇总成 v3 同构 aux -> beijing_aux.npz

汇总 grid / edges / income / occupation 四个 npz, 产出训练用 aux (gitignored)。
新增: 逐 edge 余弦职业匹配 match (E,)。

aux keys (北京 trainer 读):
  grid 级 (N,): log_M_z, log_D_z, log_W_z(房价代理), income_z, pct_kids_z(0), mean_cars_z(0)
  档/职业:      income_tier_props(N,3), soc_props(N,7), grid_industry_prop(N,19),
                epsilon(7,19), demand_share(N,7)
  edge 级 (E,): match (cosine soc_o · demand_d)

⚠ log_W_z 用房价代理 (北京无目的地工资数据); pct_kids/mean_cars 占位 0 (TODO census2020)。
"""
import numpy as np
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "processed"
OUT = PROC / "beijing_aux.npz"

def main():
    grid = np.load(PROC / "beijing_grid.npz", allow_pickle=True)
    edges = np.load(PROC / "beijing_edges.npz")
    income = np.load(PROC / "beijing_income.npz")
    occ = np.load(PROC / "beijing_occupation.npz")

    N = int(edges["N"])
    o_idx = edges["o_idx"].astype(np.int64)
    d_idx = edges["d_idx"].astype(np.int64)
    E = len(o_idx)

    log_M_z = grid["log_M_z"].astype(np.float32)
    log_D_z = edges["log_D_z"].astype(np.float32)
    income_z = income["income_z"].astype(np.float32)
    log_W_z = income_z.copy()                      # 房价代理目的地工资 (caveat)
    pct_kids_z = np.zeros(N, dtype=np.float32)     # TODO census2020
    mean_cars_z = np.zeros(N, dtype=np.float32)    # TODO census2020
    income_tier_props = income["income_tier_props"].astype(np.float32)

    soc_props = occ["soc_props"].astype(np.float32)
    grid_industry_prop = occ["grid_industry_prop"].astype(np.float32)
    epsilon = occ["epsilon"].astype(np.float32)
    demand_share = occ["demand_share"].astype(np.float32)

    # ---- 逐 edge 余弦职业匹配 (分块) ----
    soc_n = soc_props / np.linalg.norm(soc_props, axis=1, keepdims=True).clip(min=1e-9)
    dem_n = demand_share / np.linalg.norm(demand_share, axis=1, keepdims=True).clip(min=1e-9)
    match = np.empty(E, dtype=np.float32)
    CH = 5_000_000
    for s in range(0, E, CH):
        e = min(s + CH, E)
        match[s:e] = np.einsum("ek,ek->e", soc_n[o_idx[s:e]], dem_n[d_idx[s:e]]).astype(np.float32)

    np.savez_compressed(
        OUT,
        N=np.int64(N),
        log_M_z=log_M_z, log_D_z=log_D_z, log_W_z=log_W_z, income_z=income_z,
        pct_kids_z=pct_kids_z, mean_cars_z=mean_cars_z,
        income_tier_props=income_tier_props,
        soc_props=soc_props, grid_industry_prop=grid_industry_prop,
        epsilon=epsilon, demand_share=demand_share,
        match=match,
    )

    print(f"[OK] {OUT}  N={N} E={E:,}")
    for nm, a in [("log_M_z", log_M_z), ("log_D_z", log_D_z), ("log_W_z", log_W_z),
                  ("income_z", income_z)]:
        print(f"  {nm:10s} mean {a.mean():.3f} std {a.std():.3f}")
    print(f"  income_tier_props marginal {income_tier_props.mean(0).round(3).tolist()}")
    print(f"  soc_props/demand_share shape {soc_props.shape}/{demand_share.shape}, epsilon {epsilon.shape}")
    print(f"  match (edge cosine) range [{match.min():.3f}, {match.max():.3f}] "
          f"mean {match.mean():.3f} std {match.std():.3f}")
    print(f"  [!] log_W_z=房价代理; pct_kids/mean_cars=0 占位 (TODO census2020)")

if __name__ == "__main__":
    main()
