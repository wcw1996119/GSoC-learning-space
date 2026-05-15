"""Build v3 aux file with Cervero match_prob + Shen log_D_j.

Inputs:
  ../03_london_full_model_v2/data/processed/paperA_v23_aux.npz   (soc_props, grid_industry, etc.)
  ../03_london_full_model_v2/data/processed/demo_cache.npz       (F_ij_t, coords_bng)
  data/processed/empirical_soc_sic_bridge.npz                    (ε matrix)

Output:
  data/processed/paperA_v3_aux_cervero.npz with keys:
    match_prob          (1725, 1725)  — Cervero match probability
    log_D_j             (1725,)       — Shen competition log
    n_workers_per_origin (1725,)      — workforce proxy (F outflow daily sum)
    dist_km             (1725, 1725)  — Euclidean km distance matrix
    grid_industry_prop  (1725, 8)     — re-proportioned (row sum=1) industry
    # passthrough from v2:
    soc_props           (1725, 9)
    income_tier_props   (1725, 3)
    income_score_per_origin (1725,)
    wage_score_per_dest (1725,)
    imd_income_score_per_grid (1725,)
    epsilon             (9, 8)        — embedded bridge for reproducibility
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"

sys.path.insert(0, str(V3_ROOT))
from models_lib.occupation_match import to_proportion, cervero_match_prob
from models_lib.competition import shen_effective_demand


def _read_csv_column(csv_path: Path, col: str) -> np.ndarray:
    """Read named column from grid_static_features.csv (no pandas dep)."""
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return np.array([float(r[col]) for r in rows], dtype=np.float64)


def main():
    v2_aux = np.load(V2_ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    v2_cache = np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz")
    bridge = np.load(V3_ROOT / "data" / "processed" / "empirical_soc_sic_bridge.npz")

    soc_props = v2_aux["soc_props"]                  # (1725, 9), row sum = 1
    grid_industry_raw = v2_aux["grid_industry"]      # (1725, 8), v2 unit-norm
    epsilon = bridge["epsilon"]                      # (9, 8), col sum = 1

    N = soc_props.shape[0]
    assert grid_industry_raw.shape == (N, 8)
    assert epsilon.shape == (9, 8)

    # M_j and log(M_j) from total_employment (avoid pandas dep)
    # Try v3 path first (bundled), fall back to v2 path (local dev).
    v3_csv = V3_ROOT / "data" / "processed" / "grid_static_features.csv"
    v2_csv = V2_ROOT / "data" / "processed" / "grid_static_features.csv"
    csv_path = v3_csv if v3_csv.exists() else v2_csv
    M_j = _read_csv_column(csv_path, "total_employment")
    assert len(M_j) == N, f"grid_static_features rows {len(M_j)} != N {N}"
    log_M_j = np.log(M_j + 1.0)
    print(f"M_j (total employment per grid):")
    print(f"  range [{M_j.min():.0f}, {M_j.max():.0f}], mean {M_j.mean():.0f}, total {M_j.sum():.0f}")
    print(f"  log_M_j range [{log_M_j.min():.2f}, {log_M_j.max():.2f}]")

    # Convert v2's unit-norm grid_industry → row-sum=1 proportion
    grid_industry_prop = to_proportion(grid_industry_raw)
    print(f"grid_industry → proportion check: row sums in [{grid_industry_prop.sum(axis=1).min():.4f}, "
          f"{grid_industry_prop.sum(axis=1).max():.4f}]")

    # --- Cervero match_prob ---
    match_prob = cervero_match_prob(soc_props, epsilon, grid_industry_raw)
    print(f"\nCervero match_prob:")
    print(f"  shape={match_prob.shape}, dtype={match_prob.dtype}")
    print(f"  range: [{match_prob.min():.4f}, {match_prob.max():.4f}]")
    print(f"  mean = {match_prob.mean():.4f}, std = {match_prob.std():.4f}")
    eye = np.eye(N, dtype=bool)
    off = match_prob[~eye]
    pcts = [1, 5, 25, 50, 75, 95, 99]
    qs = np.percentile(off, pcts)
    print(f"  off-diag percentiles:")
    for p, q in zip(pcts, qs):
        print(f"    p{p:2d} = {q:.4f}")

    # --- Distance matrix ---
    coords = v2_cache["coords_bng"].astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_km = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0
    print(f"\ndist_km: range [{dist_km.min():.2f}, {dist_km.max():.2f}] km")

    # --- Shen competition D_j ---
    # n_workers_i: workforce per origin grid. Use Census 'population' from
    # grid_static_features.csv (Beijing-friendly via 七普 居住人口 same path).
    n_workers = _read_csv_column(csv_path, "population")
    print(f"\nn_workers_per_origin (Census population at origin grid):")
    print(f"  range [{n_workers.min():.0f}, {n_workers.max():.0f}], "
          f"mean {n_workers.mean():.0f}, total {n_workers.sum():.0f}")

    D_j = shen_effective_demand(n_workers, dist_km, beta_decay=-0.35, kernel="power")
    log_D_j = np.log(D_j + 1.0)
    print(f"\nShen D_j (β=-0.35 power decay):")
    print(f"  range [{D_j.min():.1f}, {D_j.max():.1f}]")
    print(f"  log_D_j range [{log_D_j.min():.2f}, {log_D_j.max():.2f}]")

    # --- Compare new match_prob vs old occ_match ---
    old_occ = v2_aux["occ_match"]
    print(f"\n=== match_prob vs old occ_match comparison ===")
    print(f"  old occ_match (cosine):  range [{old_occ.min():.4f}, {old_occ.max():.4f}], "
          f"mean {old_occ.mean():.4f}")
    print(f"  new match_prob (Cervero):range [{match_prob.min():.4f}, {match_prob.max():.4f}], "
          f"mean {match_prob.mean():.4f}")
    # Pearson correlation between them (off-diag)
    corr = np.corrcoef(old_occ[~eye], match_prob[~eye])[0, 1]
    print(f"  correlation (off-diag):  {corr:+.4f}")
    print(f"  (low corr = new metric captures very different signal)")

    # --- z-score standardization for fair coefficient comparison ---
    # Each feature: (raw - mean) / std → standardized version (mean 0, std 1)
    # Coefficients learned on these are 'per-stddev effect' — directly comparable.
    log_W_j = v2_aux["wage_score_per_dest"].astype(np.float64)
    income_score_orig = v2_aux["income_score_per_origin"].astype(np.float64)

    def _zscore(x: np.ndarray, name: str):
        mu = float(x.mean()); sigma = float(x.std())
        z = (x - mu) / max(sigma, 1e-9)
        print(f"  {name}: raw mean={mu:.4f} std={sigma:.4f} → z-scored")
        return z, mu, sigma

    print("\n=== z-score standardization (for fair coefficient comparison) ===")
    log_W_z, log_W_mean, log_W_std = _zscore(log_W_j, "log_W_j")
    log_M_z, log_M_mean, log_M_std = _zscore(log_M_j, "log_M_j")
    log_D_z, log_D_mean, log_D_std = _zscore(log_D_j, "log_D_j")
    match_z_off = match_prob[~np.eye(N, dtype=bool)]  # off-diag for mean/std (drop self)
    match_mean = float(match_z_off.mean()); match_std = float(match_z_off.std())
    match_z = (match_prob - match_mean) / max(match_std, 1e-9)
    print(f"  match_prob: raw mean={match_mean:.4f} std={match_std:.4f} → z-scored")
    income_z, income_mean, income_std = _zscore(income_score_orig, "income_score_per_origin")

    out = V3_ROOT / "data" / "processed" / "paperA_v3_aux_cervero.npz"
    np.savez(
        out,
        # Standardized features (USE THESE in trainer):
        log_W_z=log_W_z.astype(np.float64),
        log_M_z=log_M_z.astype(np.float64),
        log_D_z=log_D_z.astype(np.float64),
        match_z=match_z.astype(np.float64),
        income_z=income_z.astype(np.float64),
        # Scaling factors (for paper reporting + Beijing migration):
        log_W_mean=np.float64(log_W_mean), log_W_std=np.float64(log_W_std),
        log_M_mean=np.float64(log_M_mean), log_M_std=np.float64(log_M_std),
        log_D_mean=np.float64(log_D_mean), log_D_std=np.float64(log_D_std),
        match_mean=np.float64(match_mean), match_std_=np.float64(match_std),
        income_mean=np.float64(income_mean), income_std=np.float64(income_std),
        # Raw features (for diagnostics):
        match_prob=match_prob.astype(np.float64),
        log_D_j=log_D_j.astype(np.float64),
        D_j=D_j.astype(np.float64),
        log_M_j=log_M_j.astype(np.float64),
        M_j=M_j.astype(np.float64),
        log_W_j=log_W_j,
        n_workers_per_origin=n_workers.astype(np.float64),
        dist_km=dist_km.astype(np.float64),
        grid_industry_prop=grid_industry_prop.astype(np.float64),
        epsilon=epsilon.astype(np.float64),
        # Passthrough useful fields
        soc_props=soc_props.astype(np.float64),
        income_tier_props=v2_aux["income_tier_props"].astype(np.float64),
        income_score_per_origin=income_score_orig,
        wage_score_per_dest=log_W_j,
        imd_income_score_per_grid=v2_aux["imd_income_score_per_grid"].astype(np.float64),
    )
    print(f"\n[saved] {out}")
    print(f"        size: {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
