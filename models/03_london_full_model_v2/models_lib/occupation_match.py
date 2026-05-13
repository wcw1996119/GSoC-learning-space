"""OccupationMatch — agent-level fit between SOC occupation and grid industry mix.

OccMatch_ij(soc) = cos(a_soc, x_j_industry_mix)

where:
  a_soc      = (8,) affinity vector for each SOC class (hand-coded from UK SIC×SOC matrices)
  x_j_industry_mix = (8,) normalized employment-by-sector vector for grid j

For training (no per-agent ground truth), use aggregate:
  OccMatch_avg_ij = sum_soc P(soc|i) × cos(a_soc, x_j)
"""
import numpy as np

SECTORS = [
    "sec1_primary", "sec2_manufacturing", "sec3_construction",
    "sec4_retail", "sec5_fnb", "sec6_info_finance",
    "sec7_public", "sec8_other",
]

# Hand-coded SOC × sector affinity matrix (9 × 8).
# SOC (UK 2010 classification, SOC9):
#   1. Managers, directors, senior officials
#   2. Professional occupations
#   3. Associate professional & technical
#   4. Administrative & secretarial
#   5. Skilled trades
#   6. Caring, leisure & other service
#   7. Sales & customer service
#   8. Process, plant & machine operatives
#   9. Elementary occupations
# Rows = SOC 1-9; columns = SECTORS (primary, mfg, constr, retail, fnb, info_fin, public, other)
# Values 0 (no fit) — 1 (perfect fit). Normalized to unit norm per row.

SOC_INDUSTRY_AFFINITY = np.array([
    # primary mfg constr retail fnb  info_fin public other
    [0.30, 0.50, 0.50, 0.40, 0.30, 0.90, 0.70, 0.80],   # SOC1 managers
    [0.20, 0.30, 0.20, 0.20, 0.10, 0.90, 0.80, 0.80],   # SOC2 professionals (heavy info_fin, public, other-prof)
    [0.20, 0.40, 0.40, 0.30, 0.20, 0.70, 0.60, 0.70],   # SOC3 assoc prof
    [0.10, 0.30, 0.20, 0.40, 0.30, 0.70, 0.60, 0.50],   # SOC4 admin/secr
    [0.50, 0.90, 0.90, 0.30, 0.20, 0.30, 0.20, 0.40],   # SOC5 skilled trades (mfg+construction)
    [0.20, 0.20, 0.10, 0.30, 0.50, 0.20, 0.80, 0.40],   # SOC6 caring/leisure (public+fnb)
    [0.10, 0.10, 0.10, 0.90, 0.60, 0.30, 0.20, 0.30],   # SOC7 sales (heavy retail)
    [0.40, 0.90, 0.70, 0.40, 0.20, 0.20, 0.20, 0.40],   # SOC8 plant operatives (mfg+constr)
    [0.40, 0.50, 0.50, 0.50, 0.70, 0.20, 0.40, 0.40],   # SOC9 elementary
], dtype=np.float64)

# Normalize each row to unit norm (so cosine = dot product)
SOC_INDUSTRY_AFFINITY /= np.linalg.norm(SOC_INDUSTRY_AFFINITY, axis=1, keepdims=True) + 1e-12


def grid_industry_vector(grid_features_df) -> np.ndarray:
    """
    From grid_static_features.csv → (N, 8) normalized industry vector per grid.
    """
    raw = grid_features_df[SECTORS].values.astype(np.float64)
    # Normalize to proportion
    row_sum = raw.sum(axis=1, keepdims=True) + 1e-12
    proportion = raw / row_sum
    # Unit-norm normalize (so dot product = cosine)
    norms = np.linalg.norm(proportion, axis=1, keepdims=True) + 1e-12
    return proportion / norms  # (N, 8)


def occupation_match_per_soc(grid_industry: np.ndarray) -> np.ndarray:
    """
    Compute OccMatch for every (SOC, dest_grid) pair.
    Returns: (9, N) array — rows = SOC 1..9, cols = destination grid.

    OccMatch_(soc, j) = cos(a_soc, x_j) ∈ [0, 1]
    """
    return SOC_INDUSTRY_AFFINITY @ grid_industry.T  # (9, N)


def aggregate_occupation_match(grid_industry: np.ndarray, soc_props: np.ndarray) -> np.ndarray:
    """
    Aggregate OccMatch at origin level using SOC distribution per origin grid.
    soc_props: (N, 9) — proportion of each SOC at origin grid
    grid_industry: (N, 8) — industry vector at destination grid (normalized)

    Returns: (N_origin, N_dest) — aggregate OccMatch_avg_(i,j)
    """
    # Per-SOC OccMatch matrix (9, N_dest)
    om_per_soc = SOC_INDUSTRY_AFFINITY @ grid_industry.T  # (9, N_dest)
    # Weight by SOC distribution at each origin
    # soc_props[i, k] × om_per_soc[k, j] summed over k
    return soc_props @ om_per_soc  # (N_origin, N_dest)
