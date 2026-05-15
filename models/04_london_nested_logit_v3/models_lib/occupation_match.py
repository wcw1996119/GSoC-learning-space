"""Cervero 1999 occupational match using empirical SOC×SIC bridge (v3).

Replaces v2's cosine-on-unit-norm OccMatch with Cervero proportion product:

    match_prob[i, j] = Σ_k soc_props[i, k] · (Σ_c ε[k, c] · grid_industry_prop[j, c])
                     = soc_props[i] @ ε @ grid_industry_prop[j].T

where:
    soc_props          (N, 9) — origin SOC proportions, rows sum to 1
    ε                  (9, 8) — empirical P(SOC=k | SIC=c), columns sum to 1
                                 from ONS adhoc 2978 (APS 2022-2024)
    grid_industry_prop (N, 8) — destination SIC proportions, rows sum to 1
                                 (raw proportion, NOT unit-norm — that was v2's bug)
    match_prob         (N, N) — bounded in [0, max(ε)], ~[0, 1] in practice

Anchor: Cervero R, Rood T, Appleyard B (1999) "Tracking Accessibility: Employment
and Housing Opportunities in the San Francisco Bay Area" E&P A 31:1259-1278.

Beijing portability: same formula, replace ε with 中国劳动统计年鉴 industry×SOC
cross-tab; replace grid_industry_prop with POI or 经普 industry mix at workplace.
"""
from __future__ import annotations

import numpy as np

SECTORS = [
    "sec1_primary", "sec2_manufacturing", "sec3_construction",
    "sec4_retail", "sec5_fnb", "sec6_info_finance",
    "sec7_public", "sec8_other",
]


def to_proportion(grid_industry: np.ndarray) -> np.ndarray:
    """Convert (N, 8) industry vector to row-sum=1 proportion.

    Accepts either raw counts, L1-normalised proportions (already sum=1),
    or L2-normalised unit vectors (v2's storage format). Returns sum=1.
    """
    row_sum = grid_industry.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    prop = grid_industry / row_sum
    return prop


def cervero_match_prob(soc_props: np.ndarray,
                       epsilon: np.ndarray,
                       grid_industry: np.ndarray) -> np.ndarray:
    """Compute Cervero match probability for all (i, j) pairs.

    Args:
      soc_props:     (N, 9) origin SOC proportions, row sum=1
      epsilon:       (9, 8) empirical P(SOC | SIC), column sum=1
      grid_industry: (N, 8) destination industry, any normalisation
                            (we'll convert to row-sum=1 proportion)

    Returns:
      match_prob:    (N, N) entries in [0, ~1]
    """
    gi_prop = to_proportion(grid_industry)
    # 3-way matmul
    return soc_props @ epsilon @ gi_prop.T


def cervero_match_prob_pairwise(soc_props_i: np.ndarray,
                                 epsilon: np.ndarray,
                                 grid_industry_j: np.ndarray) -> float:
    """Single (i, j) match probability — for diagnostics / sanity tests."""
    prop_j = grid_industry_j / max(grid_industry_j.sum(), 1.0)
    return float(soc_props_i @ epsilon @ prop_j)
