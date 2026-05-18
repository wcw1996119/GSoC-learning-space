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


def cervero_match_cosine(soc_props: np.ndarray,
                         epsilon: np.ndarray,
                         grid_industry: np.ndarray) -> np.ndarray:
    """Cosine similarity between origin SOC profile and destination's expected SOC demand.

    Replaces cervero_match_prob (joint consistency probability, max bounded by
    max(ε) ≈ 0.5) with a proper [0, 1] similarity score.

    Formula:
        expected_soc[j] = ε @ grid_industry_prop[j]    (workplace's expected SOC demand)
        match[i, j]     = cos(soc_props[i], expected_soc[j])

    Anchored to:
      - Delgado, Porter, Stern 2014 (NBER WP 20375) — cosine on occupational
        linkages is the standard in industry-cluster definitions.
      - Hu & Wang 2020 — matching factor in 2SFCA accessibility (TR Part D).
      - v2's original OccMatch design (cosine on hand-coded SOC×sector affinity)
        — restores v2's [0, 1] property while keeping v3's empirically derived ε.

    Args:
      soc_props:     (N, 9) origin SOC proportions, row sum = 1
      epsilon:       (9, 8) empirical P(SOC=k | SIC=l), column sum = 1
      grid_industry: (N, 8) destination industry mix (any normalisation; we
                            convert to row-sum=1 proportion)

    Returns:
      match: (N, N) cosine similarity in [0, 1]. Higher = better skill alignment.
    """
    gi_prop = to_proportion(grid_industry)
    # Workplace j's expected SOC demand: (N, 9) = grid_industry @ ε^T
    expected_soc = gi_prop @ epsilon.T

    soc_norm = np.linalg.norm(soc_props, axis=1, keepdims=True)
    soc_norm = np.where(soc_norm > 1e-12, soc_norm, 1.0)
    soc_unit = soc_props / soc_norm

    dest_norm = np.linalg.norm(expected_soc, axis=1, keepdims=True)
    dest_norm = np.where(dest_norm > 1e-12, dest_norm, 1.0)
    dest_unit = expected_soc / dest_norm

    return soc_unit @ dest_unit.T   # (N, N) ∈ [0, 1]


def per_soc_demand_share(soc_props: np.ndarray,
                          epsilon: np.ndarray,
                          grid_industry: np.ndarray) -> np.ndarray:
    """Per-SOC destination demand share — individual-level match for ABM agents.

    For each destination j and worker SOC=k, returns the share of jobs at j that
    demand SOC=k workers (given j's industry mix). This is the *individual*
    counterpart to the aggregate place-level match: an ABM agent with SOC=k
    looking at destination j sees demand_share[j, k] as "what fraction of j's
    jobs are for me".

    Formula:
        expected_soc[j, k] = Σ_c industry_prop[j, c] · ε[k, c]
        demand_share[j, k] = expected_soc[j, k] / Σ_{k'} expected_soc[j, k']

    Args:
      soc_props:     (N, 9) origin SOC proportions — unused here but kept in
                              signature for parallelism with cervero_match_*.
                              Used downstream as mixture weight: P(SOC=k | origin=i).
      epsilon:       (9, 8) empirical P(SOC=k | SIC=c), column sum = 1
      grid_industry: (N, 8) destination industry mix; any normalisation accepted

    Returns:
      demand_share: (N, 9) entries in [0, 1]; rows sum to 1.
                            Indexed as demand_share[destination j, worker SOC k].

    Anchored to:
      - Stoll MA, Houston G (2005) "Spatial mismatch and occupational match" —
        Stoll's "effective jobs accessible to skill group k" formulation.
      - Schwanen T et al (2003) "Travel behaviour and the urban form" —
        individual-level occupational sorting in commute patterns.

    Beijing portability: same ε formula, replace grid_industry with POI mix.
    """
    del soc_props  # signature consistency; not used here
    gi_prop = to_proportion(grid_industry)
    expected_soc = gi_prop @ epsilon.T          # (N, 9)
    row_sum = expected_soc.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    return expected_soc / row_sum               # (N, 9), rows sum to 1
