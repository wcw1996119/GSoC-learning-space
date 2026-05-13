"""Per-agent β heterogeneity using literature-derived VOT spread.

Core idea: aggregate inverse model recovers β_mean(income, dest_wage). Agent
layer adds individual idiosyncratic taste shock ε_n drawn from a Normal
calibrated to WebTAG / Wardman VOT spread:

    β_n(i, j) = β_base · (1 + φ·z_o(i) + ψ·z_w(j) + ε_n)
    ε_n ~ N(0, σ²(income_tier_n))

σ values derive from coefficient-of-variation (CoV) on VOT in published
revealed-preference work (Wardman 2014 meta-analysis: CoV ≈ 0.4 across
commuting studies). Calibrating CoV per income tier from WebTAG TAG A1.3:

    Low income   : CoV ≈ 0.40 (VOT £8 ± £3)
    Mid income   : CoV ≈ 0.40 (VOT £13 ± £5)
    High income  : CoV ≈ 0.40 (VOT £22 ± £8)

Choosing CoV uniformly across tiers keeps the heterogeneity story simple:
the **mean** β shifts with income (via interaction φ, ψ), the spread is
proportional. Tunable via ``cov_by_tier`` argument.

This avoids needing individual NTS/LTDS data — published VOT distributions
are public via DfT WebTAG and Wardman 2014 review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# Default coefficient of variation per income tier (1=low, 2=mid, 3=high).
# From Wardman 2014 meta-analysis + DfT WebTAG TAG A1.3 cross-tabulation.
DEFAULT_COV_BY_TIER: Dict[int, float] = {1: 0.40, 2: 0.40, 3: 0.40}

# Reference WebTAG VOT central values (£/h, 2024 prices) — for documentation
# and external calibration anchors. Not used in computation.
WEBTAG_VOT_BY_TIER_GBP_PER_HOUR: Dict[int, float] = {1: 8.0, 2: 13.0, 3: 22.0}


@dataclass
class AgentBetaConfig:
    """Container for per-agent β assignment hyperparameters."""
    cov_by_tier: Dict[int, float]              # σ_VOT / VOT per income tier
    seed: int = 42                             # RNG seed for ε draws
    clip_min: float = -2.0                     # ε clamped to [-clip, +clip]
    clip_max: float = +2.0                     # to prevent extreme outliers


def assign_per_agent_epsilon(
    income_tier_per_agent: np.ndarray,
    config: AgentBetaConfig,
) -> np.ndarray:
    """Draw ε_n ~ N(0, σ²(tier_n)) for each agent."""
    rng = np.random.default_rng(config.seed)
    n = len(income_tier_per_agent)
    eps = np.zeros(n, dtype=np.float64)
    for tier, cov in config.cov_by_tier.items():
        mask = income_tier_per_agent == tier
        if mask.sum() == 0:
            continue
        # ε ~ N(0, CoV²) — interpreted as multiplicative deviation around 1
        eps[mask] = rng.normal(loc=0.0, scale=cov, size=mask.sum())
    eps = np.clip(eps, config.clip_min, config.clip_max)
    return eps


def per_agent_beta(
    base_beta_t: float,
    phi: float,
    psi: float,
    z_o_per_agent: np.ndarray,             # origin IMD z-score per agent (N_agents,)
    z_w_per_dest: np.ndarray,              # dest wage z-score per dest (N_grids,)
    epsilon_per_agent: np.ndarray,         # (N_agents,)
) -> np.ndarray:
    """Compute (N_agents, N_grids) personalised β_eff matrix.

    β_n(j) = β_base · (1 + φ·z_o(n) + ψ·z_w(j) + ε_n)
    """
    N = len(z_o_per_agent)
    M = len(z_w_per_dest)
    interaction = (1.0
                   + phi * z_o_per_agent[:, None]      # (N, 1)
                   + psi * z_w_per_dest[None, :]       # (1, M)
                   + epsilon_per_agent[:, None])       # (N, 1)
    return base_beta_t * interaction                    # (N, M)


def simulate_agent_choice_softmax(
    V_j: np.ndarray,                            # (N_grids,) destination attractiveness
    t_io: np.ndarray,                           # (N_agents, N_grids) origin-to-dest time
    log_d_io: np.ndarray,                       # (N_agents, N_grids) origin-to-dest log distance
    occ_match_io: np.ndarray,                   # (N_agents, N_grids) occ match score
    beta_eff_per_agent: np.ndarray,             # (N_agents, N_grids) β_n(j)
    delta: float,                               # OccMatch coefficient
    gamma: float,                               # distance coefficient (negative)
    alpha_V: float = 1.0,                       # GNN V coefficient (fixed)
    home_grid_idx_per_agent: np.ndarray = None, # (N_agents,) for self-loop masking
    self_loop_logit: float = -1e9,
) -> np.ndarray:
    """Per-agent softmax over destinations. Returns (N_agents, N_grids) P_n(j).

    Utility (per agent n, destination j):
        U(n, j) = α·V_j + β_n(j)·t(n, j) + γ·log_d(n, j) + δ·OccMatch(n, j)
    """
    N, M = beta_eff_per_agent.shape
    U = (alpha_V * V_j[None, :]
         + beta_eff_per_agent * t_io
         + gamma * log_d_io
         + delta * occ_match_io)                 # (N, M)
    if home_grid_idx_per_agent is not None:
        # Mask self-loop (no commuting to home grid)
        rows = np.arange(N)
        U[rows, home_grid_idx_per_agent] = self_loop_logit
    # Numerically stable softmax
    U = U - U.max(axis=1, keepdims=True)
    expU = np.exp(U)
    P = expU / expU.sum(axis=1, keepdims=True)
    return P


def aggregate_per_agent_to_OD(
    P_per_agent: np.ndarray,                    # (N_agents, N_grids) P_n(j)
    home_grid_idx_per_agent: np.ndarray,        # (N_agents,)
    n_grids: int,
) -> np.ndarray:
    """Aggregate (N_agents, N_grids) per-agent destination probs to a (N_grids, N_grids)
    OD matrix. Each agent contributes 1.0 of mass to its origin row, distributed
    across destinations per their softmax.
    """
    F = np.zeros((n_grids, n_grids), dtype=np.float64)
    for i in range(n_grids):
        agent_mask = home_grid_idx_per_agent == i
        if agent_mask.sum() == 0:
            continue
        F[i] = P_per_agent[agent_mask].sum(axis=0)
    return F


__all__ = [
    "AgentBetaConfig",
    "DEFAULT_COV_BY_TIER",
    "WEBTAG_VOT_BY_TIER_GBP_PER_HOUR",
    "assign_per_agent_epsilon",
    "per_agent_beta",
    "simulate_agent_choice_softmax",
    "aggregate_per_agent_to_OD",
]
