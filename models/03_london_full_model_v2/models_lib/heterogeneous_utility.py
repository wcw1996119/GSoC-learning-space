"""HeterogeneousUtility — v2.2 utility with agent-level heterogeneity.

V_ij = α_V·V_j(t) + α_occ·OccMatch_(soc, j) − β_(income, mode)·t_ij(t) − γ·log d_ij + b

Key v2.2 changes from v2.1:
1. β is HETEROGENEOUS — fixed lookup table β_income[tier] × β_mode[mode], NOT learned.
   Avoids identifiability with α_V, anchors to literature (WebTAG VOT).

2. α_V, α_occ, γ enforced > 0 via softplus reparameterization. Avoids
   the "learned alpha_occ ≈ -0.05" failure mode where data identifiability is weak.

3. Per-agent OccMatch — uses agent's specific SOC, not aggregate distribution.

Theoretical priors (β_income × β_mode multiplicative, per-minute disutility):
  Low income (SOC6-9, tier 1):    β_income = 0.12  (high cost sensitivity)
  Mid income (SOC3-5, tier 2):    β_income = 0.07  (WebTAG default VOT)
  High income (SOC1-2, tier 3):   β_income = 0.04  (low cost sensitivity)

  Car:                             β_mode   = 1.0
  PT (peak crowding):              β_mode   = 1.2
  Active (walk/bike):              β_mode   = 1.5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Literature-anchored fixed β table
INCOME_TIERS = ["low", "mid", "high"]   # 1-indexed in agents.csv → 0-indexed here
BETA_INCOME = torch.tensor([0.12, 0.07, 0.04], dtype=torch.float32)

MODES = ["car", "pt", "active"]
BETA_MODE = torch.tensor([1.0, 1.2, 1.5], dtype=torch.float32)
MODE_TO_IDX = {"car": 0, "pt": 1, "active": 2}


def beta_for(income_tier: int, mode: str) -> float:
    """Look up agent's β. income_tier in {1, 2, 3}."""
    return float(BETA_INCOME[income_tier - 1] * BETA_MODE[MODE_TO_IDX[mode]])


class HeterogeneousUtility(nn.Module):
    """v2.2 utility with heterogeneous β (fixed table) and softplus-positive priors.

    Learnable parameters: alpha_V, alpha_occ, gamma, bias  (4 total)
    Non-learnable: beta (looked up per agent from income × mode table)

    Forward signatures:
      utility_aggregate(V_j, occ, t_ij, log_d, beta_origin) — for aggregate training
      utility_personal(V_j, occ, t_ij, log_d, income_tier, mode_idx) — for per-agent inference
    """

    def __init__(self, alpha_V_init: float = 0.5,
                 alpha_occ_init: float = 0.7,    # strong positive prior per supervisor
                 gamma_init: float = 0.5):
        super().__init__()
        # softplus(x) = log(1 + e^x). To get target value v, init raw to log(e^v - 1)
        def _inv_softplus(v):
            import math
            return math.log(math.exp(v) - 1.0)

        self.alpha_V_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_V_init)))
        self.alpha_occ_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_occ_init)))
        self.gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(gamma_init)))
        self.bias = nn.Parameter(torch.tensor(0.0))

        # Register fixed β table as buffer (moves with .to(device) but not learned)
        self.register_buffer("beta_income", BETA_INCOME.clone())
        self.register_buffer("beta_mode", BETA_MODE.clone())

    @property
    def alpha_V(self):
        return F.softplus(self.alpha_V_raw)

    @property
    def alpha_occ(self):
        return F.softplus(self.alpha_occ_raw)

    @property
    def gamma(self):
        return F.softplus(self.gamma_raw)

    def beta(self, income_tier: int, mode_idx: int) -> torch.Tensor:
        """β for one agent."""
        return self.beta_income[income_tier - 1] * self.beta_mode[mode_idx]

    def utility_aggregate(self, V_j_t, occ_match_avg, t_ij_t, log_d_ij, beta_origin):
        """Aggregate utility for training.
        beta_origin: scalar per origin grid — population-weighted β.
        """
        return (
            self.alpha_V * V_j_t
            + self.alpha_occ * occ_match_avg
            - beta_origin * t_ij_t
            - self.gamma * log_d_ij
            + self.bias
        )

    def utility_personal(self, V_j_t, occ_match_personal, t_ij_t, log_d_ij,
                          income_tier: int, mode_idx: int):
        """Per-agent utility for inference."""
        beta = self.beta(income_tier, mode_idx)
        return (
            self.alpha_V * V_j_t
            + self.alpha_occ * occ_match_personal
            - beta * t_ij_t
            - self.gamma * log_d_ij
            + self.bias
        )


def compute_beta_per_origin(agents_df, n_grids: int) -> torch.Tensor:
    """Compute population-weighted β per origin grid (for aggregate training).

    β_origin(i) = Σ_(tier, mode) P(tier, mode | i) × β_income[tier] × β_mode[mode]
    """
    import numpy as np
    beta_origin = np.zeros(n_grids, dtype=np.float32)
    counts = np.zeros(n_grids, dtype=np.int32)

    for _, row in agents_df.iterrows():
        i = int(row["home_grid_idx"])
        tier = int(row["income_tier"])
        mode = str(row["mode_initial"])
        b = beta_for(tier, mode)
        beta_origin[i] += b
        counts[i] += 1

    # Average per grid (where agents exist)
    has_agents = counts > 0
    beta_origin[has_agents] /= counts[has_agents]
    # Default for grids with no agents: use mid×car as neutral
    beta_origin[~has_agents] = beta_for(2, "car")

    return torch.tensor(beta_origin, dtype=torch.float32)
