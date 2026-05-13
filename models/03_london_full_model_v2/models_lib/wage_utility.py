"""WageUtility — v2.3 utility with destination wage attractiveness + continuous β(income).

V_ij = α_V·V_j(t)
     + α_occ·OccMatch(soc, j)
     + α_wage·log(wage_j / median_wage)            ← NEW: agents are attracted to high-wage destinations
     − β(income_real)·t_ij(t)                       ← β is a continuous function of agent's real income
     − γ·log d_ij + bias

Key v2.3 changes from v2.2:
1. wage_j enters the utility positively — direct economic mechanism, separate from β
2. β is now continuous in income_real (not categorical 3-tier)
3. Training-time β is the population-weighted mean of agents' personal β at each origin

Trainable params: α_V, α_occ, α_wage, γ, bias (softplus on the first 4)
Non-trainable: β table (literature anchored, agent-level lookup at inference)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v):
    return math.log(math.exp(v) - 1.0)


class WageUtility(nn.Module):
    def __init__(
        self,
        alpha_V_init: float = 0.5,
        alpha_occ_init: float = 0.7,
        alpha_wage_init: float = 0.3,
        gamma_init: float = 0.5,
    ):
        super().__init__()
        self.alpha_V_raw   = nn.Parameter(torch.tensor(_inv_softplus(alpha_V_init)))
        self.alpha_occ_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_occ_init)))
        self.alpha_wage_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_wage_init)))
        self.gamma_raw     = nn.Parameter(torch.tensor(_inv_softplus(gamma_init)))
        self.bias          = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha_V(self):     return F.softplus(self.alpha_V_raw)
    @property
    def alpha_occ(self):   return F.softplus(self.alpha_occ_raw)
    @property
    def alpha_wage(self):  return F.softplus(self.alpha_wage_raw)
    @property
    def gamma(self):       return F.softplus(self.gamma_raw)

    def utility_aggregate(self, V_j_t, occ_match_avg, log_wage_norm, t_ij_t,
                            log_d_ij, beta_origin):
        """For training: utility with population-mean β per origin."""
        return (
            self.alpha_V * V_j_t
            + self.alpha_occ * occ_match_avg
            + self.alpha_wage * log_wage_norm
            - beta_origin * t_ij_t
            - self.gamma * log_d_ij
            + self.bias
        )

    def utility_personal(self, V_j_t, occ_match_personal, log_wage_norm,
                          t_ij_t, log_d_ij, beta_personal):
        """For inference: utility with each agent's own β."""
        return (
            self.alpha_V * V_j_t
            + self.alpha_occ * occ_match_personal
            + self.alpha_wage * log_wage_norm
            - beta_personal * t_ij_t
            - self.gamma * log_d_ij
            + self.bias
        )


def compute_beta_per_origin_v23(agents_df, n_grids: int) -> torch.Tensor:
    """Population-weighted mean β_personal per origin grid (using real income β)."""
    import numpy as np
    beta_origin = np.zeros(n_grids, dtype=np.float32)
    counts = np.zeros(n_grids, dtype=np.int32)
    for _, row in agents_df.iterrows():
        i = int(row["home_grid_idx"])
        beta_origin[i] += float(row["beta_personal"])
        counts[i] += 1
    has = counts > 0
    beta_origin[has] /= counts[has]
    beta_origin[~has] = 0.07  # default for empty grids
    return torch.tensor(beta_origin, dtype=torch.float32)
