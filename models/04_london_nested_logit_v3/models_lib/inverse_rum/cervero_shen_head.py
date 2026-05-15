"""CerveroShenHead — D→M nested logit with Cervero match + Shen competition.

v3 redesign of dm_nested_logit_head.py:

  Upper nest (destination):
      V_dest(i, j, t) = α · log(W_j)               ← wage attractiveness (NEW)
                      + γ · log(M_j)               ← jobs count, Hansen gravity (NEW form)
                      + ν · log(D_j)               ← Shen competition (NEW)
                      + δ · match_prob[i, j]       ← Cervero match (NEW)
                      + λ_b(j) · IV_mode(i, j, t)  ← inclusive value from lower nest
                      + (blend · V_GNN_jt)         ← optional Wang TB-ResNet

  Lower nest (mode | destination):
      V_mode(i, j, t, m) = ASC_m
                         + β_t_m · t_{ij,m}        ← linear time disutility per mode
                         + θ_inc_m · income_i      ← income×mode interaction (V_lower scope)

  Where:
      λ_b ∈ [ε_min, 1] via λ_b = ε_min + (1-ε_min) · sigmoid
      IV_mode(i,j,t) = log Σ_m exp(V_mode / λ_{b(j)})

Anchored to:
  - Cervero R, Rood T, Appleyard B (1999) — occupational match formula
  - Shen Q (1998) — competition-adjusted accessibility
  - Hansen WG (1959) — gravity model
  - McFadden D (1973) — conditional logit / nested logit

Changes vs v3 DMNestedLogitHead:
  REMOVED:
    γ · log_d_ij         (distance decay redundant with t·β captured in IV)
    β_slope · log_d      (per-mode log_d slope — redundant)
    δ_slope · log_d      (match × distance interaction — behaviorally unjustified)
    θ_per_mode · X_mode-level features (Hansen accessibility / mean_t / n_jobs_30min)

  ADDED:
    α · log(W_j)         (wage attractiveness)
    ν · log(D_j)         (Shen competition)
    δ · match_prob       (Cervero match — replaces old cosine OccMatch)
    θ_inc_m · income_i   (income × mode interaction in V_lower)

Parameter inventory (defaults M=3, B=33):
    raw_alpha_wage              scalar       wage coefficient (NEW, > 0)
    raw_gamma_M                 scalar       jobs coefficient (NEW, > 0)
    raw_nu_D                    scalar       competition coefficient (NEW, < 0)
    delta_match                 scalar       Cervero match coefficient (NEW)
    raw_beta_t_per_mode         (M,)         time disutility per mode
    asc_per_mode                (M,)         ASC per mode
    theta_inc_per_mode          (M,)         income × mode interaction
    raw_lambda_borough          (B,)         borough nest correlation
    raw_gnn_blend               scalar       (optional) Wang blend

Total free params (defaults): 1 + 1 + 1 + 1 + 3 + 3 + 3 + 33 + 1 = 47 RUM params
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    return math.log(math.exp(v) - 1.0)


def _inv_sigmoid(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return math.log(s / (1 - s))


class CerveroShenHead(nn.Module):
    """D→M nested logit head with Cervero match + Shen competition + Hansen attractiveness.

    Parameters
    ----------
    n_modes : int
        Number of modes (default 3 for car/transit/walk).
    n_boroughs : int
        Number of borough λ values to learn (default 33).
    alpha_wage_init : float
        Initial α (log W_j coefficient). Default 0.1, sign-free.
    gamma_M_init : float
        Initial γ (log M_j coefficient). Enforced > 0 via softplus. Default 0.5.
    nu_D_init : float
        Initial ν (log D_j coefficient). Enforced < 0 via -softplus. Default -0.1.
    delta_match_init : float
        Initial δ (match_prob coefficient). Sign-free. Default 1.0 (mild positive).
    beta_t_init : float
        Initial β_t,m (time disutility). Negative, enforced via -softplus. Default -0.07.
    theta_inc_init : float
        Initial θ_inc,m (income × mode interaction). Default 0.
    lambda_init : float
        Initial λ_b. Default 0.95 (close to flat MNL).
    lambda_eps_min : float
        Lower bound on λ_b. Default 0.05.
    use_gnn_blend : bool
        Enable Wang TB-ResNet blend.
    gnn_blend_init : float
        Initial blend. Default 0.5.
    blend_max : float
        Hard cap. Default 1.0.
    """

    def __init__(
        self,
        n_modes: int = 3,
        n_boroughs: int = 33,
        alpha_wage_init: float = 0.1,
        gamma_M_init: float = 0.5,
        nu_D_init: float = -0.1,
        delta_match_init: float = 0.1,
        beta_t_init: float = -0.07,
        beta_t_slope_init: float = -0.01,
        theta_inc_init: float = 0.0,
        lambda_init: float = 0.95,
        lambda_eps_min: float = 0.05,
        use_gnn_blend: bool = False,
        gnn_blend_init: float = 0.5,
        blend_max: float = 1.0,
        gnn_mode: str = "convex",  # "convex" (v3b/c legacy) | "residual" (Wang TB-ResNet)
        gnn_residual_scale_init: float = 0.1,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.n_boroughs = n_boroughs
        self.lambda_eps_min = float(lambda_eps_min)
        self.use_gnn_blend = use_gnn_blend
        self.blend_max = float(blend_max)
        self.gnn_mode = str(gnn_mode)
        assert self.gnn_mode in ("convex", "residual"), f"unknown gnn_mode={gnn_mode}"

        # α (wage) — enforced >= 0 via softplus (Cervero/Hansen/Wang behavioral prior)
        raw_a = _inv_softplus(max(alpha_wage_init, 1e-3))
        self.raw_alpha_wage = nn.Parameter(torch.tensor(raw_a))

        # γ (M_j) — enforced > 0 via softplus, since jobs always positively attract
        raw_g = _inv_softplus(max(gamma_M_init, 1e-3))
        self.raw_gamma_M = nn.Parameter(torch.tensor(raw_g))

        # ν (D_j) — enforced < 0 via -softplus, since competition repels
        raw_n = _inv_softplus(max(-nu_D_init, 1e-3))
        self.raw_nu_D = nn.Parameter(torch.tensor(raw_n))

        # δ (match_prob) — enforced >= 0 via softplus (Cervero behavioral prior)
        raw_d = _inv_softplus(max(delta_match_init, 1e-3))
        self.raw_delta_match = nn.Parameter(torch.tensor(raw_d))

        # β_t,0 per mode (intercept) — enforced < 0 via -softplus
        target_b = -beta_t_init
        raw_b = _inv_softplus(target_b)
        self.raw_beta_t_per_mode = nn.Parameter(torch.full((n_modes,), raw_b))

        # β_t,1 per mode (slope on log_d) — enforced <= 0 via -softplus.
        # Captures distance-dependent VOT: at longer trips, each commute minute hurts MORE.
        # Lit anchor: Wachs et al. 1993, Levinson 2010 distance-dependent value of time.
        target_b1 = max(-beta_t_slope_init, 1e-3)
        raw_b1 = _inv_softplus(target_b1)
        self.raw_beta_t_slope_per_mode = nn.Parameter(torch.full((n_modes,), raw_b1))

        # ASC per mode — free
        self.asc_per_mode = nn.Parameter(torch.zeros(n_modes))

        # θ_inc per mode — free (income × mode interaction)
        self.theta_inc_per_mode = nn.Parameter(torch.full((n_modes,), float(theta_inc_init)))

        # θ_kids per mode — free (household has dep children × mode interaction)
        self.theta_kids_per_mode = nn.Parameter(torch.zeros(n_modes))

        # θ_cars per mode — free (mean cars per household × mode interaction)
        self.theta_cars_per_mode = nn.Parameter(torch.zeros(n_modes))

        # λ_b per borough — bounded
        s = (lambda_init - lambda_eps_min) / (1.0 - lambda_eps_min)
        raw_l = _inv_sigmoid(s)
        self.raw_lambda_borough = nn.Parameter(torch.full((n_boroughs,), raw_l))

        # Wang blend (convex mode, legacy)
        if use_gnn_blend and self.gnn_mode == "convex":
            if self.blend_max <= 0.0:
                self.register_buffer("raw_gnn_blend", torch.tensor(-1e9))
                self._blend_is_buffer = True
            else:
                init = float(gnn_blend_init) / max(self.blend_max, 1e-6)
                init = min(max(init, 1e-4), 1 - 1e-4)
                raw_init = math.log(init / (1.0 - init))
                self.raw_gnn_blend = nn.Parameter(torch.tensor(raw_init))
                self._blend_is_buffer = False
        else:
            self.raw_gnn_blend = None
            self._blend_is_buffer = False

        # Wang TB-ResNet residual scale (additive mode):
        #   V_dest = V_RUM + w_NN · V_GNN_raw
        # w_NN ≥ 0 via softplus, learnable; small init (0.1) so NN starts as
        # minor correction and grows only if data needs it.
        if use_gnn_blend and self.gnn_mode == "residual":
            raw_w = _inv_softplus(max(float(gnn_residual_scale_init), 1e-4))
            self.raw_gnn_residual_scale = nn.Parameter(torch.tensor(raw_w))
        else:
            self.raw_gnn_residual_scale = None

    # =============================================================================
    # Transformed parameter properties
    # =============================================================================

    @property
    def alpha_wage(self) -> torch.Tensor:
        """α ≥ 0 (wage attracts, Cervero/Wang prior)."""
        return F.softplus(self.raw_alpha_wage)

    @property
    def gamma_M(self) -> torch.Tensor:
        """γ ≥ 0 (jobs attractiveness, must be non-negative behaviourally)."""
        return F.softplus(self.raw_gamma_M)

    @property
    def nu_D(self) -> torch.Tensor:
        """ν ≤ 0 (competition repels, must be non-positive behaviourally)."""
        return -F.softplus(self.raw_nu_D)

    @property
    def delta_match(self) -> torch.Tensor:
        """δ ≥ 0 — Cervero 1999 multiplicative match-gravity coefficient.

        Used in V_upper as: γ_effective(i, j) = γ + δ · match_prob_raw[i, j],
        i.e. high-match destinations get an amplified jobs-count attractor.
        This is Cervero's A_i = Σ E_j · match[i, j] / d^γ in log-utility form.
        Match modulates the gravity coefficient instead of being a separate
        additive term — the prior v3b spec ('+ δ·match_z') was lit-inconsistent.
        """
        return F.softplus(self.raw_delta_match)

    @property
    def beta_t_per_mode(self) -> torch.Tensor:
        """β_t,m (intercept) ≤ 0 (time disutility at log_d=0, per mode)."""
        return -F.softplus(self.raw_beta_t_per_mode)

    @property
    def beta_t_slope_per_mode(self) -> torch.Tensor:
        """β_t,m,1 (slope on log_d) ≤ 0 — long-distance amplifies time disutility."""
        return -F.softplus(self.raw_beta_t_slope_per_mode)

    @property
    def lambda_per_borough(self) -> torch.Tensor:
        return self.lambda_eps_min + (1.0 - self.lambda_eps_min) * torch.sigmoid(
            self.raw_lambda_borough
        )

    @property
    def gnn_blend(self) -> Optional[torch.Tensor]:
        """Convex blend ∈ [0, blend_max] — legacy v3b/c mode only."""
        if self.raw_gnn_blend is None:
            return None
        return self.blend_max * torch.sigmoid(self.raw_gnn_blend)

    @property
    def gnn_residual_scale(self) -> Optional[torch.Tensor]:
        """w_NN ≥ 0 — Wang TB-ResNet additive residual scale (V_dest = V_RUM + w·V_NN)."""
        if self.raw_gnn_residual_scale is None:
            return None
        return F.softplus(self.raw_gnn_residual_scale)

    def lambda_for_destination(self, grid_borough_idx: torch.Tensor) -> torch.Tensor:
        return self.lambda_per_borough[grid_borough_idx]

    # =============================================================================
    # Snapshot for json logging
    # =============================================================================

    def snapshot(self) -> dict:
        with torch.no_grad():
            return {
                "alpha_wage": float(self.alpha_wage),
                "gamma_M": float(self.gamma_M),
                "nu_D": float(self.nu_D),
                "delta_match": float(self.delta_match),
                "beta_t_per_mode": self.beta_t_per_mode.tolist(),
                "beta_t_slope_per_mode": self.beta_t_slope_per_mode.tolist(),
                "asc_per_mode": self.asc_per_mode.tolist(),
                "theta_inc_per_mode": self.theta_inc_per_mode.tolist(),
                "theta_kids_per_mode": self.theta_kids_per_mode.tolist(),
                "theta_cars_per_mode": self.theta_cars_per_mode.tolist(),
                "lambda_per_borough": self.lambda_per_borough.tolist(),
                "lambda_b_mean": float(self.lambda_per_borough.mean()),
                "lambda_b_std": float(self.lambda_per_borough.std()),
                "lambda_b_min": float(self.lambda_per_borough.min()),
                "lambda_b_max": float(self.lambda_per_borough.max()),
                "gnn_blend": float(self.gnn_blend) if self.gnn_blend is not None else None,
                "gnn_residual_scale": float(self.gnn_residual_scale) if self.gnn_residual_scale is not None else None,
                "gnn_mode": self.gnn_mode,
            }
