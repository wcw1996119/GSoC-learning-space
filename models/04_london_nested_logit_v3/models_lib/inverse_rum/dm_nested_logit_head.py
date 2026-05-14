"""DM_NestedLogitHead — Destination→Mode nested logit with borough-level λ.

Implements Phase 3b' of the v3 design (see methodology/02_dm_nested_design.md).
Replaces v2's MixtureRUMHead with a structurally different choice model:

  Upper nest (destination):
      V_dest(i, j, t) = γ · log_d_ij
                      + δ(d_ij) · OccMatch_ij             ← non-linear (linear in log_d)
                      + α · V_GNN_jt                       ← GNN destination embedding
                      + λ_{b(j)} · IV_mode(i, j, t)        ← inclusive value from lower nest

  Lower nest (mode | destination):
      V_mode(i, j, t, m) = β_m(d_ij) · t_{ij,m}            ← non-linear time disutility
                         + θ_m^T · X_{i, m}                ← mode-level origin features
                         + α_m                              ← mode bias

  Where:
      β_m(d_ij) = β_m,0 + β_m,1 · log_d_ij                 (steeper at long distance)
      δ(d_ij)   = δ_0   + δ_1   · log_d_ij                 (matching-distance interaction)
      λ_b ∈ (0, 1] via λ_b = ε_min + (1-ε_min) · sigmoid   (33 boroughs by default)
      IV_mode(i,j,t) = log Σ_m exp(V_mode(i,j,t,m) / λ_{b(j)})

  Probabilities:
      P_M(m | i, j, t) = softmax_m( V_mode / λ_{b(j)} )
      P_D(j | i, t)    = softmax_j( V_dest )
      P(j, m | i, t)   = P_D · P_M
      P(j | i, t)      = P_D · Σ_m P_M = P_D    (marginalised since we observe aggregate F_{ij,t})

  Loss (in trainer, not here):
      L_total = NLL_destination  +  λ_kl · cross_entropy_mode

This head only declares & exposes parameters; the actual forward computation
(V_dest, V_mode, IV, KL term) lives in train_dm_nested.py to keep the head
testable and dependency-free.

Parameter inventory (default M=3, F_mode=3, K=3, B=33):
    raw_beta_intercept      (M,)         per-mode time disutility intercept
    raw_beta_slope          (M,)         per-mode time disutility slope on log_d
    alpha_per_mode          (M,)         mode bias
    theta_per_mode          (M, F_mode)  mode-level feature weights
    raw_gamma               scalar       gravity log-distance
    delta_intercept         scalar       OccMatch intercept
    delta_slope             scalar       OccMatch slope on log_d
    raw_kappa               (K-1,)       tier scaling (κ_0=1 anchored)
    raw_lambda_borough      (B,)         borough-level nest correlation
    raw_gnn_blend           scalar       optional Wang TB-ResNet blend

Total free params (defaults):
    3 + 3 + 3 + 9 + 1 + 1 + 1 + 2 + 33 + 1 = 57 RUM params
    (compared to v2 MixtureRUMHead: ~76 params; nested is parsimonious)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    """Inverse of softplus, used to initialise raw_x so softplus(raw_x) = target_v."""
    return math.log(math.exp(v) - 1.0)


def _inv_sigmoid(s: float) -> float:
    """Inverse of sigmoid, used to initialise raw_x so sigmoid(raw_x) ≈ s."""
    s = min(max(s, 1e-6), 1 - 1e-6)
    return math.log(s / (1 - s))


class DMNestedLogitHead(nn.Module):
    """Destination→Mode nested-logit head with borough-level λ and non-linear V.

    Parameters
    ----------
    n_modes : int
        Number of travel modes (default 3 for car/transit/walk).
    n_tiers : int
        Number of income tiers (default 3).
    n_boroughs : int
        Number of distinct λ values to learn (default 33 London boroughs).
        Set to 1 for the "single-λ" simplification, or 2-3 for category-λ.
    n_mode_features : int
        Number of mode-level origin features X_{i,m} (default 3).
    beta_intercept_init : float
        Initial β_m,0 (negative, time disutility). Default -0.07.
    beta_slope_init : float
        Initial β_m,1 (slope on log_d). Default 0 (start linear).
    gamma_init : float
        Initial γ (log-distance gravity coefficient). Default -0.5.
    delta_intercept_init : float
        Initial δ_0 (OccMatch coefficient). Default 0.
    delta_slope_init : float
        Initial δ_1 (OccMatch × log_d interaction). Default 0.
    kappa_init : float
        Initial κ_k for k≥1 (softplus pre-image). Default 0 → κ = ln(2) ≈ 0.69.
    lambda_init : float
        Initial λ_b ∈ (ε_min, 1]. Default 0.95 (close to flat MNL).
    lambda_eps_min : float
        Lower bound on λ_b to avoid gradient explosion. Default 0.05.
    use_gnn_blend : bool
        Enable Wang TB-ResNet blend (1-δ)·V_RUM + δ·V_GNN.
    gnn_blend_init : float
        Initial blend value (default 0.5).
    blend_max : float
        Hard cap on blend (default 1.0 = free; 0.3 = Wang theory-dominant; 0 = pure RUM).
    """

    def __init__(
        self,
        n_modes: int = 3,
        n_tiers: int = 3,
        n_boroughs: int = 33,
        n_mode_features: int = 3,
        beta_intercept_init: float = -0.07,
        beta_slope_init: float = 0.0,
        gamma_init: float = -0.5,
        delta_intercept_init: float = 0.0,
        delta_slope_init: float = 0.0,
        kappa_init: float = 0.0,
        lambda_init: float = 0.95,
        lambda_eps_min: float = 0.05,
        use_gnn_blend: bool = False,
        gnn_blend_init: float = 0.5,
        blend_max: float = 1.0,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.n_tiers = n_tiers
        self.n_boroughs = n_boroughs
        self.n_mode_features = n_mode_features
        self.lambda_eps_min = float(lambda_eps_min)
        self.use_gnn_blend = use_gnn_blend
        self.blend_max = float(blend_max)

        # ----- β_m(d) = β_m,0 + β_m,1 · log_d_ij (per-mode, distance-dependent) -----
        # β_m,0 enforced negative via -softplus
        # β_m,1 is unconstrained (typically small negative, makes far trips steeper)
        target_b0 = -beta_intercept_init                                   # positive
        raw_b0 = _inv_softplus(target_b0)
        self.raw_beta_intercept = nn.Parameter(torch.full((n_modes,), raw_b0))
        self.beta_slope = nn.Parameter(torch.full((n_modes,), float(beta_slope_init)))

        # ----- α_m: mode bias (free) -----
        self.alpha_per_mode = nn.Parameter(torch.zeros(n_modes))

        # ----- θ_m: mode-level feature weights (M, F_mode), free -----
        self.theta_per_mode = nn.Parameter(torch.zeros(n_modes, n_mode_features))

        # ----- γ · log_d (negative) -----
        target_g = -gamma_init
        raw_g = _inv_softplus(target_g)
        self.raw_gamma = nn.Parameter(torch.tensor(raw_g))

        # ----- δ(d) = δ_0 + δ_1 · log_d, both free -----
        self.delta_intercept = nn.Parameter(torch.tensor(float(delta_intercept_init)))
        self.delta_slope = nn.Parameter(torch.tensor(float(delta_slope_init)))

        # ----- κ_k tier scaling (κ_0=1 anchored, κ_k>0 = 1 + softplus(raw)) -----
        if n_tiers > 1:
            self.raw_kappa = nn.Parameter(torch.full((n_tiers - 1,), float(kappa_init)))
        else:
            self.raw_kappa = None

        # ----- λ_b: borough-level nest correlation in [ε_min, 1] -----
        # λ_b = ε_min + (1-ε_min) · sigmoid(raw_lambda_borough)
        s = (lambda_init - lambda_eps_min) / (1.0 - lambda_eps_min)
        raw_l = _inv_sigmoid(s)
        self.raw_lambda_borough = nn.Parameter(torch.full((n_boroughs,), raw_l))

        # ----- Wang TB-ResNet GNN/RUM blend (optional) -----
        if use_gnn_blend:
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

    # =============================================================================
    # Properties exposing learned values (transformed to natural scale)
    # =============================================================================

    @property
    def beta_intercept_per_mode(self) -> torch.Tensor:
        """β_m,0 of shape (M,), enforced negative via -softplus."""
        return -F.softplus(self.raw_beta_intercept)

    @property
    def gamma(self) -> torch.Tensor:
        """γ scalar, enforced negative via -softplus."""
        return -F.softplus(self.raw_gamma)

    @property
    def kappa(self) -> torch.Tensor:
        """κ_k of shape (K,). κ_0 = 1 fixed; κ_k≥1 = 1 + softplus(raw)."""
        if self.raw_kappa is None:
            return torch.ones(self.n_tiers, device=self.raw_beta_intercept.device)
        kappa_rest = 1.0 + F.softplus(self.raw_kappa)
        return torch.cat(
            [torch.ones(1, device=self.raw_kappa.device), kappa_rest]
        )

    @property
    def lambda_per_borough(self) -> torch.Tensor:
        """λ_b of shape (B,), bounded in [ε_min, 1]."""
        return self.lambda_eps_min + (1.0 - self.lambda_eps_min) * torch.sigmoid(
            self.raw_lambda_borough
        )

    @property
    def gnn_blend(self) -> Optional[torch.Tensor]:
        """Wang TB-ResNet blend ∈ (0, blend_max], or None when disabled."""
        if self.raw_gnn_blend is None:
            return None
        return self.blend_max * torch.sigmoid(self.raw_gnn_blend)

    # =============================================================================
    # Helper functions used by the trainer
    # =============================================================================

    def beta_at_log_d(self, log_d: torch.Tensor) -> torch.Tensor:
        """β_m(d) = β_m,0 + β_m,1 · log_d_ij, evaluated for given log_d.

        Parameters
        ----------
        log_d : (N, N) or (T, N, N) — log distance.

        Returns
        -------
        beta : same leading shape as log_d, with an extra trailing M dimension.
            Each [i, j, m] = β_m,0 + β_m,1 · log_d[i, j].
        """
        b0 = self.beta_intercept_per_mode                                  # (M,)
        b1 = self.beta_slope                                                # (M,)
        return b0.view(*([1] * log_d.dim()), -1) + b1.view(
            *([1] * log_d.dim()), -1
        ) * log_d.unsqueeze(-1)                                            # (..., N, N, M)

    def delta_at_log_d(self, log_d: torch.Tensor) -> torch.Tensor:
        """δ(d) = δ_0 + δ_1 · log_d_ij, evaluated for given log_d.

        Returns same shape as log_d.
        """
        return self.delta_intercept + self.delta_slope * log_d

    def lambda_for_destination(self, grid_borough_idx: torch.Tensor) -> torch.Tensor:
        """λ_{b(j)} per destination, shape (N,).

        Parameters
        ----------
        grid_borough_idx : (N,) long tensor mapping each destination to its borough.
        """
        return self.lambda_per_borough[grid_borough_idx]                    # (N,)

    def theta_dot_X(self, mode_level_X: torch.Tensor) -> torch.Tensor:
        """θ_m^T · X_{i, m} of shape (N, M).

        Parameters
        ----------
        mode_level_X : (N, M, F_mode) — per-origin per-mode feature vectors.

        Returns
        -------
        (N, M) inner product.
        """
        # theta_per_mode: (M, F_mode)
        # mode_level_X:   (N, M, F_mode)
        # result:         (N, M)
        return (mode_level_X * self.theta_per_mode.unsqueeze(0)).sum(dim=-1)

    # =============================================================================
    # Convenience: snapshot all learned values for logging / final eval
    # =============================================================================

    def snapshot(self) -> dict:
        """Snapshot of current parameter values (for json logging)."""
        with torch.no_grad():
            return {
                "beta_intercept_per_mode": self.beta_intercept_per_mode.tolist(),
                "beta_slope_per_mode": self.beta_slope.tolist(),
                "alpha_per_mode": self.alpha_per_mode.tolist(),
                "theta_per_mode": self.theta_per_mode.tolist(),                # (M, F_mode)
                "gamma": float(self.gamma),
                "delta_intercept": float(self.delta_intercept),
                "delta_slope": float(self.delta_slope),
                "kappa": self.kappa.tolist(),
                "lambda_per_borough": self.lambda_per_borough.tolist(),
                "gnn_blend": float(self.gnn_blend) if self.gnn_blend is not None else None,
            }
