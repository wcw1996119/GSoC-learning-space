"""Nested-logit RUM head (v3, minimal smoke version).

Extends v2's MixtureRUMHead with a per-mode dispersion parameter λ_m that
scales the destination-softmax temperature. This is a *partial* nested logit:
upper-nest mode choice is still pinned to observed Census π_m (we don't have
mode-specific OD data to learn it), but λ_m on the lower-nest destination
softmax captures within-mode destination correlation — the same mathematical
object that nested-logit λ plays in the full GEV framework.

If λ_m emerges < 1 after training, the nested structure is doing real work.
If λ_m ≈ 1 for all modes, IIA is fine on this data and v2's flat mixture
already captures the structure.

Param structure (extends v2):
    β_{t,m,k}  =  -softplus(raw_beta[t,m]) × κ_k        (same as v2)
    κ_k        =  1 (k=0) or 1 + softplus(raw_kappa[k]) (same as v2)
    γ          =  -softplus(raw_gamma)                  (same as v2)
    δ_k        =  delta_anchor (k=0) or anchor + raw_delta[k-1]  (same as v2)
    blend      =  blend_max · sigmoid(raw_blend)        (same as v2 TB-ResNet)

    NEW λ_m    =  ε_min + (1 - ε_min) · sigmoid(raw_lambda_m)  ∈ [ε_min, 1]
                 initialised so λ_m ≈ 1 (matches v2 behaviour at start).

ε_min default 0.05 to avoid λ → 0 numerical instability (gradient explosion
when 1/λ blows up).

See methodology/01_nested_logit_minimal.md for the math + diagnostic plan.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    return math.log(math.exp(v) - 1.0)


class NestedLogitRUMHead(nn.Module):
    """Mode × tier nested-logit RUM head.

    Same as v2's MixtureRUMHead but the per-mode destination softmax is
    computed at temperature 1/λ_m. λ_m ∈ [ε_min, 1] learned per mode.
    """

    def __init__(
        self,
        n_hours: int,
        n_modes: int = 3,
        n_tiers: int = 3,
        beta_init_neg: float = -0.07,
        gamma_init_neg: float = -0.5,
        delta_init: float = 0.0,
        kappa_init: float = 0.0,
        tier_specific_delta: bool = False,
        fixed_beta_mask: Optional[torch.Tensor] = None,
        fixed_beta_vals: Optional[torch.Tensor] = None,
        use_gnn_blend: bool = False,
        gnn_blend_init: float = 0.5,
        blend_max: float = 1.0,
        mode_specific_gamma: bool = False,
        # Nested logit additions
        lambda_init: float = 0.99,         # initial λ_m, close to 1 (≈v2)
        lambda_eps_min: float = 0.05,      # lower bound for λ_m
    ):
        super().__init__()
        self.n_hours = n_hours
        self.n_modes = n_modes
        self.n_tiers = n_tiers
        self.tier_specific_delta = tier_specific_delta
        self.use_gnn_blend = use_gnn_blend

        self.register_buffer("alpha", torch.tensor(1.0))

        target_b = -beta_init_neg
        raw_b = _inv_softplus(target_b)
        self.raw_beta_t = nn.Parameter(torch.full((n_hours, n_modes), raw_b))

        if fixed_beta_mask is None:
            fixed_beta_mask = torch.zeros(n_modes, dtype=torch.bool)
        if fixed_beta_vals is None:
            fixed_beta_vals = torch.zeros(n_modes)
        self.register_buffer("fixed_beta_mask", fixed_beta_mask.bool())
        self.register_buffer("fixed_beta_vals", fixed_beta_vals.float())

        target_g = -gamma_init_neg
        raw_g = _inv_softplus(target_g)
        self.mode_specific_gamma = mode_specific_gamma
        if mode_specific_gamma:
            self.raw_gamma = nn.Parameter(torch.full((n_modes,), raw_g))
        else:
            self.raw_gamma = nn.Parameter(torch.tensor(raw_g))

        if tier_specific_delta:
            self.delta_anchor = float(delta_init)
            if n_tiers > 1:
                self.raw_delta = nn.Parameter(torch.zeros(n_tiers - 1))
            else:
                self.raw_delta = None
            self.delta = None
        else:
            self.delta = nn.Parameter(torch.tensor(float(delta_init)))
            self.raw_delta = None

        if n_tiers > 1:
            self.raw_kappa = nn.Parameter(torch.full((n_tiers - 1,), float(kappa_init)))
        else:
            self.raw_kappa = None

        self.blend_max = float(blend_max)
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

        # ----- NESTED LOGIT: per-mode λ_m -----
        # λ_m = ε_min + (1 - ε_min) * sigmoid(raw_lambda_m)
        # initialize raw_lambda so λ_m ≈ lambda_init (default ≈ 1)
        self.lambda_eps_min = float(lambda_eps_min)
        # Solve: lambda_init = eps + (1-eps) * sigmoid(raw)
        # → sigmoid(raw) = (lambda_init - eps) / (1 - eps)
        s = (lambda_init - lambda_eps_min) / (1.0 - lambda_eps_min)
        s = min(max(s, 1e-4), 1 - 1e-4)
        raw_l = math.log(s / (1 - s))
        self.raw_lambda_m = nn.Parameter(torch.full((n_modes,), raw_l))

    @property
    def lambda_per_mode(self) -> torch.Tensor:
        """λ_m of shape (M,), bounded in [ε_min, 1]."""
        return self.lambda_eps_min + (1.0 - self.lambda_eps_min) * torch.sigmoid(
            self.raw_lambda_m
        )

    @property
    def beta_t_per_mode(self) -> torch.Tensor:
        beta_learned = -F.softplus(self.raw_beta_t)
        if self.fixed_beta_mask.any():
            fixed_b = self.fixed_beta_vals.view(1, -1).expand(self.n_hours, -1)
            return torch.where(
                self.fixed_beta_mask.view(1, -1).expand(self.n_hours, -1),
                fixed_b,
                beta_learned,
            )
        return beta_learned

    @property
    def kappa(self) -> torch.Tensor:
        if self.raw_kappa is None:
            return torch.ones(self.n_tiers, device=self.raw_beta_t.device)
        kappa_rest = 1.0 + F.softplus(self.raw_kappa)
        return torch.cat([torch.ones(1, device=self.raw_beta_t.device), kappa_rest])

    @property
    def gamma(self) -> torch.Tensor:
        return -F.softplus(self.raw_gamma)

    def beta_full(self) -> torch.Tensor:
        b_tm = self.beta_t_per_mode
        kap = self.kappa
        return b_tm.unsqueeze(-1) * kap.view(1, 1, -1)

    @property
    def gnn_blend(self) -> Optional[torch.Tensor]:
        if self.raw_gnn_blend is None:
            return None
        return self.blend_max * torch.sigmoid(self.raw_gnn_blend)

    def delta_per_tier(self) -> torch.Tensor:
        if self.tier_specific_delta:
            if self.raw_delta is None:
                return torch.full((self.n_tiers,), self.delta_anchor,
                                  device=self.raw_beta_t.device)
            anchor = torch.full((1,), self.delta_anchor, device=self.raw_delta.device)
            return torch.cat([anchor, anchor + self.raw_delta])
        else:
            return self.delta.expand(self.n_tiers)
