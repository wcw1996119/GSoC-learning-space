"""RUM head for mode × tier mixture model.

Used by ``DualBranchInverseTrainer`` when both mode_share_per_origin and
income_tier_props are supplied. Replaces the simpler scalar β / single-mode
head from inverse_trainer.py.

Parameter structure (factorized to keep aggregate-data identifiability):

    β_{t, m, k}  =  -softplus(raw_beta[t, m]) × κ[k]

Where:
  raw_beta[t, m]  : (T, M) — per hour per mode time sensitivity (free)
  κ[k]            : (K,) — multiplicative tier scaling
                    κ[0] = 1.0 fixed (anchor for identification);
                    κ[k>0] = 1 + softplus(raw_kappa[k]) for monotone tiers

Plus shared scalars:
  γ      : log-distance coefficient (negative)
  δ      : OccMatch coefficient (positive expected)
  α      : 1.0 fixed buffer

This factorization gives 24 × M + (K - 1) + 2 = 72 + 2 + 2 = 76 free params
for the M=K=3 case, vs 216 params for fully free (T, M, K). Much friendlier
to identifiability under aggregate observed F.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    return math.log(math.exp(v) - 1.0)


class MixtureRUMHead(nn.Module):
    """Mode × tier mixture RUM head.

    Forward returns logits per (t, i, j, m, k) — but typically the trainer
    uses the helper methods below to compute mixture log-prob directly.
    """

    def __init__(
        self,
        n_hours: int,
        n_modes: int = 3,
        n_tiers: int = 3,
        beta_init_neg: float = -0.07,
        gamma_init_neg: float = -0.5,
        delta_init: float = 0.0,
        kappa_init: float = 0.0,        # initialised so all tiers start equal
        tier_specific_delta: bool = False,
        beta_monotone_walk_first: bool = False,
        fixed_beta_mask: Optional[torch.Tensor] = None,
        fixed_beta_vals: Optional[torch.Tensor] = None,
        use_gnn_blend: bool = False,
        gnn_blend_init: float = 0.5,
        blend_max: float = 1.0,
        mode_specific_gamma: bool = False,
    ):
        super().__init__()
        self.n_hours = n_hours
        self.n_modes = n_modes
        self.n_tiers = n_tiers
        self.tier_specific_delta = tier_specific_delta
        self.beta_monotone_walk_first = beta_monotone_walk_first
        self.use_gnn_blend = use_gnn_blend

        # alpha fixed at 1.0 (V_jt absorbs all scale; see inverse_trainer.py).
        self.register_buffer("alpha", torch.tensor(1.0))

        # raw_beta_t (T, M) — softplus-negated to enforce β < 0.
        # If beta_monotone_walk_first=True, mode order MUST be [walk, transit, car]
        # and β increments are nested via softplus to enforce |β_walk| ≥ |β_transit| ≥ |β_car|.
        target_b = -beta_init_neg
        raw_b = _inv_softplus(target_b)
        self.raw_beta_t = nn.Parameter(torch.full((n_hours, n_modes), raw_b))

        # Optional: literature-anchored β values for some modes (e.g. β_walk).
        # When fixed_beta_mask[m]=True, β[:, m] returns fixed_beta_vals[m]
        # (broadcast over T) instead of -softplus(raw_beta_t[:, m]).
        # raw_beta_t[:, m] for fixed modes still exists but receives no gradient
        # (where() detaches gradient flow at masked positions).
        if fixed_beta_mask is None:
            fixed_beta_mask = torch.zeros(n_modes, dtype=torch.bool)
        if fixed_beta_vals is None:
            fixed_beta_vals = torch.zeros(n_modes)
        self.register_buffer("fixed_beta_mask", fixed_beta_mask.bool())
        self.register_buffer("fixed_beta_vals", fixed_beta_vals.float())

        # raw_gamma — softplus-negated to enforce γ < 0.
        # If mode_specific_gamma=True, raw_gamma has shape (M,) so each mode
        # gets its own distance-decay coefficient. Walking commute typically
        # has steeper decay than transit/car, so this lifts a strong constraint
        # from the model. Backward-compat: default False keeps scalar gamma so
        # old checkpoints load cleanly.
        target_g = -gamma_init_neg
        raw_g = _inv_softplus(target_g)
        self.mode_specific_gamma = mode_specific_gamma
        if mode_specific_gamma:
            self.raw_gamma = nn.Parameter(torch.full((n_modes,), raw_g))
        else:
            self.raw_gamma = nn.Parameter(torch.tensor(raw_g))

        # delta (OccMatch coef).
        # Default: scalar (one δ for all tiers).
        # tier_specific_delta=True: shape (K,), δ[0] = anchor (= delta_init, fixed),
        #   δ[k>0] free → captures tier-varying labour-match preference.
        if tier_specific_delta:
            # raw_delta (K-1,) for tiers 1..K-1; tier 0 anchor at delta_init
            self.delta_anchor = float(delta_init)
            if n_tiers > 1:
                self.raw_delta = nn.Parameter(torch.zeros(n_tiers - 1))
            else:
                self.raw_delta = None
            self.delta = None  # not used when tier_specific
        else:
            self.delta = nn.Parameter(torch.tensor(float(delta_init)))
            self.raw_delta = None

        # kappa[k] for k=1..K-1; κ[0] fixed at 1.0
        # κ[k] = 1 + softplus(raw_kappa[k]) so κ ≥ 1 (monotone increase by tier)
        # If you don't want monotonicity, change to 1 + raw_kappa (unconstrained).
        if n_tiers > 1:
            self.raw_kappa = nn.Parameter(torch.full((n_tiers - 1,), float(kappa_init)))
        else:
            self.raw_kappa = None

        # Wang-style GNN/RUM blend (TB-ResNet, Wang et al. 2021):
        #   V_total = (1 - blend) * V_RUM + blend * V_GNN
        # blend ∈ (0, blend_max) via blend_max * sigmoid(raw); hard architectural
        # cap means δ ≤ blend_max no matter what gradient steps. blend_max=1.0
        # gives the original free-sigmoid behaviour; blend_max=0.3 enforces Wang's
        # recommended theory-dominant regime; blend_max=0.0 disables V_GNN entirely
        # (pure RUM mode). Trained blend value quantifies how much the GNN encoder
        # corrects RUM. If use_gnn_blend=False, the trainer uses additive logits
        # (current DUAL_HET default).
        self.blend_max = float(blend_max)
        if use_gnn_blend:
            if self.blend_max <= 0.0:
                # Pure-RUM mode: register blend at 0, no learnable param
                self.register_buffer(
                    "raw_gnn_blend",
                    torch.tensor(-1e9),  # sigmoid → 0
                )
                self._blend_is_buffer = True
            else:
                init = float(gnn_blend_init) / max(self.blend_max, 1e-6)
                init = min(max(init, 1e-4), 1 - 1e-4)
                raw_init = math.log(init / (1.0 - init))  # logit
                self.raw_gnn_blend = nn.Parameter(torch.tensor(raw_init))
                self._blend_is_buffer = False
        else:
            self.raw_gnn_blend = None
            self._blend_is_buffer = False

    @property
    def beta_t_per_mode(self) -> torch.Tensor:
        """β[t, m] of shape (T, M) — already negative.

        For modes flagged in ``fixed_beta_mask``, returns the literature-anchored
        value (no gradient). Other modes return -softplus(raw_beta_t[:, m]).
        """
        beta_learned = -F.softplus(self.raw_beta_t)                       # (T, M)
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
        """κ[k] of shape (K,). κ[0] = 1.0 fixed; κ[k>0] = 1 + softplus(raw_kappa[k])."""
        if self.raw_kappa is None:
            return torch.ones(self.n_tiers, device=self.raw_beta_t.device)
        kappa_rest = 1.0 + F.softplus(self.raw_kappa)              # ≥ 1
        return torch.cat([torch.ones(1, device=self.raw_beta_t.device), kappa_rest])

    @property
    def gamma(self) -> torch.Tensor:
        return -F.softplus(self.raw_gamma)

    def beta_full(self) -> torch.Tensor:
        """Full β tensor of shape (T, M, K). For diagnostic / VOT computation."""
        b_tm = self.beta_t_per_mode                                # (T, M)
        kap = self.kappa                                            # (K,)
        return b_tm.unsqueeze(-1) * kap.view(1, 1, -1)             # (T, M, K)

    @property
    def gnn_blend(self) -> Optional[torch.Tensor]:
        """Wang-style blend scalar ∈ (0, blend_max) or None when disabled.

        V_total = (1 - blend) * V_RUM + blend * V_GNN

        Hard cap via blend_max * sigmoid(raw_gnn_blend). When blend_max=1 the
        scalar lives in (0, 1) as before; when blend_max=0.3 the scalar is
        architecturally capped at 0.3 — Wang's recommended theory-dominant
        regime (TB-ResNet, 2021). When blend_max=0 the buffer is fixed at
        sigmoid(-1e9) ≈ 0, disabling V_GNN completely (pure-RUM ablation).
        """
        if self.raw_gnn_blend is None:
            return None
        return self.blend_max * torch.sigmoid(self.raw_gnn_blend)

    def delta_per_tier(self) -> torch.Tensor:
        """δ tensor of shape (K,). Either constant (broadcast scalar) or
        anchored (δ[0] = anchor, δ[k>0] = anchor + raw_delta[k-1])."""
        if self.tier_specific_delta:
            if self.raw_delta is None:
                return torch.full((self.n_tiers,), self.delta_anchor,
                                  device=self.raw_beta_t.device)
            anchor = torch.full((1,), self.delta_anchor, device=self.raw_delta.device)
            return torch.cat([anchor, anchor + self.raw_delta])    # (K,)
        else:
            return self.delta.expand(self.n_tiers)                 # broadcast (K,)
