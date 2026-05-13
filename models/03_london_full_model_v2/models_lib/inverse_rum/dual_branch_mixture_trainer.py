"""Dual-branch encoder + mode×tier mixture RUM trainer.

Extends ``DualBranchInverseTrainer`` with:
  * Mode mixture: each origin has Pr(car), Pr(transit), Pr(walk) from Census;
    each mode has its own time sensitivity β_t,m.
  * Tier mixture: each origin has Pr(low/mid/high income); each tier scales
    β by a learnable κ_k (κ_0 = 1 anchor).
  * Combined: P(j | i, t) = Σ_m Σ_k π_m(i) × π_k(i) × softmax(α V + β_{t,m,k} t_{ij}^{(m)} + γ log_d + δ OccMatch)

Parameters reverse-engineered:
  - Encoder weights (≈33k) — same as DUAL_MIN
  - β_{t,m}: (24, M) — per-hour per-mode time sensitivity
  - κ_k: (K-1,) — tier scaling (κ_0 = 1 fixed)
  - γ: log-distance coefficient
  - δ: OccMatch coefficient (optional)
  - α: 1.0 fixed

Compute: M × K = 9 softmax computations per epoch (9× DUAL_MIN). Memory uses
incremental ``torch.logaddexp`` so we never stack all 9 (T, N, N) tensors.

Identifiability: with 76 free RUM parameters + 33k encoder vs 5.7M observed
OD-hour cells, strongly over-identified. mode-specific β identifiable when
t_ij^(m) varies across (i, j) pairs (Manski-McFadden 1981). κ_k identifiable
because π_k(i) varies across origins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_branch_encoder import DualBranchEncoder
from .mixture_head import MixtureRUMHead
from .inverse_trainer import TrainingLog


def make_distance_aware_mode_share(
    origin_mode_share: torch.Tensor,
    dist_km: torch.Tensor,
    mode_names: list,
    walk_threshold_km: float = 5.0,
    cycle_threshold_km: float = 15.0,
) -> torch.Tensor:
    """Build pair-level mode share with distance-restricted active modes.

    Standard 'choice set generation' (Manski 1977; Ben-Akiva & Boccara 1995):
    walk is only an option for OD pairs with d_ij < walk_threshold_km
    (default 5 km); cycle (if present) within cycle_threshold_km. For pairs
    beyond the threshold, the active-mode share is redistributed
    proportionally to motorised modes (car, transit).

    Parameters
    ----------
    origin_mode_share : (N, M) — per-origin mode share, rows sum to 1.
    dist_km           : (N, N) — pairwise Euclidean distance in km.
    mode_names        : list of M mode-name strings (e.g., ['car', 'transit', 'walk']).
    walk_threshold_km : trips below this distance keep their walk share.
    cycle_threshold_km: trips below this distance keep their cycle share (if present).

    Returns
    -------
    pair_mode_share : (N, N, M) — per (origin, destination) mode share, rows sum to 1.
    """
    N, M = origin_mode_share.shape
    pair = origin_mode_share.unsqueeze(1).expand(N, N, M).clone()

    active_modes = []
    if "walk" in mode_names:
        active_modes.append((mode_names.index("walk"), walk_threshold_km))
    if "cycle" in mode_names:
        active_modes.append((mode_names.index("cycle"), cycle_threshold_km))

    if not active_modes:
        return pair

    motorised_idx = [m for m in range(M) if m not in [a[0] for a in active_modes]]

    for m_idx, threshold in active_modes:
        infeasible = dist_km >= threshold
        share_to_redistribute = pair[..., m_idx].clone()
        if motorised_idx:
            mot_total = pair[..., motorised_idx].sum(dim=-1).clamp(min=1e-9)
            for m_keep in motorised_idx:
                scale = pair[..., m_keep] / mot_total
                pair[..., m_keep] = torch.where(
                    infeasible,
                    pair[..., m_keep] + share_to_redistribute * scale,
                    pair[..., m_keep],
                )
        pair[..., m_idx] = torch.where(
            infeasible, torch.zeros_like(share_to_redistribute), share_to_redistribute
        )

    return pair


class DualBranchMixtureTrainer:
    """Joint inverse-RUM trainer: dual-branch encoder + mode×tier mixture head.

    Canonical use::

        trainer = DualBranchMixtureTrainer(
            X_static=X_s, X_dynamic=X_d, edge_index=edge_index,
            observed_OD=F_ij_t,
            t_ij_per_mode={'car': t_car, 'transit': t_transit, 'walk': t_walk},
            mode_share_per_origin=pi_m,         # (N, M)
            income_tier_props=pi_k,             # (N, K)
            log_d_ij=log_d, occ_match=occ_match,
            train_mask=tm, val_mask=vm,
            epochs=50, patience=15,
        )
        encoder, head, log = trainer.fit()
    """

    def __init__(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        observed_OD: torch.Tensor,
        t_ij_per_mode: Dict[str, torch.Tensor],
        mode_share_per_origin: torch.Tensor,
        income_tier_props: torch.Tensor,
        log_d_ij: torch.Tensor,
        occ_match: Optional[torch.Tensor] = None,
        encoder: Optional[DualBranchEncoder] = None,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
        seed: int = 42,
        hidden_dim: int = 32,
        gru_hidden: int = 32,
        n_sage_layers: int = 2,
        tcn_kernels: Tuple[int, ...] = (3, 5, 7),
        lr_theta: float = 1e-3,
        lr_rum: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = False,
        tier_specific_delta: bool = False,
        fixed_beta_per_mode: Optional[Dict[str, float]] = None,
        use_gnn_blend: bool = False,
        gnn_blend_init: float = 0.5,
        mode_specific_gamma: bool = False,
        # ---- IRM (Module 1, Paper A Day 1) ----
        irm_variant: str = "none",                # "none" | "rex" | "irmv1"
        irm_lambda: float = 0.0,                  # 0 disables
        irm_warmup_epochs: int = 0,
        origin_env_idx: Optional[torch.Tensor] = None,  # (N,) long; e.g. grid_borough_idx
        irm_n_envs: Optional[int] = None,
    ):
        # ---- shapes ----------------------------------------------------------
        if X_static.dim() != 2:
            raise ValueError(f"X_static must be (N, F_s); got {tuple(X_static.shape)}")
        N, F_s = X_static.shape
        if X_dynamic.dim() != 3 or X_dynamic.shape[1] != N:
            raise ValueError(
                f"X_dynamic must be (T, N, F_d) with N={N}; got {tuple(X_dynamic.shape)}"
            )
        T, _, F_d = X_dynamic.shape
        if observed_OD.shape != (T, N, N):
            raise ValueError(f"observed_OD must be ({T}, {N}, {N}); got {tuple(observed_OD.shape)}")
        if log_d_ij.shape != (N, N):
            raise ValueError(f"log_d_ij must be ({N}, {N})")

        # mode share — accept (N, M) [origin-level] or (N, N, M) [pair-level]
        ms = mode_share_per_origin
        ms = ms if isinstance(ms, torch.Tensor) else torch.tensor(ms)
        if ms.dim() == 2:
            if ms.shape[0] != N:
                raise ValueError(
                    f"mode_share (N, M) must have N={N}; got {tuple(ms.shape)}"
                )
            M = ms.shape[1]
            # Broadcast to pair-level: same mode share for all destinations
            ms = ms.unsqueeze(1).expand(N, N, M).contiguous()                  # (N, N, M)
        elif ms.dim() == 3:
            if ms.shape[0] != N or ms.shape[1] != N:
                raise ValueError(
                    f"mode_share (N, N, M) must have N={N}; got {tuple(ms.shape)}"
                )
            M = ms.shape[2]
        else:
            raise ValueError(
                f"mode_share must be (N, M) or (N, N, M); got {tuple(ms.shape)}"
            )

        # tier props (N, K)
        ip = income_tier_props
        ip = ip if isinstance(ip, torch.Tensor) else torch.tensor(ip)
        if ip.dim() != 2 or ip.shape[0] != N:
            raise ValueError(
                f"income_tier_props must be (N={N}, K); got {tuple(ip.shape)}"
            )
        K = ip.shape[1]

        # t_ij_per_mode dict — broadcast (N, N) → (T, N, N) if needed
        t_dict_validated = {}
        for name, tens in t_ij_per_mode.items():
            t = tens if isinstance(tens, torch.Tensor) else torch.tensor(tens)
            if t.dim() == 2:
                if t.shape != (N, N):
                    raise ValueError(f"t_ij_per_mode[{name}] (N,N) must be ({N},{N})")
                t = t.unsqueeze(0).expand(T, N, N)
            elif t.dim() == 3:
                if t.shape != (T, N, N):
                    raise ValueError(f"t_ij_per_mode[{name}] (T,N,N) must be ({T},{N},{N})")
            t_dict_validated[name] = t.contiguous().float()

        if len(t_dict_validated) != M:
            raise ValueError(f"len(t_ij_per_mode) = {len(t_dict_validated)} != M = {M}")

        # OccMatch
        if occ_match is not None:
            occ_t = occ_match if isinstance(occ_match, torch.Tensor) else torch.tensor(occ_match)
            if occ_t.shape != (N, N):
                raise ValueError(f"occ_match must be ({N}, {N})")
            self.occ_match = occ_t.float()
            self._enable_occ_match = True
        else:
            self.occ_match = None
            self._enable_occ_match = False

        # masks
        if val_mask is None:
            val_mask = torch.zeros(N, dtype=torch.bool)
            val_mask[::5] = True
        if train_mask is None:
            train_mask = ~val_mask
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        assert (train_mask & val_mask).sum().item() == 0

        # ---- store -----------------------------------------------------------
        self.N, self.T, self.F_s, self.F_d, self.M, self.K = N, T, F_s, F_d, M, K
        self.device = device
        self.seed = seed
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.lr_theta = lr_theta
        self.lr_rum = lr_rum
        self.weight_decay = weight_decay

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.X_static = X_static.to(device).float()
        self.X_dynamic = X_dynamic.to(device).float()
        self.edge_index = edge_index.to(device)
        self.observed_OD = observed_OD.to(device).float()
        self.log_d_ij = log_d_ij.to(device).float()
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.t_per_mode = {name: t.to(device) for name, t in t_dict_validated.items()}
        self.mode_names = list(t_dict_validated.keys())
        self.pi_m_pair = ms.to(device).float()                            # (N, N, M) — pair-level
        self.pi_k = ip.to(device).float()                                 # (N, K)
        if self.occ_match is not None:
            self.occ_match = self.occ_match.to(device)

        # log priors per (origin, destination, mode, tier): shape (N, N, M, K)
        # log_pi_mk_pair[i, j, m, k] = log π_m(i, j) + log π_k(i)
        log_pi_m = torch.log(self.pi_m_pair.clamp(min=1e-9))              # (N, N, M)
        log_pi_k = torch.log(self.pi_k.clamp(min=1e-9))                   # (N, K)
        # broadcast: (N, N, M, 1) + (N, 1, 1, K) -> (N, N, M, K)
        self.log_pi_mk_pair = log_pi_m.unsqueeze(-1) + log_pi_k.view(N, 1, 1, K)

        # ---- model -----------------------------------------------------------
        if encoder is None:
            encoder = DualBranchEncoder(
                static_dim=F_s, dyn_dim=F_d,
                hidden_dim=hidden_dim, gru_hidden=gru_hidden,
                n_sage_layers=n_sage_layers, tcn_kernels=tcn_kernels,
            )
        self.encoder = encoder.to(device)

        # Translate fixed_beta_per_mode dict to tensors aligned with mode_names order
        fixed_mask_t = torch.zeros(M, dtype=torch.bool)
        fixed_vals_t = torch.zeros(M)
        if fixed_beta_per_mode:
            for mname, val in fixed_beta_per_mode.items():
                if mname not in self.mode_names:
                    raise ValueError(
                        f"fixed_beta_per_mode key '{mname}' not in mode_names {self.mode_names}"
                    )
                idx = self.mode_names.index(mname)
                fixed_mask_t[idx] = True
                fixed_vals_t[idx] = float(val)

        self.rum = MixtureRUMHead(n_hours=T, n_modes=M, n_tiers=K,
                                   tier_specific_delta=tier_specific_delta,
                                   fixed_beta_mask=fixed_mask_t,
                                   fixed_beta_vals=fixed_vals_t,
                                   use_gnn_blend=use_gnn_blend,
                                   gnn_blend_init=gnn_blend_init,
                                   mode_specific_gamma=mode_specific_gamma).to(device)
        self._tier_specific_delta = tier_specific_delta
        self._fixed_beta_per_mode = fixed_beta_per_mode or {}
        self._use_gnn_blend = use_gnn_blend

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": lr_theta,
                 "weight_decay": weight_decay},
                {"params": self.rum.parameters(), "lr": lr_rum,
                 "weight_decay": 0.0},
            ]
        )

        # ---- IRM config ------------------------------------------------------
        if irm_variant not in ("none", "rex", "irmv1"):
            raise ValueError(
                f"irm_variant must be one of 'none'/'rex'/'irmv1'; got '{irm_variant}'"
            )
        self.irm_variant = irm_variant
        self.irm_lambda = float(irm_lambda)
        self.irm_warmup_epochs = int(irm_warmup_epochs)
        if irm_variant != "none" and self.irm_lambda > 0:
            if origin_env_idx is None:
                raise ValueError("origin_env_idx required when irm_variant != 'none'")
            env_idx = origin_env_idx if isinstance(origin_env_idx, torch.Tensor) \
                else torch.tensor(origin_env_idx)
            if env_idx.shape != (N,):
                raise ValueError(
                    f"origin_env_idx must be (N={N},); got {tuple(env_idx.shape)}"
                )
            self.origin_env_idx = env_idx.to(device).long()
            self.irm_n_envs = int(irm_n_envs) if irm_n_envs is not None \
                else int(self.origin_env_idx.max().item() + 1)
        else:
            self.origin_env_idx = None
            self.irm_n_envs = 0

    # ---------------------------------------------------------------- forward
    def _forward_V(self, norm_stats=None) -> torch.Tensor:
        return self.encoder(self.X_static, self.X_dynamic, self.edge_index,
                            norm_stats=norm_stats)

    # ---------------------------------------------------------------- log P
    def _per_hour_log_p(self, V_jt: torch.Tensor,
                       dummy_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """log P(j | i, t) under mode×tier mixture.

        For each (m, k):
            β_eff = β_{t,m} × κ_k                                  (T,)
            logits[t, i, j] = α V_{t,j} + β_eff × t_{ij}^{(m)} + γ log_d + δ OccMatch
            log_p_mk[t, i, j] = log_softmax_j(logits)
        Then mixture:
            log P(j | i, t) = logsumexp_{m, k} (log_p_mk + log π_m(i) + log π_k(i))

        Uses incremental ``torch.logaddexp`` to avoid stacking all 9 (T, N, N)
        tensors. Peak memory ~2 (T, N, N) tensors at any time.
        """
        T, N = V_jt.shape
        alpha = self.rum.alpha                                            # 1.0
        gamma_raw = self.rum.gamma                                        # scalar or (M,)
        if gamma_raw.dim() == 0:
            gamma_vec = gamma_raw.unsqueeze(0).expand(self.M)             # (M,) broadcast
        else:
            gamma_vec = gamma_raw                                         # (M,)
        beta_per_mode = self.rum.beta_t_per_mode                          # (T, M)
        kappa = self.rum.kappa                                            # (K,)
        # Per-tier δ — shape (K,); equals constant if tier_specific_delta=False
        delta_k = self.rum.delta_per_tier() if self._enable_occ_match else None
        # Wang-style blend scalar in (0,1); None when disabled (additive default)
        blend = self.rum.gnn_blend                                        # scalar or None

        V_b = V_jt.view(T, 1, N).expand(T, N, N)                          # (T, N, N)
        d_b = self.log_d_ij.view(1, N, N)                                 # (1, N, N)

        log_p = None
        for m, name in enumerate(self.mode_names):
            t_m = self.t_per_mode[name]                                   # (T, N, N)
            for k in range(self.K):
                beta_eff = (beta_per_mode[:, m] * kappa[k]).view(T, 1, 1) # (T, 1, 1)
                # V_RUM_ij = β t + γ log d + δ_occ OccMatch (pure structural utility)
                V_rum = beta_eff * t_m + gamma_vec[m] * d_b
                if delta_k is not None and self.occ_match is not None:
                    V_rum = V_rum + delta_k[k] * self.occ_match.view(1, N, N)
                # V_GNN_ij = α · V_jt (broadcast over origins)
                V_gnn = alpha * V_b
                if blend is None:
                    # Default additive logits (current DUAL_HET behaviour)
                    logits = V_gnn + V_rum
                else:
                    # Wang TB-ResNet (2021): V_total = (1 - δ) V_RUM + δ V_GNN
                    logits = (1.0 - blend) * V_rum + blend * V_gnn
                # IRM-v1 trick (Arjovsky 2019): scale logits by dummy w=1 so
                # `d softmax / d w` ≠ 0; gradient is used for the env-invariance
                # penalty downstream. Forward output unchanged at w=1.
                if dummy_scale is not None:
                    logits = logits * dummy_scale
                log_p_mk = torch.log_softmax(logits, dim=-1)              # (T, N, N)
                # Pair-level log mixture prior: log π_m(i, j) + log π_k(i)
                log_w_ij = self.log_pi_mk_pair[:, :, m, k].view(1, N, N)  # (1, N, N)
                log_p_mk_weighted = log_p_mk + log_w_ij
                if log_p is None:
                    log_p = log_p_mk_weighted
                else:
                    log_p = torch.logaddexp(log_p, log_p_mk_weighted)
        return log_p                                                       # (T, N, N)

    def _nll(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T, N, _ = log_p.shape
        mask_f = mask.view(1, N, 1).float()
        flow = self.observed_OD * mask_f
        denom = max(int((flow.sum(dim=2) > 0).sum()), 1)
        return -(flow * log_p).sum() / denom

    def _per_origin_nll(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """NLL summed over (t, j) for each origin i — used for IRM penalties.

        Returns (N,) tensor. Masked-out origins are 0 by construction (flow
        is zeroed); pass ``mask`` to IRM penalty to skip them when computing
        per-env risks.
        """
        T, N, _ = log_p.shape
        mask_f = mask.view(1, N, 1).float()
        flow = self.observed_OD * mask_f                       # (T, N, N)
        # per origin: sum over t and j, then divide by n_active_hours_for_i
        nll_per_origin_unnorm = -(flow * log_p).sum(dim=(0, 2))           # (N,)
        # normalize by # active hours per origin (so V-REx variance is in
        # comparable units to the main NLL)
        n_active_t_per_i = (flow.sum(dim=2) > 0).float().sum(dim=0)        # (N,)
        return nll_per_origin_unnorm / n_active_t_per_i.clamp(min=1.0)     # (N,)

    def _cpc(self, log_p: torch.Tensor, mask: torch.Tensor) -> float:
        with torch.no_grad():
            P = log_p.exp()
            row_sum = self.observed_OD.sum(dim=2, keepdim=True)
            pred = P * row_sum
            num = 2.0 * torch.minimum(pred[:, mask], self.observed_OD[:, mask]).sum()
            den = (pred[:, mask].sum() + self.observed_OD[:, mask].sum()).clamp(min=1.0)
            return float(num / den)

    # ---------------------------------------------------------------- fit
    def fit(self) -> Tuple[DualBranchEncoder, MixtureRUMHead, TrainingLog]:
        log = TrainingLog()
        # IRM mode selection: IRM intentionally trades in-sample NLL for OOD,
        # so using val_nll for best_state + patience early-stop systematically
        # kills IRM training right after warmup. Use val CPC instead (higher
        # = better fit quality, decoupled from penalty term). Also disable
        # early-stop in IRM mode (Arjovsky 2019 / Krueger 2021 train fixed
        # epochs).
        irm_mode = (self.irm_variant != "none" and self.irm_lambda > 0)
        best_metric = -float("inf") if irm_mode else float("inf")
        no_improve = 0
        best_state = None

        from .irm_penalty import (
            compute_rex_penalty, compute_irmv1_penalty, warmup_lambda,
        )

        for ep in range(self.epochs):
            self.encoder.train(); self.rum.train()
            self.optimizer.zero_grad()

            # IRM: only attach dummy_scale to the graph when IRM-v1 is *active*
            # this epoch (post warm-up). Saves a tiny bit of memory + matches
            # ERM-only forward exactly outside the penalty regime.
            cur_lambda = warmup_lambda(ep, self.irm_warmup_epochs, self.irm_lambda)
            need_dummy = (self.irm_variant == "irmv1" and cur_lambda > 0)
            dummy_scale = None
            if need_dummy:
                dummy_scale = torch.tensor(1.0, requires_grad=True, device=self.device)

            V_jt = self._forward_V()
            log_p = self._per_hour_log_p(V_jt, dummy_scale=dummy_scale)
            loss_main = self._nll(log_p, self.train_mask)

            penalty = log_p.new_zeros(())
            if self.irm_variant != "none" and cur_lambda > 0:
                per_origin_nll = self._per_origin_nll(log_p, self.train_mask)
                if self.irm_variant == "rex":
                    penalty = compute_rex_penalty(
                        per_origin_nll, self.origin_env_idx,
                        self.train_mask.bool(), self.irm_n_envs,
                    )
                else:  # irmv1
                    penalty = compute_irmv1_penalty(
                        per_origin_nll, self.origin_env_idx,
                        self.train_mask.bool(), dummy_scale, self.irm_n_envs,
                    )
            loss = loss_main + cur_lambda * penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.rum.parameters()), 1.0
            )
            self.optimizer.step()

            self.encoder.eval(); self.rum.eval()
            with torch.no_grad():
                V_e = self._forward_V()
                log_p_e = self._per_hour_log_p(V_e)
                val_nll = self._nll(log_p_e, self.val_mask).item()
                cpc_val = self._cpc(log_p_e, self.val_mask)

            log.epoch.append(ep)
            # Track *main* NLL (not total loss) so train_nll stays comparable to
            # val_nll across IRM / non-IRM runs.
            log.train_nll.append(float(loss_main.item()))
            log.val_nll.append(val_nll)
            log.cpc_val.append(cpc_val)
            log.alpha.append(float(self.rum.alpha.item()))
            log.beta_mean.append(float(self.rum.beta_t_per_mode.mean().item()))
            # gamma is scalar by default; (M,) when mode_specific_gamma=True.
            # log keeps a per-epoch scalar trace — store the mean.
            log.gamma.append(float(self.rum.gamma.mean().item()))

            # IRM warmup boundary: at the first epoch where the penalty kicks
            # in (ep == warmup_epochs and lambda > 0), the loss landscape shifts
            # — val_nll is expected to rise (we're trading in-sample for OOD).
            # Reset selection state so post-warmup epochs get a fresh shot at
            # "best".
            is_warmup_boundary = (
                irm_mode
                and self.irm_warmup_epochs > 0
                and ep == self.irm_warmup_epochs
            )
            if is_warmup_boundary:
                best_metric = -float("inf") if irm_mode else float("inf")
                no_improve = 0

            # Pick metric & direction based on mode
            metric = cpc_val if irm_mode else val_nll
            improved = (metric > best_metric + 1e-4) if irm_mode \
                else (metric < best_metric - 1e-4)
            if improved:
                best_metric = metric
                no_improve = 0
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                    "rum": {k: v.detach().clone() for k, v in self.rum.state_dict().items()},
                }
            else:
                no_improve += 1

            if self.verbose:
                bm = self.rum.beta_t_per_mode.mean(dim=0)                # (M,)
                kap = self.rum.kappa                                     # (K,)
                g = self.rum.gamma
                if g.dim() == 0:
                    g_str = f"{g.item():+.3f}"
                else:
                    g_str = "[" + ",".join(f"{x:+.3f}" for x in g.tolist()) + "]"
                irm_str = ""
                if self.irm_variant != "none" and cur_lambda > 0:
                    irm_str = (f" | irm({self.irm_variant}) λ={cur_lambda:.3g} "
                               f"pen={float(penalty.item()):.3e}")
                print(
                    f"ep {ep:3d} | tnll {loss_main.item():.3f} | vnll {val_nll:.3f} | "
                    f"cpc {cpc_val:.3f} | "
                    f"β_mean[{','.join(self.mode_names)}]=["
                    + ",".join(f"{b:+.3f}" for b in bm.tolist()) + "] | "
                    f"κ=[" + ",".join(f"{k:.2f}" for k in kap.tolist()) + "] | "
                    f"γ=" + g_str + irm_str
                )
            # Early stop: ERM only. IRM trains fixed epochs (Arjovsky 2019).
            if not irm_mode and no_improve >= self.patience:
                if self.verbose:
                    print(f"early stop ep {ep}")
                break

        # ERM: load best_state by val_nll (current behavior).
        # IRM: use *last-epoch* state per Arjovsky 2019 / Krueger 2021 protocol.
        # IRM intentionally shifts representation away from ERM's in-sample
        # optimum; selecting on any in-sample criterion (val_nll or val_cpc)
        # systematically pulls the model back to the warm-up ERM checkpoint.
        if best_state is not None and not irm_mode:
            self.encoder.load_state_dict(best_state["encoder"])
            self.rum.load_state_dict(best_state["rum"])
        return self.encoder, self.rum, log

    # ---------------------------------------------------------------- predict
    def predict_OD(self, norm_stats=None) -> torch.Tensor:
        self.encoder.eval(); self.rum.eval()
        with torch.no_grad():
            V = self._forward_V(norm_stats=norm_stats)
            log_p = self._per_hour_log_p(V)
            P = log_p.exp()
            row_sum = self.observed_OD.sum(dim=2, keepdim=True)
            return (P * row_sum).cpu()
