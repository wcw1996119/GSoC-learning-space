"""Train v3 D→M nested logit with Cervero match + Shen competition.

Replaces train_dm_nested.py for the v3b milestone (lit-anchored redesign).

V specification (see methodology/02_dm_nested_design.md after lit-review revision):
  V_upper(i, j) = α·log(W_j) + γ·log(M_j) + ν·log(D_j) + δ·match_prob[i,j]
                 + λ_b(j)·IV_mode(i, j) + blend·V_GNN_j
  V_lower(i, j, m) = ASC_m + β_t_m·t_{ij,m} + θ_inc_m·income_score_i

Run:
    python experiments/paper_a/train_cervero_shen.py \
        --epochs 200 --patience 30 --seed 0 \
        --lambda-kl 1.0 --kl-warmup-epochs 15 \
        --blend-max 1.0 \
        --device cpu --verbose
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"

# v2 owns the `models_lib` namespace (encoder, dataloader, helpers).
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

# Load v3 head by absolute path
_spec = importlib.util.spec_from_file_location(
    "v3_cervero_shen_head",
    V3_ROOT / "models_lib" / "inverse_rum" / "cervero_shen_head.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CerveroShenHead = _mod.CerveroShenHead

from models_lib.inverse_rum.dual_branch_encoder import DualBranchEncoder
# v3-local: GAT-based encoder (Step 3) — load by absolute path so it doesn't
# collide with v2's already-cached models_lib.inverse_rum namespace.
_spec_gat = importlib.util.spec_from_file_location(
    "v3_dual_branch_gat_encoder",
    V3_ROOT / "models_lib" / "inverse_rum" / "dual_branch_gat_encoder.py",
)
_mod_gat = importlib.util.module_from_spec(_spec_gat)
_spec_gat.loader.exec_module(_mod_gat)
DualBranchGATEncoder = _mod_gat.DualBranchGATEncoder
from models_lib.inverse_rum.dual_branch_mixture_trainer import make_distance_aware_mode_share
from experiments.paper_a.compare_four_variants import load_data


def forward_cs(
    encoder: DualBranchEncoder,
    rum: CerveroShenHead,
    X_static: torch.Tensor,
    X_dynamic: torch.Tensor,
    edge_index: torch.Tensor,
    t_per_mode: Dict[str, torch.Tensor],
    mode_names: list,
    log_d_ij: torch.Tensor,          # (N, N) — for β_t,1 · log_d · t distance-modulated VOT
    match_prob: torch.Tensor,        # (N, N) — Cervero
    log_M_j: torch.Tensor,           # (N,) — Hansen
    log_W_j: torch.Tensor,           # (N,) — wage
    log_D_j: torch.Tensor,           # (N,) — Shen competition
    income_score_per_origin: torch.Tensor,  # (N,)
    pct_kids_per_origin: torch.Tensor,      # (N,) — % households with dependent children
    mean_cars_per_origin: torch.Tensor,     # (N,) — mean cars per household
    income_tier_props: torch.Tensor,        # (N, K) — P(tier=k | origin i), K = n_income_tiers
    pi_m_pair: torch.Tensor,
    grid_borough_idx: torch.Tensor,
    observed_OD: torch.Tensor,
    train_mask: torch.Tensor,
    soc_props_per_origin: Optional[torch.Tensor] = None,    # (N, 9)
    per_soc_demand_share_j: Optional[torch.Tensor] = None,  # (N, 9), row-sum 1
):
    _V_gnn_raw = encoder(X_static, X_dynamic, edge_index)
    # Encoder may return (T, N) [legacy per-grid] or (T, N, N) [pair-aware NN]
    pair_aware_nn = (_V_gnn_raw.dim() == 3)
    if pair_aware_nn:
        T, N, _ = _V_gnn_raw.shape
        V_gnn_jt = None                    # not used in pair-aware mode
        V_gnn_pair = _V_gnn_raw            # (T, N, N) ready
    else:
        T, N = _V_gnn_raw.shape
        V_gnn_jt = _V_gnn_raw              # (T, N) legacy
        V_gnn_pair = None
    M = rum.n_modes

    # ---- precompute per-destination λ_{b(j)} ----
    lambda_per_j = rum.lambda_for_destination(grid_borough_idx)       # (N,)
    lam_view = lambda_per_j.view(1, 1, N)                             # broadcast on j-axis

    # ---- V_lower: β_t(d) = β_t,0 + β_t,1 · log_d  (distance-modulated VOT), per mode ----
    beta_t = rum.beta_t_per_mode                                       # (M,) intercept, neg
    beta_t_slope = rum.beta_t_slope_per_mode                           # (M,) slope on log_d, neg
    asc = rum.asc_per_mode                                             # (M,)
    theta_inc = rum.theta_inc_per_mode                                 # (M,)
    theta_kids = rum.theta_kids_per_mode                               # (M,)
    theta_cars = rum.theta_cars_per_mode                               # (M,)
    log_d_T = log_d_ij.unsqueeze(0)                                    # (1, N, N) broadcast

    log_iv = None
    ce_accum = None

    # Min commute time across modes — used by tier-threshold kink + consideration filter
    t_min_per_pair = torch.stack([t_per_mode[m] for m in mode_names], dim=0).min(dim=0).values  # (T, N, N)

    # ----- Lexicographic consideration filter (Swait 2001 / Cascetta 2001) -----
    # Layer 1 (match): match_pass = sigmoid(k_match · (match(i,j) - thresh_match))
    # Layer 2 (cost):  cost_pass  = sigmoid(k_cost  · (thresh_ratio_k - cost(i,j)/budget_k))
    # log(filter_pass) added to V_dest as log-mask (j fails → V_dest → -∞).
    if rum.use_consideration_filter:
        # Layer 2: cost burden filter (always per-tier; SOC doesn't change income/budget)
        cost_per_pair_tn = (
            rum.theta_t_cost * t_min_per_pair
            + rum.theta_d_cost * log_d_ij.view(1, N, N)
        )                                                                   # (T, N, N)
        budgets = rum.cost_budget_per_tier                                  # (K_tier,)
        thresholds = rum.cost_thresh_per_tier                               # (K_tier,)
        cost_ratio = cost_per_pair_tn.unsqueeze(0) / budgets.view(-1, 1, 1, 1)
        cost_pass_per_tier = torch.sigmoid(
            rum.k_cost_filter * (thresholds.view(-1, 1, 1, 1) - cost_ratio)
        )                                                                   # (K_tier, T, N, N)
        log_cost_pass_per_tier = torch.log(cost_pass_per_tier.clamp(min=1e-9))

        # ----- L3 time gate (Stage 2 soft decision tree) -----
        # Multiplicative routing: p_pass_time = σ(k_time · (T_max[tier] - t_min(i, j)))
        # log_sigmoid numerically stable for very negative arguments (log(0) safe).
        # Merged into log_cost_pass_per_tier so per-class loop uniformly reads cost_term_c.
        if getattr(rum, "use_time_gate", False):
            T_max = rum.T_max_per_tier               # (K_tier,) > 0
            k_time = rum.k_time_gate                 # scalar > 0
            # t_min_per_pair is (T, N, N)
            time_diff = T_max.view(-1, 1, 1, 1) - t_min_per_pair.unsqueeze(0)  # (K_tier, T, N, N)
            log_pass_time = F.logsigmoid(k_time * time_diff)                   # (K_tier, T, N, N)
            log_cost_pass_per_tier = log_cost_pass_per_tier + log_pass_time

        # Layer 1: occupation match filter
        if rum.use_soc_mixture:
            if getattr(rum, "frozen_log_match_mask_per_soc", None) is not None:
                # Lit-anchored frozen mask (Stoll-Houston 2005 / B1 method).
                # Pre-computed (S, N) log-mask injected by trainer; bypass learnable τ_s.
                log_match_pass_per_soc = rum.frozen_log_match_mask_per_soc      # (S, N)
            else:
                # Per-SOC learnable: τ_s and k_s applied to demand_share[j, s].
                # log_match_pass_per_soc[s, j] = log σ(k_s · (input[s, j] - τ_s))
                # If use_z_score_match: input = (demand_share - mean_s) / std_s (z-score).
                # Else: input = raw demand_share.
                tau_per_soc = rum.match_filter_thresh_per_soc.view(rum.n_soc, 1)   # (S, 1)
                k_per_soc = rum.k_match_filter_per_soc.view(rum.n_soc, 1)          # (S, 1)
                ds_T = per_soc_demand_share_j.t()                                  # (S, N)
                if getattr(rum, "use_z_score_match", False):
                    ds_input = (
                        (ds_T - rum._ds_mean_per_soc.view(rum.n_soc, 1))
                        / rum._ds_std_per_soc.view(rum.n_soc, 1)
                    )
                else:
                    ds_input = ds_T
                match_pass_per_soc = torch.sigmoid(k_per_soc * (ds_input - tau_per_soc))
                log_match_pass_per_soc = torch.log(match_pass_per_soc.clamp(min=1e-9))  # (S, N)
            log_filter_mask_per_tier = None                                    # assembled in loop
        else:
            # Legacy: scalar τ on cosine_match[i, j], optional floor
            match_pass = torch.sigmoid(
                rum.k_match_filter * (match_prob - rum.match_filter_thresh)
            )                                                                  # (N, N)
            log_match_pass = torch.log(match_pass.clamp(min=1e-9))             # (N, N)
            log_filter_mask_per_tier = (
                log_match_pass.view(1, 1, N, N) + log_cost_pass_per_tier
            )                                                                  # (K_tier, T, N, N)
            log_match_pass_per_soc = None
    else:
        log_filter_mask_per_tier = None
        log_match_pass_per_soc = None
        log_cost_pass_per_tier = None

    for m_idx, name in enumerate(mode_names):
        t_m = t_per_mode[name]                                        # (T, N, N)
        # β_t_m(d) = β_t_m,0 + β_t_m,1 · log_d_ij  → per-pair time-disutility coeff
        beta_t_m_at_d = beta_t[m_idx] + beta_t_slope[m_idx] * log_d_T  # (1, N, N)
        # V_lower_m(i, j, t) = β_t_m(d) · t + ASC_m + Σ_a θ_a_m · attr_a_i
        agent_term = (theta_inc[m_idx] * income_score_per_origin
                      + theta_kids[m_idx] * pct_kids_per_origin
                      + theta_cars[m_idx] * mean_cars_per_origin).view(1, N, 1)
        V_lower_m = (beta_t_m_at_d * t_m
                     + asc[m_idx]
                     + agent_term)                                     # (T, N, N)
        scaled = V_lower_m / lam_view                                  # (T, N, N)
        if log_iv is None:
            log_iv = scaled
        else:
            log_iv = torch.logaddexp(log_iv, scaled)
        pi_m_ij = pi_m_pair[:, :, m_idx].unsqueeze(0)                  # (1, N, N)
        if ce_accum is None:
            ce_accum = pi_m_ij * scaled
        else:
            ce_accum = ce_accum + pi_m_ij * scaled

    IV_mode = log_iv                                                   # (T, N, N)
    ce_mode_per_ijt = IV_mode - ce_accum                               # (T, N, N)

    # ---- V_upper (Cervero 1999 multiplicative + optional tier-mixture) ----
    # Single (legacy):  γ_eff = γ + δ · match;  V = γ_eff·log_M + α·log_W + ν·log_D
    # Tier mixture: each (α_k, γ_k, ν_k, δ_k) is tier-specific (k = 0/1/2 for low/mid/high)
    #               final P(j|i) = Σ_k π_k(i) · softmax_j(V_upper_k)
    alpha_w = rum.alpha_wage          # scalar (legacy) or (K,) tensor (tier mixture)
    gamma_M = rum.gamma_M
    nu_D = rum.nu_D
    delta_m = rum.delta_match

    if rum.use_match_gate:
        k_gate = rum.gate_steepness
        tau = rum.gate_threshold
        match_signal = torch.sigmoid(k_gate * (match_prob - tau))      # (N, N) ∈ [0, 1]
    else:
        match_signal = match_prob                                       # raw linear

    # Stoll-Houston (2005) coupling: log(M · match) → effective accessible jobs.
    # When enabled, replace the weak δ_match·match·log_M interaction with the
    # strong γ-weighted log_match term: V_M = γ · (log_M + log_match).
    if getattr(rum, "use_stoll_match", False):
        log_match_signal = torch.log(match_signal.clamp(min=1e-6))      # (N, N) ∈ (-13.8, 0]
    else:
        log_match_signal = None

    # Push-pull cross term (if enabled): V += ξ · push_i · log_M_j
    # push_i = log_D_j[i] - log_M_j[i] = labor surplus at origin i
    # (high D_j at i = many workers reach i; low M_j at i = few jobs at i → push out)
    if rum.use_push_pull:
        push_origin = (log_D_j - log_M_j)                                # (N,) labor surplus at each node
        V_push_pull = (rum.xi_push_pull
                       * push_origin.view(1, N, 1)
                       * log_M_j.view(1, 1, N))                          # (1, N, N) broadcast on T
    else:
        V_push_pull = None

    # Self-loop hour-of-day boost (if enabled):
    #   V[t, i, j] += β[b(i), t]  if i == j else 0
    # Captures CBD lunch-time pulse — workers go to destinations within their own
    # borough at midday, which gravity model (γ·log_M_j alone) can't see.
    # Residual diag showed CBD self-loops have peak at midday, not evening rush.
    if rum.use_self_loop_boost:
        T_t = log_iv.shape[0]  # T
        boost = rum.self_loop_boost                                       # (n_boroughs, T)
        boost_per_node = boost[grid_borough_idx]                          # (N, T)
        boost_per_t_per_i = boost_per_node.t()                            # (T, N)
        eye = torch.eye(N, dtype=boost.dtype, device=boost.device).unsqueeze(0)  # (1, N, N)
        V_self_loop = boost_per_t_per_i.unsqueeze(-1) * eye               # (T, N, N), zero off-diagonal
    else:
        V_self_loop = None

    # Top-K busy destination boost (Option 2, finer than borough-level):
    #   V[t, i, j] += busy_dest_boost[k(j), t]   if j in top-K busy else 0
    # Captures grid-level CBD destination patterns within busy boroughs that
    # borough-level self_loop_boost cannot resolve (e.g. 816→726 under-prediction).
    if rum.n_busy_dest > 0 and rum.busy_dest_boost is not None:
        boost_kT = rum.busy_dest_boost                                    # (K, T)
        dest_to_k = rum.busy_dest_to_k_idx                                # (N,) in [-1, K-1]
        is_busy = (dest_to_k >= 0)                                        # (N,) bool
        k_clamped = dest_to_k.clamp(min=0)                                # (N,) — non-busy mapped to 0, masked later
        # Index busy_dest_boost by destination idx
        boost_per_j = boost_kT[k_clamped]                                 # (N, T)
        # Zero out non-busy destinations
        boost_per_j = boost_per_j * is_busy.float().unsqueeze(-1)         # (N, T)
        boost_per_tj = boost_per_j.t()                                    # (T, N) — per (t, dest j)
        V_busy_dest = boost_per_tj.unsqueeze(1)                           # (T, 1, N) broadcast on origin i
    else:
        V_busy_dest = None

    if rum.use_tier_mixture:
        # ---- Latent-class mixture: K = n_classes (27 if SOC×tier; else 3) ----
        # Each class c has its own complete utility-function parameter set.
        # Per-class V_rum_c is computed and immediately combined with NN + softmaxed
        # to produce a per-class log_P contribution; we accumulate via logaddexp
        # rather than stacking, to keep peak memory at ~one (T, N, N) tensor.
        K = rum.n_classes
        if rum.use_soc_mixture:
            S = rum.n_soc
            assert per_soc_demand_share_j is not None and soc_props_per_origin is not None, \
                "use_soc_mixture requires per_soc_demand_share_j and soc_props_per_origin"
            log_demand_share = torch.log(per_soc_demand_share_j.clamp(min=1e-6))  # (N, S)
            K_tier = rum.n_income_tiers
            joint = (
                income_tier_props.view(N, K_tier, 1) * soc_props_per_origin.view(N, 1, S)
            ).reshape(N, K)                                                  # (N, K)
            log_joint_per_class = torch.log(joint.t() + 1e-9)                # (K, N)
            # Stoll-Houston coupling on aggregate cosine match — same as baseline.
            # γ_M[c] couples to BOTH log_M_j AND log_cosine_match[i, j] so the
            # OD-pair-specific signal is preserved (option B+ vs option B).
            log_cosine_match = torch.log(match_signal.clamp(min=1e-6))       # (N, N)
        else:
            log_joint_per_class = torch.log(income_tier_props.t() + 1e-9)    # (K, N)

        # Precompute NN scale once (class-independent) — used both inside the
        # class loop for V_dest_c and for downstream diagnostics.
        # Pair-aware NN: V_gnn is (T, N, N) directly. Legacy: broadcast (T, N) → (T, N, N).
        if pair_aware_nn:
            V_gnn = V_gnn_pair
        else:
            V_gnn = V_gnn_jt.view(T, 1, N).expand(T, N, N)
        if rum.gnn_mode == "residual":
            w_nn = rum.gnn_residual_scale
            V_nn_scaled = (torch.zeros(T, N, N, device=V_gnn.device)
                           if w_nn is None else w_nn * V_gnn)
            _gnn_mult = None
            _gnn_g = None
            _gnn_blend = None
        elif rum.gnn_mode == "mult":
            g_mult = rum.gnn_mult_scale
            if g_mult is None:
                _gnn_mult = None
                V_nn_scaled = torch.zeros(T, N, N, device=V_gnn.device)
            else:
                _gnn_mult = 1.0 + g_mult * torch.sigmoid(V_gnn)
                V_nn_scaled = (_gnn_mult - 1.0)
            _gnn_g = None
            _gnn_blend = None
        elif rum.gnn_mode == "moe":
            _gnn_g = rum.gate_per_origin.view(1, N, 1)
            V_nn_scaled = _gnn_g * V_gnn
            _gnn_mult = None
            _gnn_blend = None
        else:  # convex
            _gnn_blend = rum.gnn_blend
            V_nn_scaled = (torch.zeros(T, N, N, device=V_gnn.device)
                           if _gnn_blend is None else _gnn_blend * V_gnn)
            _gnn_mult = None
            _gnn_g = None

        from torch.utils.checkpoint import checkpoint as _ckpt
        use_ckpt = rum.training or any(p.requires_grad for p in rum.parameters())

        # Pre-extract per-class kink params as tensors (small) so checkpoint can take them.
        # The actual (T, N, N) kink_term is computed inside the checkpoint to avoid saving 27 of them.
        if rum.use_tier_threshold:
            T_thresholds = rum.T_threshold_per_tier                          # (K,)
            beta_kinks = rum.beta_t_kink_per_tier                            # (K,)
            k_sharps = rum.k_sharpness_per_tier                              # (K,)
        # Consideration filter (Swait 2001 / Cascetta 2001): when enabled,
        # log_filter_mask_per_tier has shape (n_tiers, T, N, N) — same across SOCs.

        def _class_logp_step(V_M_c, V_other_c, log_pi_c_row,
                              T_c, beta_kink_c, k_sharp_c, has_kink_t,
                              filter_term, match_term_per_soc, cost_term_per_tier):
            # All (T, N, N) intermediates created here are NOT retained for backward.
            V_rum_c = (V_M_c + V_other_c).unsqueeze(0) + lam_view * IV_mode
            if V_push_pull is not None:
                V_rum_c = V_rum_c + V_push_pull
            if V_self_loop is not None:
                V_rum_c = V_rum_c + V_self_loop
            if V_busy_dest is not None:
                V_rum_c = V_rum_c + V_busy_dest
            if has_kink_t:
                sig = torch.sigmoid((t_min_per_pair - T_c) / k_sharp_c)
                V_rum_c = V_rum_c + beta_kink_c * t_min_per_pair * sig
            if filter_term is not None:
                V_rum_c = V_rum_c + filter_term
            elif match_term_per_soc is not None:
                # Per-SOC mode: combine match (N,) + cost (T, N, N) INSIDE checkpoint
                # so the (T, N, N) combined tensor isn't retained for backward.
                V_rum_c = (V_rum_c
                          + match_term_per_soc.view(1, 1, N)
                          + cost_term_per_tier)
            if rum.gnn_mode == "residual":
                V_dest_c = V_rum_c + V_nn_scaled
            elif rum.gnn_mode == "mult":
                V_dest_c = V_rum_c if _gnn_mult is None else V_rum_c * _gnn_mult
            elif rum.gnn_mode == "moe":
                V_dest_c = (1.0 - _gnn_g) * V_rum_c + _gnn_g * V_gnn
            else:
                V_dest_c = (V_rum_c if _gnn_blend is None
                            else (1.0 - _gnn_blend) * V_rum_c + V_nn_scaled)
            log_P_c = F.log_softmax(V_dest_c, dim=-1)
            return log_pi_c_row.view(1, N, 1) + log_P_c

        log_P_accum = None
        V_rum_dest_avg = torch.zeros(T, N, N, device=V_gnn.device)
        has_kink = rum.use_tier_threshold
        for c in range(K):
            # Structured-heterogeneity lookup:
            #   tier_idx selects tier-only params (α_W, ν_D, T, β_kink, k_sharp)
            #   soc_idx selects SOC-only params (δ_match)
            #   c selects tier×SOC params (γ_M)
            if rum.use_soc_mixture:
                tier_idx = c // S
                soc_idx = c % S
                log_dshare_for_c = log_demand_share[:, soc_idx]              # (N,)
                # Stoll coupling on cosine match + per-SOC refinement
                V_M_c = (gamma_M[c] * (log_M_j.view(1, N) + log_cosine_match)
                         + delta_m[soc_idx] * log_dshare_for_c.view(1, N))   # (N, N)
            else:
                tier_idx = c                                                  # plain tier mixture
                if log_match_signal is not None:
                    V_M_c = gamma_M[c] * (log_M_j.view(1, N) + log_match_signal)
                else:
                    gamma_eff_c = gamma_M[c] + delta_m[c] * match_signal
                    V_M_c = gamma_eff_c * log_M_j.view(1, N)
            V_other_c = (alpha_w[tier_idx] * log_W_j
                         + nu_D[tier_idx] * log_D_j).view(1, N)
            T_c = T_thresholds[tier_idx] if has_kink else log_M_j.new_zeros(())
            bk_c = beta_kinks[tier_idx] if has_kink else log_M_j.new_zeros(())
            ks_c = k_sharps[tier_idx] if has_kink else log_M_j.new_ones(())
            # Consideration filter assembly:
            #   - Per-SOC mode: pass match (N,) and cost (T, N, N) SEPARATELY so the
            #     combined (T, N, N) tensor only lives inside the checkpointed function
            #     and is not retained for backward (saves 27 × 286 MB).
            #   - Legacy mode: filter_term_c = log_filter_mask_per_tier[tier]
            if log_match_pass_per_soc is not None:
                filter_term_c = None
                match_term_c = log_match_pass_per_soc[soc_idx]          # (N,) — tiny
                cost_term_c = log_cost_pass_per_tier[tier_idx]          # (T, N, N) — 3 unique shared
            elif log_filter_mask_per_tier is not None:
                filter_term_c = log_filter_mask_per_tier[tier_idx]
                match_term_c = None
                cost_term_c = None
            else:
                filter_term_c = None
                match_term_c = None
                cost_term_c = None
            if use_ckpt:
                weighted_c = _ckpt(_class_logp_step,
                                   V_M_c, V_other_c, log_joint_per_class[c],
                                   T_c, bk_c, ks_c, has_kink,
                                   filter_term_c, match_term_c, cost_term_c,
                                   use_reentrant=False)
            else:
                weighted_c = _class_logp_step(V_M_c, V_other_c,
                                              log_joint_per_class[c],
                                              T_c, bk_c, ks_c, has_kink,
                                              filter_term_c, match_term_c, cost_term_c)
            log_P_accum = (weighted_c if log_P_accum is None
                           else torch.logaddexp(log_P_accum, weighted_c))
            # Detached diagnostic accumulator
            with torch.no_grad():
                V_rum_c_diag = (V_M_c + V_other_c).unsqueeze(0) + lam_view * IV_mode
                if V_push_pull is not None: V_rum_c_diag = V_rum_c_diag + V_push_pull
                if V_self_loop is not None: V_rum_c_diag = V_rum_c_diag + V_self_loop
                if V_busy_dest is not None: V_rum_c_diag = V_rum_c_diag + V_busy_dest
                if has_kink:
                    sig_d = torch.sigmoid((t_min_per_pair - T_c) / ks_c)
                    V_rum_c_diag = V_rum_c_diag + bk_c * t_min_per_pair * sig_d
                if filter_term_c is not None:
                    V_rum_c_diag = V_rum_c_diag + filter_term_c
                elif match_term_c is not None:
                    V_rum_c_diag = (V_rum_c_diag + match_term_c.view(1, 1, N)
                                    + cost_term_c)
                V_rum_dest_avg = V_rum_dest_avg + V_rum_c_diag / K
        log_P_D = log_P_accum
        V_rum_dest = V_rum_dest_avg
    else:
        # single-RUM path (original)
        if log_match_signal is not None:
            V_M = gamma_M * (log_M_j.view(1, N) + log_match_signal)      # (N, N)
        else:
            gamma_effective = gamma_M + delta_m * match_signal           # (N, N)
            V_M = gamma_effective * log_M_j.view(1, N)                   # (N, N)
        V_other = (alpha_w * log_W_j.view(1, 1, N)
                   + nu_D * log_D_j.view(1, 1, N))                       # (1, 1, N)
        V_rum_dest = (V_M.unsqueeze(0) + V_other + lam_view * IV_mode)   # (T, N, N)
        if V_push_pull is not None:
            V_rum_dest = V_rum_dest + V_push_pull
        if V_self_loop is not None:
            V_rum_dest = V_rum_dest + V_self_loop
        if V_busy_dest is not None:
            V_rum_dest = V_rum_dest + V_busy_dest

    # Single-RUM path: compute V_gnn + V_dest + log_P_D.
    # (Tier-mixture path already computed log_P_D and V_rum_dest_avg / V_nn_scaled inside the per-class loop.)
    if not rum.use_tier_mixture:
        if pair_aware_nn:
            V_gnn = V_gnn_pair
        else:
            V_gnn = V_gnn_jt.view(T, 1, N).expand(T, N, N)               # (T, N, N)
        if rum.gnn_mode == "residual":
            w_nn = rum.gnn_residual_scale
            V_nn_scaled = (torch.zeros(T, N, N, device=V_gnn.device)
                           if w_nn is None else w_nn * V_gnn)
            V_dest = V_rum_dest + V_nn_scaled
        elif rum.gnn_mode == "mult":
            g_mult = rum.gnn_mult_scale
            if g_mult is None:
                V_nn_scaled = torch.zeros(T, N, N, device=V_gnn.device)
                V_dest = V_rum_dest
            else:
                gate = torch.sigmoid(V_gnn)
                mult = 1.0 + g_mult * gate
                V_dest = V_rum_dest * mult
                V_nn_scaled = V_dest - V_rum_dest
        elif rum.gnn_mode == "moe":
            gate = rum.gate_per_origin
            g = gate.view(1, N, 1)
            V_dest = (1.0 - g) * V_rum_dest + g * V_gnn
            V_nn_scaled = g * V_gnn
        else:  # convex
            blend = rum.gnn_blend
            V_nn_scaled = (torch.zeros(T, N, N, device=V_gnn.device)
                           if blend is None else blend * V_gnn)
            V_dest = (V_rum_dest if blend is None
                      else (1.0 - blend) * V_rum_dest + V_nn_scaled)
        log_P_D = torch.log_softmax(V_dest, dim=-1)                      # (T, N, N)

    # ---- losses ----
    mask_f = train_mask.view(1, N, 1).float()
    flow = observed_OD * mask_f
    flow_sum = flow.sum().clamp(min=1.0)

    nll_dest = -(flow * log_P_D).sum() / flow_sum
    ce_mode = (flow * ce_mode_per_ijt).sum() / flow_sum

    # Wang TB-ResNet δ_measured = ‖V_NN‖ / (‖V_RUM‖ + ‖V_NN‖) — norm-ratio
    # Use RMS over (T, N, N) cells to keep scale comparable.
    with torch.no_grad():
        rum_rms = torch.sqrt((V_rum_dest ** 2).mean())
        nn_rms = torch.sqrt((V_nn_scaled ** 2).mean())
        delta_measured = (nn_rms / (rum_rms + nn_rms + 1e-9)).item()
        nn_norm_sq_diag = (V_nn_scaled ** 2).mean().item()

        # ----- Attribute-share decomposition -----
        # Per-attribute RMS in logit space. Variance-based share (RMS² as proxy
        # variance under independence; not exact since terms share log_M_j etc,
        # but standard ANOVA-style decomposition for interpretation).
        if rum.use_tier_mixture:
            a_avg, g_avg, n_avg, d_avg = (alpha_w.mean(), gamma_M.mean(),
                                          nu_D.mean(), delta_m.mean())
        else:
            a_avg, g_avg, n_avg, d_avg = alpha_w, gamma_M, nu_D, delta_m
        comp = {
            "gravity_gamma_logM": (g_avg * log_M_j.view(1, 1, N)).expand(T, N, N),
            "wage_alpha_logW":    (a_avg * log_W_j.view(1, 1, N)).expand(T, N, N),
            "competition_nu_logD":(n_avg * log_D_j.view(1, 1, N)).expand(T, N, N),
            "mode_choice_IV":     (lam_view * IV_mode).expand(T, N, N),
        }
        if log_match_signal is not None:
            # Stoll-Houston: match contributes via γ · log_match (same elasticity as gravity)
            comp["stoll_match_gamma_logmatch"] = (
                g_avg * log_match_signal.view(1, N, N)
            ).expand(T, N, N)
        elif rum.use_soc_mixture:
            # SOC-mixture path uses TWO match channels:
            #   (1) γ_M[c] · log cosine_match[i, j]  — OD-pair via γ_M coupling (Stoll, strong)
            #   (2) δ_match[soc(c)] · log demand_share[j, soc(c)]  — per-SOC additive (weak)
            # Track BOTH so attribute_shares show the real match contribution.
            comp["stoll_match_gamma_logcosine"] = (
                g_avg * log_cosine_match.view(1, N, N)
            ).expand(T, N, N)
            avg_log_dshare = soc_props_per_origin @ log_demand_share.t()        # (N, N)
            comp["soc_match_delta_logdshare"] = (
                d_avg * avg_log_dshare.view(1, N, N)
            ).expand(T, N, N)
        else:
            comp["occmatch_delta_M"] = (
                d_avg * match_signal.view(1, N, N) * log_M_j.view(1, 1, N)
            ).expand(T, N, N)
        if V_self_loop is not None:
            comp["self_loop_boost"] = V_self_loop
        if V_busy_dest is not None:
            comp["busy_dest_boost"] = V_busy_dest.expand(T, N, N)
        if V_push_pull is not None:
            comp["push_pull"] = V_push_pull.expand(T, N, N)
        comp["nn_residual"] = V_nn_scaled
        attribute_rms = {k: float(torch.sqrt((v ** 2).mean())) for k, v in comp.items()}
        total_var = sum(v ** 2 for v in attribute_rms.values()) + 1e-9
        attribute_shares = {k: (v ** 2) / total_var for k, v in attribute_rms.items()}
        # Destination-attractor-only share (excl. mode_choice_IV, which is a
        # logsumexp summary of mode-level time cost and dominates raw variance
        # by scale). This is the share among "what makes destination j attractive".
        dest_keys = [k for k in attribute_rms if k != "mode_choice_IV"]
        dest_var = sum(attribute_rms[k] ** 2 for k in dest_keys) + 1e-9
        attribute_shares_dest = {k: (attribute_rms[k] ** 2) / dest_var for k in dest_keys}
        # ----- Across-j variance share (TRUE destination-discriminating importance) -----
        # Softmax over destinations is invariant to any constant offset, so only the
        # variance of each component ACROSS j (within fixed t,i) determines prediction.
        # Compute var(component, dim=j) then mean over (t, i). This is the metric
        # that reflects how much each attribute actually shifts destination probabilities.
        across_j_var = {}
        for k, v in comp.items():
            # v shape (T, N, N); want var over last dim, mean over (T, N)
            across_j_var[k] = float(v.var(dim=-1, unbiased=False).mean())
        across_j_total = sum(across_j_var.values()) + 1e-9
        attribute_shares_across_j = {k: v / across_j_total for k, v in across_j_var.items()}

    # NN-norm regularization (training only): pushes ‖V_NN‖² down to enforce theory-dominance
    nn_norm_sq = (V_nn_scaled ** 2).mean()

    # Orthogonality regularizer: push V_NN ⊥ V_RUM in mean-centered logit space.
    # Encourages NN to learn signal that RUM structurally cannot express.
    # Returned as a (differentiable) cos² similarity scalar; trainer multiplies by λ_ortho.
    # V_rum_dest is always (T, N, N) — tier-mixture path now stores class-averaged
    # representative in V_rum_dest (per-class accumulated then divided by K).
    rum_flat = V_rum_dest.reshape(-1)
    nn_flat = V_nn_scaled.reshape(-1)
    rum_c = rum_flat - rum_flat.mean()
    nn_c = nn_flat - nn_flat.mean()
    ortho_cos_sq = (rum_c @ nn_c) ** 2 / (
        (rum_c.pow(2).sum() + 1e-9) * (nn_c.pow(2).sum() + 1e-9)
    )

    # FEATURE-LEVEL orthogonality: V_NN_pair (T,N,N) must be orthogonal to
    # log_cosine_match (N,N). Memory-efficient: compute on time-averaged V_NN
    # in (N,N) space rather than expanding cosine_match to (T,N,N).
    # cosine_match is time-invariant; only V_NN's t-average can correlate with it.
    if rum.use_soc_mixture:
        V_nn_t_avg = V_nn_scaled.mean(dim=0)                         # (N, N) - 12 MB
        match_flat_NN = log_cosine_match.reshape(-1)                 # (N×N,)
        nn_flat_NN = V_nn_t_avg.reshape(-1)                          # (N×N,)
        match_c2 = match_flat_NN - match_flat_NN.mean()
        nn_c2 = nn_flat_NN - nn_flat_NN.mean()
        match_ortho_cos_sq = (match_c2 @ nn_c2) ** 2 / (
            (match_c2.pow(2).sum() + 1e-9) * (nn_c2.pow(2).sum() + 1e-9)
        )
    else:
        match_ortho_cos_sq = torch.tensor(0.0, device=V_rum_dest.device)

    # IV-balance regularizer (DIFFERENTIABLE — uses tensors built in this fwd):
    # Push IV_mode's share of across-j destination-discriminating variance toward target.
    # If IV dominates (current ~97%), force it down so gravity/wage/match must carry the
    # destination signal — attribute shares become more balanced.
    iv_var = (lam_view * IV_mode).var(dim=-1, unbiased=False).mean()
    if rum.use_tier_mixture:
        if rum.use_soc_mixture:
            # Per-SOC log_demand_share enters per class; aggregate elasticity for
            # this regularizer = γ_M · log_M_j (drop per-SOC term, it's class-specific)
            v_gravity = (gamma_M.mean() * log_M_j.view(1, N)).expand(T, N, N)
        elif log_match_signal is not None:
            v_gravity = (gamma_M.mean() * (log_M_j.view(1, N) + log_match_signal)).expand(T, N, N)
        else:
            gamma_eff_avg = gamma_M.mean() + delta_m.mean() * match_signal
            v_gravity = (gamma_eff_avg * log_M_j.view(1, N)).expand(T, N, N)
        v_wage    = (alpha_w.mean() * log_W_j.view(1, 1, N)).expand(T, N, N)
        v_comp    = (nu_D.mean()    * log_D_j.view(1, 1, N)).expand(T, N, N)
    else:
        if log_match_signal is not None:
            v_gravity = (gamma_M * (log_M_j.view(1, N) + log_match_signal)).expand(T, N, N)
        else:
            v_gravity = ((gamma_M + delta_m * match_signal) * log_M_j.view(1, N)).expand(T, N, N)
        v_wage    = (alpha_w * log_W_j.view(1, 1, N)).expand(T, N, N)
        v_comp    = (nu_D    * log_D_j.view(1, 1, N)).expand(T, N, N)
    rum_dest_var = (v_gravity.var(dim=-1, unbiased=False).mean()
                    + v_wage.var(dim=-1, unbiased=False).mean()
                    + v_comp.var(dim=-1, unbiased=False).mean())
    iv_share_across_j = iv_var / (iv_var + rum_dest_var + 1e-9)

    # ---- diagnostic ----
    with torch.no_grad():
        # tier-mixture: alpha/gamma/nu/delta are (K,) tensors → store as lists
        def _diag_val(t):
            return t.tolist() if t.dim() > 0 else float(t)
        diagnostic = {
            "use_tier_mixture": rum.use_tier_mixture,
            "use_soc_mixture": rum.use_soc_mixture,
            "alpha_wage": _diag_val(rum.alpha_wage),
            "gamma_M": _diag_val(rum.gamma_M),
            "nu_D": _diag_val(rum.nu_D),
            "delta_match": _diag_val(rum.delta_match),
            "beta_t_mean": float(rum.beta_t_per_mode.mean()),
            "beta_t_slope_mean": float(rum.beta_t_slope_per_mode.mean()),
            "T_threshold_per_tier": rum.T_threshold_per_tier.detach().cpu().tolist() if rum.T_threshold_per_tier is not None else None,
            "beta_t_kink_per_tier": rum.beta_t_kink_per_tier.detach().cpu().tolist() if rum.beta_t_kink_per_tier is not None else None,
            "k_sharpness_per_tier": rum.k_sharpness_per_tier.detach().cpu().tolist() if rum.k_sharpness_per_tier is not None else None,
            "match_filter_thresh": float(rum.match_filter_thresh) if rum.match_filter_thresh is not None else None,
            "k_match_filter": float(rum.k_match_filter) if rum.k_match_filter is not None else None,
            "match_filter_thresh_per_soc": (rum.match_filter_thresh_per_soc.detach().cpu().tolist()
                                            if rum.match_filter_thresh_per_soc is not None else None),
            "k_match_filter_per_soc": (rum.k_match_filter_per_soc.detach().cpu().tolist()
                                       if rum.k_match_filter_per_soc is not None else None),
            "match_thresh_floor": getattr(rum, "match_thresh_floor", 0.0),
            "theta_t_cost": float(rum.theta_t_cost) if rum.theta_t_cost is not None else None,
            "theta_d_cost": float(rum.theta_d_cost) if rum.theta_d_cost is not None else None,
            "cost_budget_per_tier": rum.cost_budget_per_tier.detach().cpu().tolist() if rum.cost_budget_per_tier is not None else None,
            "cost_thresh_per_tier": rum.cost_thresh_per_tier.detach().cpu().tolist() if rum.cost_thresh_per_tier is not None else None,
            "k_cost_filter": float(rum.k_cost_filter) if rum.k_cost_filter is not None else None,
            "asc_per_mode": rum.asc_per_mode.detach().cpu().tolist(),
            "theta_inc_per_mode": rum.theta_inc_per_mode.detach().cpu().tolist(),
            "theta_kids_per_mode": rum.theta_kids_per_mode.detach().cpu().tolist(),
            "theta_cars_per_mode": rum.theta_cars_per_mode.detach().cpu().tolist(),
            "lambda_b_mean": float(rum.lambda_per_borough.mean()),
            "lambda_b_std": float(rum.lambda_per_borough.std()),
            "lambda_b_min": float(rum.lambda_per_borough.min()),
            "lambda_b_max": float(rum.lambda_per_borough.max()),
            "blend": float(rum.gnn_blend) if rum.gnn_blend is not None else None,
            "gnn_residual_scale": float(rum.gnn_residual_scale) if rum.gnn_residual_scale is not None else None,
            "gnn_mult_scale": float(rum.gnn_mult_scale) if rum.gnn_mult_scale is not None else None,
            "gate_per_origin_mean": float(rum.gate_per_origin.mean()) if rum.gate_per_origin is not None else None,
            "gate_per_origin_std": float(rum.gate_per_origin.std()) if rum.gate_per_origin is not None else None,
            "gate_per_origin": rum.gate_per_origin.detach().cpu().tolist() if rum.gate_per_origin is not None else None,
            "gnn_mode": rum.gnn_mode,
            "delta_measured": delta_measured,
            "rum_rms": float(rum_rms),
            "nn_rms": float(nn_rms),
            "use_push_pull": rum.use_push_pull,
            "xi_push_pull": float(rum.xi_push_pull) if rum.xi_push_pull is not None else None,
            "attribute_rms": attribute_rms,
            "attribute_shares": attribute_shares,
            "attribute_shares_dest": attribute_shares_dest,
            "attribute_across_j_var": across_j_var,
            "attribute_shares_across_j": attribute_shares_across_j,
            "ortho_cos_sq": float(ortho_cos_sq),
            "match_ortho_cos_sq": float(match_ortho_cos_sq),
        }
        if rum.use_soc_mixture and per_soc_demand_share_j is not None:
            ds = per_soc_demand_share_j                                 # (N, S)
            # Range per SOC, swing in log-units (= γ-weighted contribution to V_dest)
            log_ds = torch.log(ds.clamp(min=1e-6))                      # (N, S)
            log_swing_per_soc = (log_ds.max(dim=0).values - log_ds.min(dim=0).values).tolist()  # (S,)
            gamma_weighted_swing = [(float(rum.gamma_M.mean()) * v) for v in log_swing_per_soc]
            diagnostic["per_soc_demand_share_per_soc_p05_p95"] = [
                [float(torch.quantile(ds[:, s], 0.05)),
                 float(torch.quantile(ds[:, s], 0.95))] for s in range(rum.n_soc)
            ]
            diagnostic["per_soc_log_demand_swing"] = log_swing_per_soc
            diagnostic["per_soc_gamma_weighted_swing"] = gamma_weighted_swing

    return {
        "log_P_D": log_P_D,
        "nll_dest": nll_dest,
        "ce_mode": ce_mode,
        "nn_norm_sq": nn_norm_sq,
        "ortho_cos_sq": ortho_cos_sq,
        "match_ortho_cos_sq": match_ortho_cos_sq,
        "iv_share_across_j": iv_share_across_j,
        "diagnostic": diagnostic,
    }


def cpc(log_P: torch.Tensor, observed_OD: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        P = log_P.exp()
        row_sum = observed_OD.sum(dim=2, keepdim=True)
        pred = P * row_sum
        num = 2.0 * torch.minimum(pred[:, mask], observed_OD[:, mask]).sum()
        den = (pred[:, mask].sum() + observed_OD[:, mask].sum()).clamp(min=1.0)
        return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blend-max", type=float, default=1.0)
    ap.add_argument("--gnn-mode", choices=["convex", "residual", "mult", "moe"], default="convex",
                    help="convex: V=(1-δ)V_RUM+δ V_NN (legacy v3b/c). "
                         "residual: V=V_RUM+w·V_NN (Wang TB-ResNet original). "
                         "mult: V=V_RUM·(1+γ·σ(V_NN)) — NN modulates RUM amplitude per (i,j,t). "
                         "moe: V=(1-g_i)·V_RUM+g_i·V_NN with g_i ∈[0,1] per-origin learnable gate.")
    ap.add_argument("--gnn-residual-scale-init", type=float, default=0.1)
    ap.add_argument("--lambda-nn-norm", type=float, default=0.0,
                    help="L2 penalty on ‖V_NN‖² to enforce Wang Path A (δ_measured < 0.30)")
    ap.add_argument("--lambda-ortho", type=float, default=0.0,
                    help="Aggregate orthogonality penalty: cos²(V_NN, V_RUM aggregate). "
                         "Encourages NN to learn signal RUM structurally cannot express. "
                         "Typical range 0.5-5.0.")
    ap.add_argument("--lambda-match-ortho", type=float, default=0.0,
                    help="Feature-level orthogonality: cos²(V_NN, log_cosine_match expanded). "
                         "Forces NN to NOT learn the cosine-match pattern, isolating match's "
                         "contribution through the RUM channel only (Bhat-style identification). "
                         "Typical 10-50 if you want match cleanly identified.")
    ap.add_argument("--lambda-iv-balance", type=float, default=0.0,
                    help="IV-magnitude balance: push IV_mode's across-j variance share "
                         "toward --target-iv-share. Forces gravity/wage/match to carry "
                         "destination signal — produces interpretable attribute portfolio.")
    ap.add_argument("--target-iv-share", type=float, default=0.5,
                    help="Target across-j variance share for IV_mode (default 0.5 = "
                         "IV and RUM-destination attractors carry equal destination signal). "
                         "Used with --lambda-iv-balance > 0.")
    ap.add_argument("--use-match-gate", action="store_true",
                    help="Decision-tree style sigmoid gate on Cervero match: "
                         "γ_eff = γ + δ · sigmoid(k(match - τ)). "
                         "k learns sharpness, τ learns threshold.")
    ap.add_argument("--gate-steepness-init", type=float, default=5.0,
                    help="Initial k (gate sharpness); softplus ≥0. Default 5.0 = moderately sharp.")
    ap.add_argument("--gate-threshold-init", type=float, default=0.13,
                    help="Initial τ (gate threshold in raw match_prob scale). Default 0.13 = data mean.")
    # Wang TB-ResNet sequential training procedure
    ap.add_argument("--train-mode", choices=["joint", "sequential"], default="joint",
                    help="joint: train all params together (legacy); "
                         "sequential: Wang's procedure — Phase 1 trains RUM only, "
                         "Phase 2 freezes RUM and trains GNN with fixed --fixed-blend δ.")
    ap.add_argument("--fixed-blend", type=float, default=None,
                    help="Override learnable blend with fixed value (Wang δ as hyperparameter).")
    ap.add_argument("--epochs-phase1", type=int, default=200,
                    help="Sequential mode: epochs for Phase 1 (RUM only). "
                         "Also used for Stage 2 (RUM-again) when --n-stages 4.")
    ap.add_argument("--n-stages", type=int, default=2, choices=[2, 4],
                    help="Sequential boosting: 2 = Wang sequential (RUM → NN); "
                         "4 = boosting (RUM → NN → RUM → NN). "
                         "Stage 2 freezes encoder + residual_scale (keeps NN value), "
                         "lets RUM retrain on top of NN baseline (identifiability test).")
    ap.add_argument("--rum-checkpoint", type=str, default=None,
                    help="Sequential mode: load Phase 1 RUM checkpoint instead of retraining.")
    ap.add_argument("--use-tier-threshold", action="store_true",
                    help="Tier-specific commute-time threshold (Bhat 1995): "
                         "V_lower += β_kink_k · max(0, t_min - T_k) per income tier. "
                         "Requires --use-tier-mixture. "
                         "Expects T_low ~25min, T_mid ~40min, T_high ~55min after training.")
    ap.add_argument("--use-consideration-filter", action="store_true",
                    help="Lexicographic two-layer filter (Swait 2001 / Cascetta 2001): "
                         "Layer 1 occupation match + Layer 2 cost burden / income budget. "
                         "Sigmoid soft masks, log(pass) added to V_dest as log-mask. "
                         "Requires --use-tier-mixture.")
    ap.add_argument("--use-stoll-match", action="store_true",
                    help="Stoll-Houston (2005) 'effective jobs' coupling: replaces "
                         "δ_match·match·log_M with γ·log(match) so occupational match "
                         "has the same elasticity as gravity. Forces match to be a "
                         "central destination-utility factor instead of a marginal term.")
    ap.add_argument("--match-thresh-floor", type=float, default=0.0,
                    help="Lower bound on match filter threshold. Default 0 = model "
                         "learns freely (typically learns ≈0, no filtering). Set 0.30 "
                         "to force destinations with match<0.30 to be filtered.")
    ap.add_argument("--use-soc-mixture", action="store_true",
                    help="Per-SOC individual-level match (ABM-friendly). Replaces aggregate "
                         "cosine-match in V_dest with mixture over 9 SOC sub-populations, "
                         "each seeing its own demand_share[j, k] at destinations. "
                         "Requires --use-tier-mixture; incompatible with --use-stoll-match, "
                         "--use-match-gate, and --use-consideration-filter (in first MVP).")
    ap.add_argument("--k-match-init", type=float, default=10.0,
                    help="Initial sharpness of match filter sigmoid. Larger = sharper "
                         "cutoff. Use ≥20 for near-hard cutoff behavior.")
    ap.add_argument("--use-z-score-match", action="store_true",
                    help="z-score normalize demand_share[:, s] before the L1 sigmoid gate. "
                         "τ_z[s] lives in z-space, automatic per-SOC relative scaling. "
                         "Solves the 'operators in CBD' problem (z=-1.7 cuts CBD even though "
                         "demand_share=0.03 isn't 0). Requires --use-soc-mixture and "
                         "--use-consideration-filter.")
    ap.add_argument("--z-tau-floor", type=float, default=-0.5,
                    help="Lower bound for τ_z in z-score match mode. Default -0.5 allows "
                         "model to learn 'slightly below mean still passes' but cannot collapse "
                         "to 'all pass'. -1.0 = looser, 0.0 = strict above-mean.")
    ap.add_argument("--use-time-gate", action="store_true",
                    help="Stage 2 L3 commute-time hard gate: log σ(k_time · (T_max[tier] - t_min(i,j))) "
                         "added to V_dest as multiplicative routing. Upgrades the soft Bhat 1995 "
                         "kink in V_lower to an explicit lexicographic gate layer. "
                         "Requires --use-tier-mixture --use-consideration-filter.")
    ap.add_argument("--T-max-low",  type=float, default=60.0,
                    help="L3 time gate T_max (min) for low income tier. Default 60.")
    ap.add_argument("--T-max-mid",  type=float, default=90.0,
                    help="L3 time gate T_max (min) for mid income tier. Default 90.")
    ap.add_argument("--T-max-high", type=float, default=120.0,
                    help="L3 time gate T_max (min) for high income tier. Default 120.")
    ap.add_argument("--k-time-gate-init", type=float, default=0.2,
                    help="Initial sharpness of L3 time gate. Default 0.2 (≈ per-minute resolution).")
    ap.add_argument("--k-time-gate-min", type=float, default=0.1,
                    help="Lower bound on L3 time gate sharpness. Default 0.1.")
    ap.add_argument("--k-match-min-per-soc", type=float, default=0.1,
                    help="Lower bound on per-SOC sharpness k_s. Default 0.1 = no bound "
                         "(back-compat). Set ≥ 20 for soft-lexicographic: forces sharp "
                         "sigmoid filter regardless of optimizer preference. Combined with "
                         "--tau-match-floor-mult, gives data-driven thresholds inside a "
                         "structural prior (Aboutaleb 2021 EBA-style).")
    ap.add_argument("--tau-match-floor-mult", type=float, default=0.0,
                    help="τ_s ≥ mult × mean(demand_share[:, s]). Default 0 = no floor "
                         "(back-compat). 0.5 = τ at least half the SOC's all-city mean. "
                         "Use with --k-match-min-per-soc ≥ 20 for soft lexicographic.")
    ap.add_argument("--use-frozen-match-mask", action="store_true",
                    help="Stoll-Houston 2005 lit-anchored hard match-set. For each SOC s, "
                         "J_s = {j : demand_share[j,s] > mean_s · multiplier}; bypasses "
                         "the learnable per-SOC τ_s (which collapses to ~0 on aggregate OD). "
                         "Requires --use-soc-mixture and --use-consideration-filter.")
    ap.add_argument("--match-mean-mult", type=float, default=1.0,
                    help="τ_s = mean(demand_share[:,s]) · multiplier for frozen mask. "
                         "1.0 = above-mean grids (Stoll-Houston). 1.2 = stricter (504 orphans). "
                         "0.8 = looser (90%% pass).")
    ap.add_argument("--match-mask-log-penalty", type=float, default=-1e9,
                    help="log-penalty for grids OUTSIDE J_s. -1e9 = hard exclude (prob = 0). "
                         "-5 = soft penalty (~0.7%% relative weight). Tune for ABM realism.")
    ap.add_argument("--use-tier-mixture", action="store_true",
                    help="Income-tier latent class: each (α, γ, ν, δ_match) tier-specific × 3 tier. "
                         "P(j|i) = Σ_k π(tier=k|i) · softmax(V_upper_k).")
    ap.add_argument("--n-income-tiers", type=int, default=3,
                    help="Number of income tiers for latent class mixture (default 3).")
    ap.add_argument("--tier-init-scale", type=float, default=0.0,
                    help="Tier mixture init perturbation. 0=symmetric (collapse-prone), "
                         "0.5 = tier k init = base · (1 + 0.5·k). Breaks symmetry so tiers can differentiate.")
    ap.add_argument("--use-weather", action="store_true",
                    help="Append weather (24,1725,6) features to X_dynamic (z-scored). "
                         "Lets GNN see hourly weather signal for richer dynamic representation.")
    ap.add_argument("--use-push-pull", action="store_true",
                    help="Add push-pull cross term: V += ξ · (log_D_i - log_M_i) · log_M_j. "
                         "Origin labor surplus × destination jobs gravity interaction.")
    ap.add_argument("--use-self-loop-boost", action="store_true",
                    help="Add per-(origin borough, hour) bias to V[i,i,t] self-loop. "
                         "Captures CBD lunch-pulse where intra-borough commute peaks at midday "
                         "instead of evening rush (residual diag finding).")
    ap.add_argument("--self-loop-l2", type=float, default=1e-3,
                    help="L2 regularization on self-loop boost params (33×24=792). "
                         "Larger = more shrinkage to zero. Default 1e-3 (modest).")
    ap.add_argument("--n-busy-dest", type=int, default=0,
                    help="K > 0 enables top-K busy destination hour boost (grid-level, "
                         "finer than borough self-loop). Captures CBD grid-specific "
                         "attraction patterns. Default 0 (off). Recommended K=50.")
    ap.add_argument("--busy-dest-l2", type=float, default=1e-3,
                    help="L2 regularization on busy_dest_boost params (K×24).")
    ap.add_argument("--use-dual-pair-encoder", action="store_true",
                    help="Full ST-GNN + OD-pair bilinear: GraphSAGE static + per-hour "
                         "GraphSAGE + GRU + Multi-Scale TCN + Gated Fusion + Origin/Dest "
                         "MLPs + bilinear → V_NN (T, N, N). Retains neighbors + temporal "
                         "+ OD-pair signals. Heavier than --use-pair-nn but is the proper "
                         "spatio-temporal NN, not a substitute. Mutually exclusive with "
                         "--use-pair-nn and --use-gat.")
    ap.add_argument("--use-pair-nn", action="store_true",
                    help="Replace per-grid NN (T, N) with pair-aware NN (T, N, N). "
                         "Lets V_NN learn OD-pair specific patterns (e.g. unusually "
                         "high commute on a given Barking→Canary Wharf pair) that the "
                         "per-grid NN cannot express. Strongly recommended when "
                         "use_soc_mixture is on and γ_M is gradient-starved.")
    ap.add_argument("--pair-rank", type=int, default=8,
                    help="Bilinear rank for PairResidualNN (default 8).")
    ap.add_argument("--pair-hidden", type=int, default=32,
                    help="MLP hidden dim inside PairResidualNN (default 32).")
    ap.add_argument("--use-gat", action="store_true",
                    help="Replace GraphSAGE with GAT (multi-head attention) in encoder. "
                         "Step 3 of 3-step GNN improvement plan.")
    ap.add_argument("--gat-heads", type=int, default=4,
                    help="Number of attention heads for GAT layers (default 4).")
    ap.add_argument("--dump-residual-diag", action="store_true",
                    help="After final eval, compute per-hour / per-borough / per-distance / "
                         "per-flow-bin CPC breakdowns + top-30 over/under-predicted cells. "
                         "Saved into result JSON to find data ceiling shape.")
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--kl-warmup-epochs", type=int, default=15)
    ap.add_argument("--lr-theta", type=float, default=1e-3)
    ap.add_argument("--lr-rum", type=float, default=1e-2)
    ap.add_argument("--lambda-init", type=float, default=0.95)
    ap.add_argument("--lambda-eps-min", type=float, default=0.05)
    ap.add_argument("--aux-path", default="data/processed/paperA_v3_aux_cervero.npz")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="evaluation_outputs/paper_a/cervero_shen_smoke.json")
    args = ap.parse_args()

    # ---- load v2 data (encoder inputs, OD flows, t_ij, etc.) ----
    print(f"Loading v2 data ...")
    t_load = time.time()
    d = load_data()
    coords = torch.from_numpy(
        np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz",
                allow_pickle=True)["coords_bng"].astype(np.float32)
    )
    dist_km = (torch.linalg.norm(coords.unsqueeze(0) - coords.unsqueeze(1), dim=-1)
               / 1000.0).clamp(min=0.1)
    pair_mode_share = make_distance_aware_mode_share(
        d["mode_share"], dist_km, ["car", "transit", "walk"], walk_threshold_km=5.0,
    )
    print(f"  loaded in {time.time()-t_load:.1f}s  N={d['N']} T={d['T']}")

    # ---- load v3 cervero aux (use z-scored features for fair coefficient comparison) ----
    aux = np.load(V3_ROOT / args.aux_path)
    # Prefer z-scored versions (mean 0 std 1) — coefficients = per-stddev effect.
    # Fall back to raw if not present (older aux files).
    use_z = all(k in aux.files for k in ("log_W_z", "log_M_z", "log_D_z", "match_z", "income_z"))
    if use_z:
        # match_prob: keep RAW [0, ~0.4] for multiplicative Cervero
        # (γ_effective = γ + δ · match_raw in V_upper)
        match_prob = torch.from_numpy(aux["match_prob"]).float()
        log_M_j = torch.from_numpy(aux["log_M_z"]).float()
        log_W_j = torch.from_numpy(aux["log_W_z"]).float()
        log_D_j = torch.from_numpy(aux["log_D_z"]).float()
        income_score = torch.from_numpy(aux["income_z"]).float()
        print(f"v3 aux loaded: log_*_z (z-scored) for additive terms, match_prob RAW for multiplicative")
    else:
        match_prob = torch.from_numpy(aux["match_prob"]).float()
        log_M_j = torch.from_numpy(aux["log_M_j"]).float()
        log_W_j = torch.from_numpy(aux["log_W_j"]).float()
        log_D_j = torch.from_numpy(aux["log_D_j"]).float()
        income_score = torch.from_numpy(aux["income_score_per_origin"]).float()
        print(f"v3 aux loaded (RAW features, no z-score): coefficients NOT directly comparable")

    # Agent attributes (z-scored if present, else zero-filled placeholder)
    if "pct_with_kids_z" in aux.files:
        pct_kids = torch.from_numpy(aux["pct_with_kids_z"]).float()
        mean_cars = torch.from_numpy(aux["mean_cars_z"]).float()
        print(f"  agent attributes loaded: pct_with_kids_z, mean_cars_z (z-scored)")
    else:
        N_grid = match_prob.shape[0]
        pct_kids = torch.zeros(N_grid).float()
        mean_cars = torch.zeros(N_grid).float()
        print(f"  WARNING: pct_with_kids / mean_cars not in aux, using zero placeholders")

    # Income-tier mixture: P(tier=k | origin i)
    if "income_tier_props" in aux.files:
        income_tier_props = torch.from_numpy(aux["income_tier_props"]).float()
        print(f"  income_tier_props loaded: shape {tuple(income_tier_props.shape)} "
              f"(tier-marginal: {income_tier_props.mean(dim=0).tolist()})")
    else:
        N_grid = match_prob.shape[0]
        income_tier_props = torch.full((N_grid, args.n_income_tiers), 1.0 / args.n_income_tiers)
        print(f"  WARNING: income_tier_props not in aux, using uniform")

    # SOC mixture: per-SOC demand share at each destination + origin SOC composition.
    # Required by --use-soc-mixture; loaded unconditionally so diagnostics can run.
    if "soc_props" in aux.files and "grid_industry_prop" in aux.files and "epsilon" in aux.files:
        # Load v3's occupation_match via absolute path (avoid v2 shadowing on sys.path).
        _om_spec = importlib.util.spec_from_file_location(
            "v3_occ_match", V3_ROOT / "models_lib" / "occupation_match.py"
        )
        _om_mod = importlib.util.module_from_spec(_om_spec)
        _om_spec.loader.exec_module(_om_mod)
        _per_soc_demand_share = _om_mod.per_soc_demand_share
        soc_props_np = aux["soc_props"]                  # (N, 9)
        grid_ind_np = aux["grid_industry_prop"]          # (N, 8)
        eps_np = aux["epsilon"]                          # (9, 8)
        demand_share_np = _per_soc_demand_share(soc_props_np, eps_np, grid_ind_np)   # (N, 9)
        soc_props_per_origin = torch.from_numpy(soc_props_np).float()
        per_soc_demand_share_j = torch.from_numpy(demand_share_np).float()
        print(f"  per-SOC: soc_props {tuple(soc_props_per_origin.shape)}, "
              f"demand_share_j {tuple(per_soc_demand_share_j.shape)}; "
              f"demand_share range [{per_soc_demand_share_j.min():.3f}, {per_soc_demand_share_j.max():.3f}], "
              f"mean per SOC = {per_soc_demand_share_j.mean(0).tolist()}")
    else:
        soc_props_per_origin = None
        per_soc_demand_share_j = None
        if args.use_soc_mixture:
            raise RuntimeError("--use-soc-mixture requires soc_props + grid_industry_prop + epsilon in aux")

    grid_borough_idx = d["grid_borough_idx"].long()
    n_boroughs = int(grid_borough_idx.max().item()) + 1

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N, T = d["N"], d["T"]
    M = pair_mode_share.shape[2]
    mode_names = ["car", "transit", "walk"]

    X_static = d["X_static"].to(device).float()
    X_dynamic = d["X_dynamic"].to(device).float()
    edge_index = d["edge_index"].to(device)
    observed_OD = d["F_ij_t"].to(device).float()
    train_mask = d["train_mask"].to(device).bool()
    val_mask = d["val_mask"].to(device).bool()
    log_d_ij = d["log_d"].to(device).float()

    # Optional: append weather (24, 1725, 6) — z-scored — to X_dynamic
    if args.use_weather:
        w_path = V2_ROOT / "data" / "processed" / "london_hourly_weather.npz"
        if w_path.exists():
            weather_raw = np.load(w_path)["weather_TNF"].astype(np.float32)   # (T, N, 6)
            # z-score per feature across (T, N)
            w_mean = weather_raw.mean(axis=(0, 1), keepdims=True)
            w_std = weather_raw.std(axis=(0, 1), keepdims=True) + 1e-9
            weather_z = (weather_raw - w_mean) / w_std
            print(f"  weather loaded: shape={weather_raw.shape}, z-scored")
            print(f"    feature std spread (z): {weather_z.std(axis=(0,1)).tolist()}")
            X_dynamic = torch.cat(
                [X_dynamic, torch.from_numpy(weather_z).to(device).float()],
                dim=-1,
            )
            print(f"  X_dynamic now {tuple(X_dynamic.shape)} (was 25-dim, +6 weather = 31-dim)")
        else:
            print(f"  WARNING: --use-weather but {w_path} not found")
    match_prob = match_prob.to(device)
    log_M_j = log_M_j.to(device)
    log_W_j = log_W_j.to(device)
    log_D_j = log_D_j.to(device)
    income_score = income_score.to(device)
    pct_kids = pct_kids.to(device)
    mean_cars = mean_cars.to(device)
    income_tier_props = income_tier_props.to(device)
    if soc_props_per_origin is not None:
        soc_props_per_origin = soc_props_per_origin.to(device)
        per_soc_demand_share_j = per_soc_demand_share_j.to(device)
    pi_m_pair = pair_mode_share.to(device).float()
    grid_borough_idx = grid_borough_idx.to(device)

    t_per_mode = {}
    for m in mode_names:
        key = "t_" + m
        tens = d[key].to(device).float()
        if tens.dim() == 2:
            tens = tens.unsqueeze(0).expand(T, N, N).contiguous()
        t_per_mode[m] = tens

    if args.use_dual_pair_encoder:
        # Full ST-GNN + OD-pair bilinear: neighbors (GraphSAGE) + temporal
        # (GRU + Multi-Scale TCN) + OD-pair (bilinear). Three signals retained.
        _dpe_spec = importlib.util.spec_from_file_location(
            "v3_dual_pair_encoder",
            V3_ROOT / "models_lib" / "inverse_rum" / "dual_branch_pair_encoder.py",
        )
        _dpe_mod = importlib.util.module_from_spec(_dpe_spec)
        _dpe_spec.loader.exec_module(_dpe_mod)
        encoder = _dpe_mod.DualBranchPairEncoder(
            static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
            hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
            pair_rank=args.pair_rank, pair_hidden=args.pair_hidden,
        ).to(device)
        print(f"        encoder: DualBranchPairEncoder "
              f"(GraphSAGE+GRU+TCN+bilinear, pair_rank={args.pair_rank}, "
              f"pair_hidden={args.pair_hidden}) → V_NN shape (T, N, N)")
    elif args.use_pair_nn:
        # Pair-aware NN: outputs (T, N, N) directly — origin × destination bilinear
        _pair_spec = importlib.util.spec_from_file_location(
            "v3_pair_nn", V3_ROOT / "models_lib" / "inverse_rum" / "pair_residual_nn.py"
        )
        _pair_mod = importlib.util.module_from_spec(_pair_spec)
        _pair_spec.loader.exec_module(_pair_mod)
        encoder = _pair_mod.PairResidualNN(
            static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
            hidden_dim=args.pair_hidden, pair_rank=args.pair_rank,
        ).to(device)
        print(f"        encoder: PairResidualNN (pair_rank={args.pair_rank}, "
              f"hidden={args.pair_hidden}) → V_NN shape (T, N, N)")
    elif args.use_gat:
        encoder = DualBranchGATEncoder(
            static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
            hidden_dim=32, gru_hidden=32, n_gat_layers=2,
            gat_heads=args.gat_heads, tcn_kernels=(3, 5, 7),
        ).to(device)
        print(f"        encoder: DualBranchGATEncoder (heads={args.gat_heads})")
    else:
        encoder = DualBranchEncoder(
            static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
            hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
        ).to(device)
    rum = CerveroShenHead(
        n_modes=M, n_boroughs=n_boroughs, n_tiers=args.n_income_tiers,
        lambda_init=args.lambda_init, lambda_eps_min=args.lambda_eps_min,
        use_gnn_blend=True, gnn_blend_init=0.5, blend_max=args.blend_max,
        gnn_mode=args.gnn_mode,
        gnn_residual_scale_init=args.gnn_residual_scale_init,
        use_match_gate=args.use_match_gate,
        gate_steepness_init=args.gate_steepness_init,
        gate_threshold_init=args.gate_threshold_init,
        use_tier_mixture=args.use_tier_mixture,
        tier_init_scale=args.tier_init_scale,
        use_push_pull=args.use_push_pull,
        use_self_loop_boost=args.use_self_loop_boost,
        n_hours=24,
        n_busy_dest=args.n_busy_dest,
        n_origins=N,
        use_tier_threshold=args.use_tier_threshold,
        use_consideration_filter=args.use_consideration_filter,
        use_stoll_match=args.use_stoll_match,
        match_thresh_floor=args.match_thresh_floor,
        k_match_init=args.k_match_init,
        use_soc_mixture=args.use_soc_mixture,
        n_soc=9,
        use_z_score_match=args.use_z_score_match,
        z_tau_floor=args.z_tau_floor,
        k_match_min_per_soc=args.k_match_min_per_soc,
        use_time_gate=args.use_time_gate,
        T_max_per_tier_init=(args.T_max_low, args.T_max_mid, args.T_max_high),
        k_time_gate_init=args.k_time_gate_init,
        k_time_gate_min=args.k_time_gate_min,
        use_frozen_match_mask=args.use_frozen_match_mask,
    ).to(device)

    # z-score normalization: inject per-SOC demand_share mean and std
    if args.use_z_score_match:
        assert per_soc_demand_share_j is not None
        with torch.no_grad():
            mean_per_soc = per_soc_demand_share_j.mean(dim=0)              # (S,)
            std_per_soc = per_soc_demand_share_j.std(dim=0).clamp(min=1e-6)  # (S,)
        rum.set_demand_share_stats(mean_per_soc, std_per_soc)
        SOC = ["mgr","prof","assoc","admin","trades","care","sales","oper","elem"]
        print(f"        z-score match: floor τ_z = {args.z_tau_floor}, "
              f"k_min = {args.k_match_min_per_soc}")
        for s in range(rum.n_soc):
            print(f"          {SOC[s]:6s}  mean={float(mean_per_soc[s]):.4f}  std={float(std_per_soc[s]):.4f}")

    # Soft-lexicographic floor: τ_s ≥ mult × mean(demand_share[:, s])
    if args.tau_match_floor_mult > 0 and args.use_soc_mixture and args.use_consideration_filter:
        assert per_soc_demand_share_j is not None
        with torch.no_grad():
            mean_per_soc = per_soc_demand_share_j.mean(dim=0)         # (S,)
            floor = mean_per_soc * float(args.tau_match_floor_mult)   # (S,)
        rum.set_tau_match_floor(floor)
        SOC = ["mgr","prof","assoc","admin","trades","care","sales","oper","elem"]
        print(f"        tau_match_floor: mult={args.tau_match_floor_mult}, k_min={args.k_match_min_per_soc}")
        for s in range(rum.n_soc):
            print(f"          {SOC[s]:6s}  floor={float(floor[s]):.4f}  (= {args.tau_match_floor_mult} × mean {float(mean_per_soc[s]):.4f})")

    # Inject the frozen lit-anchored match mask AFTER head construction
    # (mask depends on demand_share, which is computed earlier from aux data).
    if args.use_frozen_match_mask:
        assert per_soc_demand_share_j is not None, \
            "--use-frozen-match-mask requires per_soc_demand_share_j (aux soc_props/grid_industry/epsilon)"
        with torch.no_grad():
            ds_NS = per_soc_demand_share_j                              # (N, S) on device
            mean_per_soc = ds_NS.mean(dim=0, keepdim=True)               # (1, S)
            tau_per_soc = mean_per_soc * float(args.match_mean_mult)    # (1, S)
            pass_mask = ds_NS > tau_per_soc                             # (N, S) bool
            log_mask = torch.where(
                pass_mask,
                torch.zeros_like(ds_NS),
                torch.full_like(ds_NS, float(args.match_mask_log_penalty)),
            ).t().contiguous()                                          # (S, N)
        rum.set_frozen_match_mask(log_mask)
        SOC = ["mgr","prof","assoc","admin","trades","care","sales","oper","elem"]
        sizes = [int(pass_mask[:, s].sum()) for s in range(pass_mask.shape[1])]
        print(f"        frozen match mask: mult={args.match_mean_mult}, "
              f"penalty={args.match_mask_log_penalty}")
        for s in range(len(SOC)):
            tau_val = float(tau_per_soc[0, s])
            print(f"          {SOC[s]:6s}  tau={tau_val:.4f}  |J_s|={sizes[s]:4d}/{N}  "
                  f"({100*sizes[s]/N:.0f}%)")
        orphans = int((pass_mask.sum(dim=1) == 0).sum())
        print(f"          orphan grids (in 0 SOC sets): {orphans}")

    # If busy-dest boost enabled: compute top-K busy destinations from training data
    # (total observed inflow per destination), register on head as a buffer.
    if args.n_busy_dest > 0:
        K = args.n_busy_dest
        with torch.no_grad():
            # Rank destinations by train-origin inflow only (avoid val info leak).
            train_inflow = (observed_OD * train_mask.view(1, N, 1).float()).sum(dim=(0, 1))
            top_idx = torch.argsort(train_inflow, descending=True)[:K]    # (K,)
            dest_to_k = torch.full((N,), -1, dtype=torch.long, device=device)
            dest_to_k[top_idx] = torch.arange(K, device=device)
        rum.set_busy_dest_index(dest_to_k)
        print(f"        busy-dest boost: top-{K} destinations registered "
              f"(train inflow range [{float(train_inflow[top_idx[-1]]):.0f}, "
              f"{float(train_inflow[top_idx[0]]):.0f}])")
    print(f"        GNN mode: {args.gnn_mode}  λ_nn_norm: {args.lambda_nn_norm}")
    if args.use_match_gate:
        print(f"        Match gate: ON (k init {args.gate_steepness_init}, τ init {args.gate_threshold_init})")

    print(f"\n[train] starting: seed={args.seed} epochs={args.epochs} "
          f"blend_max={args.blend_max} λ_kl={args.lambda_kl} kl_warmup={args.kl_warmup_epochs}")
    print(f"        rum params: {sum(p.numel() for p in rum.parameters())}")
    print(f"        encoder params: {sum(p.numel() for p in encoder.parameters())}")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.lr_theta, "weight_decay": 1e-4},
            {"params": rum.parameters(), "lr": args.lr_rum, "weight_decay": 0.0},
        ]
    )

    # ===================================================================
    # Sequential training (Wang TB-ResNet + boosting extension)
    # n_stages=2 (Wang): Stage 0 RUM only → Stage 1 NN only (legacy behavior)
    # n_stages=4 (boosting): Stage 0 RUM → Stage 1 NN → Stage 2 RUM (NN frozen)
    #                        → Stage 3 NN (RUM frozen). Tests whether RUM in Stage 2
    #                        can absorb part of NN residual learned in Stage 1.
    # Even stages = "RUM-only" (freeze encoder + gnn scale);
    # Odd stages  = "NN-only" (freeze RUM except gnn scale).
    # ===================================================================
    SCALE_OFF = math.log(math.exp(1e-4) - 1.0)   # inv_softplus(1e-4) ≈ -9.21

    def _apply_stage(stage_idx: int) -> str:
        """Set requires_grad on encoder/rum for stage_idx. Returns human label."""
        is_rum_stage = (stage_idx % 2 == 0)
        if is_rum_stage:
            # Encoder frozen
            for p in encoder.parameters():
                p.requires_grad = False
            # Freeze gnn scale/blend. Stage 0: also force to ~0. Stage 2+: keep value.
            if rum.gnn_mode == "residual" and rum.raw_gnn_residual_scale is not None:
                if stage_idx == 0:
                    with torch.no_grad():
                        rum.raw_gnn_residual_scale.fill_(SCALE_OFF)
                rum.raw_gnn_residual_scale.requires_grad = False
            if rum.gnn_mode == "convex" and rum.raw_gnn_blend is not None:
                if stage_idx == 0:
                    with torch.no_grad():
                        rum.raw_gnn_blend.fill_(-15.0)
                rum.raw_gnn_blend.requires_grad = False
            # Unfreeze rest of RUM
            for name, p in rum.named_parameters():
                if "raw_gnn_residual_scale" in name or "raw_gnn_blend" in name:
                    continue
                p.requires_grad = True
            return f"Stage {stage_idx} [RUM-only]"
        else:
            # NN-only stage
            for p in encoder.parameters():
                p.requires_grad = True
            if rum.gnn_mode == "residual" and rum.raw_gnn_residual_scale is not None:
                # If raw is in vanishing-gradient dead zone (e.g. Stage 0's SCALE_OFF ~=-9.21,
                # softplus'(-9.21) ~= 1e-4 kills grad flow), bump back to init so encoder
                # can actually learn. Otherwise keep value from previous NN stage.
                if float(rum.raw_gnn_residual_scale) < -5.0:
                    raw_init = math.log(
                        math.exp(max(float(args.gnn_residual_scale_init), 1e-3)) - 1.0
                    )
                    with torch.no_grad():
                        rum.raw_gnn_residual_scale.fill_(raw_init)
                rum.raw_gnn_residual_scale.requires_grad = True
            if rum.gnn_mode == "convex" and rum.raw_gnn_blend is not None:
                if args.fixed_blend is not None:
                    fb = max(min(float(args.fixed_blend) / max(args.blend_max, 1e-6), 1-1e-4), 1e-4)
                    raw_b = math.log(fb / (1 - fb))
                    with torch.no_grad():
                        rum.raw_gnn_blend.fill_(raw_b)
                    rum.raw_gnn_blend.requires_grad = False
                else:
                    rum.raw_gnn_blend.requires_grad = True
            # Freeze rest of RUM
            for name, p in rum.named_parameters():
                if "raw_gnn_residual_scale" in name or "raw_gnn_blend" in name:
                    continue
                p.requires_grad = False
            return f"Stage {stage_idx} [NN-only]"

    # Compute stage durations and boundaries
    if args.train_mode == "sequential":
        n_stages = args.n_stages
        stage_durations = [args.epochs_phase1 if s % 2 == 0 else args.epochs
                           for s in range(n_stages)]
        stage_starts = [0]
        for d in stage_durations[:-1]:
            stage_starts.append(stage_starts[-1] + d)
        total_epochs = sum(stage_durations)
        # Apply Stage 0 setup
        label = _apply_stage(0)
        print(f"\n[sequential] {label}: ({stage_durations[0]} epochs)  "
              f"n_stages={n_stages} boundaries={stage_starts}")
    else:
        n_stages = 1
        stage_durations = [args.epochs]
        stage_starts = [0]
        total_epochs = args.epochs

    t0 = time.time()
    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []
    current_stage = 0
    stage_t0 = t0

    for ep in range(total_epochs):
        # Detect stage transition
        if args.train_mode == "sequential":
            next_stage = current_stage
            for s_idx, start in enumerate(stage_starts):
                if ep >= start:
                    next_stage = s_idx
            if next_stage != current_stage:
                # Log previous stage summary
                prev_cpc = history[-1]["cpc"] if history else 0.0
                print(f"[sequential] Stage {current_stage} done in "
                      f"{time.time()-stage_t0:.0f}s, CPC={prev_cpc:.4f}")
                current_stage = next_stage
                stage_t0 = time.time()
                label = _apply_stage(current_stage)
                print(f"[sequential] {label}: ({stage_durations[current_stage]} epochs)")
                # Reset optimizer for new freeze set
                optimizer = torch.optim.AdamW(
                    [{"params": [p for p in encoder.parameters() if p.requires_grad],
                      "lr": args.lr_theta, "weight_decay": 1e-4},
                     {"params": [p for p in rum.parameters() if p.requires_grad],
                      "lr": args.lr_rum, "weight_decay": 0.0}]
                )
                # Keep best_val/best_state across stages (same val set, comparable);
                # only reset patience counter so each stage gets fresh chance.
                no_improve = 0

        # ---- adjust epoch index for kl_warmup (relative to current stage start) ----
        if args.train_mode == "sequential":
            eff_ep = ep - stage_starts[current_stage]
        else:
            eff_ep = ep
        if args.kl_warmup_epochs > 0:
            kl_weight = args.lambda_kl * min(1.0, (eff_ep + 1) / args.kl_warmup_epochs)
        else:
            kl_weight = args.lambda_kl

        encoder.train(); rum.train()
        optimizer.zero_grad()
        out = forward_cs(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
            income_score, pct_kids, mean_cars, income_tier_props,
            pi_m_pair, grid_borough_idx, observed_OD, train_mask,
            soc_props_per_origin=soc_props_per_origin,
            per_soc_demand_share_j=per_soc_demand_share_j,
        )
        loss = (out["nll_dest"]
                + kl_weight * out["ce_mode"]
                + args.lambda_nn_norm * out["nn_norm_sq"]
                + args.lambda_ortho * out["ortho_cos_sq"]
                + args.lambda_match_ortho * out["match_ortho_cos_sq"]
                + args.lambda_iv_balance * (out["iv_share_across_j"] - args.target_iv_share) ** 2)
        if rum.use_self_loop_boost and args.self_loop_l2 > 0:
            loss = loss + args.self_loop_l2 * (rum.self_loop_boost ** 2).mean()
        if rum.n_busy_dest > 0 and args.busy_dest_l2 > 0:
            loss = loss + args.busy_dest_l2 * (rum.busy_dest_boost ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(rum.parameters()), 1.0
        )
        optimizer.step()

        encoder.eval(); rum.eval()
        with torch.no_grad():
            out_e = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
            )
            val_nll = float(out_e["nll_dest"])
            val_cpc = cpc(out_e["log_P_D"], observed_OD, val_mask)
            diag = out_e["diagnostic"]
            ce_val = float(out_e["ce_mode"])

        history.append({
            "ep": ep, "stage": current_stage,
            "tnll": float(out["nll_dest"].item()),
            "ce_train": float(out["ce_mode"].item()),
            "vnll": val_nll, "ce_val": ce_val, "cpc": val_cpc,
            "kl_weight": kl_weight, **{k: v for k, v in diag.items()
                                       if k not in ("asc_per_mode", "theta_inc_per_mode")},
        })

        if args.verbose or ep % 5 == 0 or ep == total_epochs - 1:
            if diag['gnn_mode'] == 'residual':
                gnn_str = (f"w_NN={diag['gnn_residual_scale']:.3f} "
                           f"δ_meas={diag['delta_measured']:.3f} "
                           f"(rum/nn rms {diag['rum_rms']:.2f}/{diag['nn_rms']:.2f})")
            elif diag['gnn_mode'] == 'mult':
                gnn_str = (f"γ_mult={diag['gnn_mult_scale']:.3f} "
                           f"δ_meas={diag['delta_measured']:.3f} "
                           f"(rum/nn rms {diag['rum_rms']:.2f}/{diag['nn_rms']:.2f})")
            elif diag['gnn_mode'] == 'moe':
                gnn_str = (f"g_mean={diag['gate_per_origin_mean']:.3f} "
                           f"g_std={diag['gate_per_origin_std']:.3f} "
                           f"δ_meas={diag['delta_measured']:.3f}")
            else:
                gnn_str = f"δ={diag['blend']:.3f}" if diag['blend'] is not None else "δ=n/a"
            # Handle tier-mixture (lists) vs single (floats) for display (ASCII only for Windows compat)
            def _mean(v):
                return sum(v) / len(v) if isinstance(v, list) else float(v)
            a, g, n, dm = _mean(diag['alpha_wage']), _mean(diag['gamma_M']), _mean(diag['nu_D']), _mean(diag['delta_match'])
            tier_tag = f"[T{rum.n_income_tiers}]" if diag.get('use_tier_mixture') else ""
            print(f"ep {ep:3d} | tnll {out['nll_dest'].item():.3f} | "
                  f"vnll {val_nll:.3f} | cpc {val_cpc:.3f} | klw {kl_weight:.2f} | {tier_tag}"
                  f"a_avg={a:+.3f} g_avg={g:+.3f} nu_avg={n:+.3f} "
                  f"dm_avg={dm:+.3f} | bt0={diag['beta_t_mean']:+.3f} "
                  f"bt1={diag['beta_t_slope_mean']:+.4f} | "
                  f"lam={diag['lambda_b_mean']:.3f} | {gnn_str}")

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            no_improve = 0
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "rum": {k: v.detach().clone() for k, v in rum.state_dict().items()},
            }
        else:
            no_improve += 1
        # Only allow early stop in the final stage (must reach all stage transitions first)
        in_final_stage = (args.train_mode != "sequential"
                          or current_stage == n_stages - 1)
        if in_final_stage and no_improve >= args.patience:
            print(f"early stop ep {ep}")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        rum.load_state_dict(best_state["rum"])

    encoder.eval(); rum.eval()
    with torch.no_grad():
        out_final = forward_cs(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
            income_score, pct_kids, mean_cars, income_tier_props,
            pi_m_pair, grid_borough_idx, observed_OD, val_mask,
            soc_props_per_origin=soc_props_per_origin,
            per_soc_demand_share_j=per_soc_demand_share_j,
        )
        final_cpc = cpc(out_final["log_P_D"], observed_OD, val_mask)
        final_diag = out_final["diagnostic"]
        full_snapshot = rum.snapshot()

        # ===================================================================
        # Ablation CPC drop — TRUE attribute importance: zero out each
        # component, measure CPC drop. This reflects prediction influence
        # rather than raw RMS magnitude (which is scale-biased toward IV_mode).
        # ===================================================================
        print("\n[ablation] muting each component, measuring CPC drop...")
        baseline_rum_state = {k: v.detach().clone() for k, v in rum.state_dict().items()}
        # Mute strategy per param:
        #   raw_* (softplus): set to -10 -> softplus ~ 1e-4 (effective 0)
        #   free params (asc/theta): set to 0 (direct mute)
        #   self_loop_boost: set to 0 (direct)
        #   raw_gnn_blend (sigmoid): set to -15 -> sigmoid ~ 3e-7
        MUTE_MAP = {
            # Destination attractor decomposition
            "gravity_gamma_logM":    ["raw_gamma_M"],
            "wage_alpha_logW":       ["raw_alpha_wage"],
            "competition_nu_logD":   ["raw_nu_D"],
            "occmatch_delta_M":      ["raw_delta_match"],
            # Mode-choice IV decomposition (per Wang's RUM family)
            "IV_time_cost_beta_t":   ["raw_beta_t_per_mode", "raw_beta_t_slope_per_mode"],
            "tier_threshold_kink":   ["raw_beta_t_kink_per_tier"],
            # Note: consideration filter is a structural layer (sigmoid mask, multiplicative
            # log effect), not simply mutable by zeroing a coef. Filter pass rates / learned
            # thresholds are reported via diagnostic instead of via ablation drop.
            "IV_mode_ASC":           ["asc_per_mode"],
            "IV_income_x_mode":      ["theta_inc_per_mode"],
            "IV_kids_x_mode":        ["theta_kids_per_mode"],
            "IV_cars_x_mode":        ["theta_cars_per_mode"],
            # Other
            "self_loop_boost":       ["self_loop_boost"],
            "nn_residual":           ["raw_gnn_residual_scale", "raw_gnn_blend"],
        }
        # Direct-zero params (not raw softplus)
        ZERO_DIRECT = {"asc_per_mode", "theta_inc_per_mode", "theta_kids_per_mode",
                       "theta_cars_per_mode", "self_loop_boost"}
        ablation_cpc = {"baseline": final_cpc}
        ablation_drop = {}
        for comp_name, param_names in MUTE_MAP.items():
            muted = {k: v.clone() for k, v in baseline_rum_state.items()}
            any_muted = False
            for pn in param_names:
                if pn in muted:
                    if pn in ZERO_DIRECT:
                        muted[pn] = torch.zeros_like(muted[pn])
                    elif pn == "raw_gnn_blend":
                        muted[pn] = torch.full_like(muted[pn], -15.0)
                    else:
                        muted[pn] = torch.full_like(muted[pn], -10.0)
                    any_muted = True
            if not any_muted:
                continue
            rum.load_state_dict(muted)
            out_abl = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
            )
            abl_cpc = cpc(out_abl["log_P_D"], observed_OD, val_mask)
            ablation_cpc[comp_name] = abl_cpc
            ablation_drop[comp_name] = final_cpc - abl_cpc
            print(f"  mute {comp_name:<24s} CPC {abl_cpc:.4f}  drop {final_cpc - abl_cpc:+.4f}")
        # Restore best RUM
        rum.load_state_dict(baseline_rum_state)
        final_diag["ablation_cpc"] = ablation_cpc
        final_diag["ablation_drop"] = ablation_drop

        # ===================================================================
        # Structural ablation — temporarily disable filter component as a whole.
        # Reveals the TRUE contribution of (match_filter + cost_filter) combined.
        # ===================================================================
        if rum.use_consideration_filter:
            print("\n[structural-ablation] disable consideration filter entirely...")
            orig_filter_flag = rum.use_consideration_filter
            rum.use_consideration_filter = False
            out_no_filter = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
            )
            cpc_no_filter = cpc(out_no_filter["log_P_D"], observed_OD, val_mask)
            rum.use_consideration_filter = orig_filter_flag
            final_diag["structural_ablation"] = {
                "no_consideration_filter_cpc": cpc_no_filter,
                "no_consideration_filter_drop": final_cpc - cpc_no_filter,
            }
            print(f"  no filter: CPC={cpc_no_filter:.4f}  drop={final_cpc - cpc_no_filter:+.4f}")

            # Ablation: turn off ONLY the match-pass sub-layer (keep cost-pass)
            print("\n[ablation] disable match-pass only (keep cost-pass)...")
            with torch.no_grad():
                if rum.use_soc_mixture and getattr(rum, "frozen_log_match_mask_per_soc", None) is not None:
                    # Frozen mask mode: report (S, N) binary set sizes
                    log_mask = rum.frozen_log_match_mask_per_soc.detach().cpu()  # (S, N)
                    in_set = (log_mask > -1e6)                                    # bool (S, N)
                    ds_np = per_soc_demand_share_j.detach().cpu().numpy()
                    import numpy as _np
                    pass_per_soc = []
                    for s in range(rum.n_soc):
                        size_s = int(in_set[s].sum())
                        pass_per_soc.append({
                            "soc": s,
                            "set_size": size_s,
                            "set_frac": size_s / int(in_set.shape[1]),
                            "demand_share_p05": float(_np.quantile(ds_np[:, s], 0.05)),
                            "demand_share_p95": float(_np.quantile(ds_np[:, s], 0.95)),
                            "demand_share_mean": float(ds_np[:, s].mean()),
                        })
                    final_diag["match_filter_stats_per_soc"] = pass_per_soc
                    final_diag["match_filter_mode"] = "frozen_mask"
                    for s, row in enumerate(pass_per_soc):
                        print(f"  SOC {s+1}: |J_s|={row['set_size']:4d}/{int(in_set.shape[1])} "
                              f"({100*row['set_frac']:.0f}%)  ds_mean={row['demand_share_mean']:.3f}")
                elif rum.use_soc_mixture:
                    # Per-SOC mode: 9 thresholds operate on demand_share[j, s]
                    tau_per_soc = rum.match_filter_thresh_per_soc.detach().cpu().tolist()
                    k_per_soc = rum.k_match_filter_per_soc.detach().cpu().tolist()
                    ds_np = per_soc_demand_share_j.detach().cpu().numpy()    # (N, S)
                    import numpy as _np
                    pass_per_soc = []
                    for s in range(rum.n_soc):
                        mp_s = 1.0 / (1.0 + _np.exp(-k_per_soc[s] * (ds_np[:, s] - tau_per_soc[s])))
                        below = float(_np.mean(ds_np[:, s] < tau_per_soc[s]))
                        pass_per_soc.append({
                            "soc": s,
                            "tau": tau_per_soc[s],
                            "k_sharpness": k_per_soc[s],
                            "frac_below_thresh": below,
                            "mean_pass": float(mp_s.mean()),
                            "demand_share_p05": float(_np.quantile(ds_np[:, s], 0.05)),
                            "demand_share_p95": float(_np.quantile(ds_np[:, s], 0.95)),
                        })
                    final_diag["match_filter_stats_per_soc"] = pass_per_soc
                    final_diag["match_filter_mode"] = "learnable_per_soc_tau"
                    for s, row in enumerate(pass_per_soc):
                        print(f"  SOC {s+1}: τ={row['tau']:.3f}  k={row['k_sharpness']:.1f}  "
                              f"below_thresh={100*row['frac_below_thresh']:.1f}%  "
                              f"mean_pass={row['mean_pass']:.3f}")
                else:
                    # Legacy scalar
                    k_m_real = float(rum.k_match_filter)
                    t_m_real = float(rum.match_filter_thresh)
                    match_pass = torch.sigmoid(k_m_real * (match_prob - t_m_real))
                    n_pairs_below_thresh = int((match_prob < t_m_real).sum())
                    total_pairs = int(match_prob.numel())
                    mean_match_pass = float(match_pass.mean())
                    median_match_pass = float(match_pass.median())
                    p05_match_pass = float(torch.quantile(match_pass.reshape(-1), 0.05))
                    p25_match_pass = float(torch.quantile(match_pass.reshape(-1), 0.25))
                    final_diag["match_filter_stats"] = {
                        "threshold_actual": t_m_real,
                        "k_sharpness": k_m_real,
                        "pairs_below_threshold_pct": 100 * n_pairs_below_thresh / total_pairs,
                        "mean_match_pass": mean_match_pass,
                        "median_match_pass": median_match_pass,
                        "p05_match_pass": p05_match_pass,
                        "p25_match_pass": p25_match_pass,
                        "match_data_min": float(match_prob.min()),
                        "match_data_max": float(match_prob.max()),
                        "match_data_mean": float(match_prob.mean()),
                    }
            if not rum.use_soc_mixture:
                print(f"  match threshold = {t_m_real:.4f}  (data min={float(match_prob.min()):.4f}, "
                      f"max={float(match_prob.max()):.4f})")
                print(f"  pairs below threshold: {n_pairs_below_thresh}/{total_pairs} "
                      f"({100*n_pairs_below_thresh/total_pairs:.2f}%)")
                print(f"  match_pass: mean={mean_match_pass:.3f}  median={median_match_pass:.3f}  "
                      f"p05={p05_match_pass:.3f}  p25={p25_match_pass:.3f}")

        # ===================================================================
        # Cosine-match-only ablation: feed match_prob = ones so log_cosine_match
        # = 0 in V_M_c. Isolates the γ_M·log_cosine_match contribution from
        # the γ_M·log_M_j gravity (otherwise lumped under "gravity_gamma_logM").
        # ===================================================================
        print("\n[ablation] mute cosine_match only (γ_M·log_cosine_match = 0)...")
        with torch.no_grad():
            out_no_match = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij,
                torch.ones_like(match_prob),              # cosine = 1 → log = 0
                log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
            )
        cpc_no_cosine_match = cpc(out_no_match["log_P_D"], observed_OD, val_mask)
        final_diag["cosine_match_ablation"] = {
            "cpc_no_cosine_match": cpc_no_cosine_match,
            "cosine_match_drop": final_cpc - cpc_no_cosine_match,
        }
        print(f"  no cosine_match: CPC={cpc_no_cosine_match:.4f}  "
              f"drop={final_cpc - cpc_no_cosine_match:+.4f}")

        # ===================================================================
        # JOINT ablation: NN-residual OFF and cosine_match OFF together.
        # Difference (joint - NN-only - cosine-only) ≈ true cosine_match
        # contribution when NN cannot serve as redundant backup.
        # ===================================================================
        print("\n[ablation] mute NN AND cosine_match together...")
        # First measure NN-only (no NN, cosine_match active)
        baseline_state = {k: v.detach().clone() for k, v in rum.state_dict().items()}
        muted_nn = {k: v.clone() for k, v in baseline_state.items()}
        for pn in ("raw_gnn_residual_scale", "raw_gnn_blend"):
            if pn in muted_nn:
                muted_nn[pn] = torch.full_like(muted_nn[pn], -15.0)
        rum.load_state_dict(muted_nn)
        with torch.no_grad():
            out_no_nn = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, match_prob,
                log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
            )
            cpc_no_nn = cpc(out_no_nn["log_P_D"], observed_OD, val_mask)
            # Joint: NN off AND cosine_match off
            out_no_both = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, torch.ones_like(match_prob),
                log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
            )
            cpc_no_both = cpc(out_no_both["log_P_D"], observed_OD, val_mask)
        rum.load_state_dict(baseline_state)
        # True cosine_match contribution when NN is unavailable:
        cosine_match_drop_no_nn = cpc_no_nn - cpc_no_both
        final_diag["joint_nn_cosine_ablation"] = {
            "cpc_no_nn_only":              cpc_no_nn,
            "cpc_no_nn_and_no_cosine":     cpc_no_both,
            "cosine_match_drop_under_no_nn": cosine_match_drop_no_nn,
            "nn_drop_alone":               final_cpc - cpc_no_nn,
            "joint_drop_from_baseline":    final_cpc - cpc_no_both,
        }
        print(f"  no NN only:                  CPC={cpc_no_nn:.4f}  drop={final_cpc-cpc_no_nn:+.4f}")
        print(f"  no NN + no cosine_match:     CPC={cpc_no_both:.4f}  drop={final_cpc-cpc_no_both:+.4f}")
        print(f"  ⭐ TRUE cosine_match effect (when NN absent): {cosine_match_drop_no_nn:+.4f}")

        # ===================================================================
        # Sub-population CPC breakdown
        # Group val origins by (dominant income tier × high/low kids)
        # Validates whether the decision-tree + heterogeneous utility captures
        # different commuting patterns across sub-populations consistently.
        # ===================================================================
        print("\n[sub-pop] CPC by (income tier × kids) on val origins...")
        dom_tier = income_tier_props.argmax(dim=1).cpu().numpy()                # (N,)
        kids_med = float(pct_kids.median().cpu())
        high_kids = (pct_kids > kids_med).cpu().numpy()                          # (N,) bool
        val_np = val_mask.cpu().numpy()
        P_full = out_final["log_P_D"].exp()                                      # (T, N, N)
        row_sum_full = observed_OD.sum(dim=2, keepdim=True)                     # (T, N, 1)
        pred_full = P_full * row_sum_full                                        # (T, N, N)

        sub_pop_cpc = {}
        n_tiers = income_tier_props.shape[1]
        for ti in range(n_tiers):
            for ki, kids_label in enumerate(("low_kids", "high_kids")):
                kids_flag = (high_kids == bool(ki))
                in_group = val_np & (dom_tier == ti) & kids_flag                # (N,)
                if in_group.sum() == 0:
                    continue
                idx = torch.from_numpy(in_group).to(device)
                p_sub = pred_full[:, idx]                                        # (T, n_sub, N)
                o_sub = observed_OD[:, idx]                                      # (T, n_sub, N)
                num = 2.0 * torch.minimum(p_sub, o_sub).sum()
                den = (p_sub.sum() + o_sub.sum()).clamp(min=1.0)
                cpc_group = float(num / den)
                tier_name = ("low", "mid", "high")[ti] if ti < 3 else f"t{ti}"
                key = f"{tier_name}_x_{kids_label}"
                sub_pop_cpc[key] = {
                    "cpc": cpc_group,
                    "n_origins": int(in_group.sum()),
                }
                print(f"  {key:<24s} n={int(in_group.sum()):>4d}  CPC={cpc_group:.4f}")
        final_diag["sub_population_cpc"] = sub_pop_cpc
        final_diag["pct_kids_median"] = kids_med

        # ===================================================================
        # Residual diagnostic — find where the model fails (CPC ceiling shape)
        # ===================================================================
        residual_diag = None
        if args.dump_residual_diag:
            print("\n[residual-diag] computing breakdowns...")
            P_full = out_final["log_P_D"].exp()                            # (T, N, N)
            row_sum_full = observed_OD.sum(dim=2, keepdim=True)            # (T, N, 1)
            pred_full = P_full * row_sum_full                              # (T, N, N) — flow

            def _cpc_subset(p, o):
                num = 2.0 * torch.minimum(p, o).sum()
                den = (p.sum() + o.sum()).clamp(min=1.0)
                return float(num / den)

            T_, N_ = P_full.shape[0], P_full.shape[1]

            # Per-hour CPC (val origins only)
            per_hour = []
            for tt in range(T_):
                per_hour.append(_cpc_subset(pred_full[tt, val_mask], observed_OD[tt, val_mask]))

            # Per-borough CPC (val origins grouped by their borough)
            per_borough = {}
            for b in range(rum.n_boroughs):
                mb = val_mask & (grid_borough_idx == b)
                if int(mb.sum()) > 0:
                    per_borough[int(b)] = {
                        "cpc": _cpc_subset(pred_full[:, mb], observed_OD[:, mb]),
                        "n_val_origins": int(mb.sum()),
                    }

            # Per-distance-bin CPC
            n_bins = 10
            d_min, d_max = float(log_d_ij.min()), float(log_d_ij.max())
            edges = torch.linspace(d_min, d_max, n_bins + 1, device=device)
            val_mask_3d = val_mask.view(1, N_, 1).expand(T_, N_, N_)
            per_dist = []
            for k in range(n_bins):
                # (i, j) pairs in this distance bin
                pair_mask = (log_d_ij >= edges[k]) & (log_d_ij < edges[k + 1])
                # broadcast to (T, N, N) and AND with val origin mask
                cell_mask = pair_mask.unsqueeze(0) & val_mask_3d
                if int(cell_mask.sum()) > 0:
                    per_dist.append({
                        "bin": k,
                        "log_d_low": float(edges[k]),
                        "log_d_high": float(edges[k + 1]),
                        "cpc": _cpc_subset(pred_full[cell_mask], observed_OD[cell_mask]),
                        "n_cells": int(cell_mask.sum()),
                        "obs_mean": float(observed_OD[cell_mask].mean()),
                    })

            # Per-flow-magnitude bin CPC (val cells only)
            obs_val = observed_OD * val_mask_3d.float()
            obs_nz = obs_val[obs_val > 0]
            per_flow = []
            if obs_nz.numel() > 0:
                pcts = torch.quantile(obs_nz, torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0], device=device))
                for k in range(len(pcts) - 1):
                    bin_mask = (obs_val >= pcts[k]) & (obs_val < pcts[k + 1])
                    if int(bin_mask.sum()) > 0:
                        per_flow.append({
                            "bin": k,
                            "flow_low": float(pcts[k]),
                            "flow_high": float(pcts[k + 1]),
                            "cpc": _cpc_subset(pred_full[bin_mask], observed_OD[bin_mask]),
                            "n_cells": int(bin_mask.sum()),
                        })

            # Top-30 worst over- and under-predicted cells (val only)
            residual = (pred_full - observed_OD) * val_mask_3d.float()    # (T, N, N)
            over_vals, over_idx = torch.topk(residual.flatten(), k=30)     # most positive (over-pred)
            under_vals, under_idx = torch.topk((-residual).flatten(), k=30)  # most negative (under-pred)
            N2 = N_ * N_

            def _unflat(idx):
                tt = int(idx // N2)
                ii = int((idx % N2) // N_)
                jj = int(idx % N_)
                return tt, ii, jj

            def _cell_record(idx):
                tt, ii, jj = _unflat(idx)
                return {
                    "t": tt, "i": ii, "j": jj,
                    "pred": float(pred_full[tt, ii, jj]),
                    "obs": float(observed_OD[tt, ii, jj]),
                    "residual": float(pred_full[tt, ii, jj] - observed_OD[tt, ii, jj]),
                    "log_d": float(log_d_ij[ii, jj]),
                    "borough_i": int(grid_borough_idx[ii]),
                    "borough_j": int(grid_borough_idx[jj]),
                }

            top_over = [_cell_record(i) for i in over_idx.tolist()]
            top_under = [_cell_record(i) for i in under_idx.tolist()]

            residual_diag = {
                "per_hour_cpc": per_hour,
                "per_borough_cpc": per_borough,
                "per_distance_bin": per_dist,
                "per_flow_bin": per_flow,
                "top_over_predicted": top_over,
                "top_under_predicted": top_under,
                "n_val_origins": int(val_mask.sum()),
                "T": int(T_),
                "N": int(N_),
            }
            print(f"  per-hour CPC range: [{min(per_hour):.3f}, {max(per_hour):.3f}]")
            print(f"  per-borough CPC range: [{min(b['cpc'] for b in per_borough.values()):.3f}, "
                  f"{max(b['cpc'] for b in per_borough.values()):.3f}] ({len(per_borough)} boroughs with val origins)")
            print(f"  per-distance-bin CPC range: [{min(d['cpc'] for d in per_dist):.3f}, "
                  f"{max(d['cpc'] for d in per_dist):.3f}]")
            print(f"  per-flow-bin CPC range: [{min(f['cpc'] for f in per_flow):.3f}, "
                  f"{max(f['cpc'] for f in per_flow):.3f}]")

    elapsed = time.time() - t0
    result = {
        "config": "cervero_shen_smoke",
        "seed": args.seed,
        "epochs_actual": len(history),
        "epochs_max": args.epochs,
        "lambda_kl": args.lambda_kl,
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "blend_max": args.blend_max,
        "train_mode": args.train_mode,
        "n_stages": args.n_stages if args.train_mode == "sequential" else 1,
        "stage_durations": stage_durations if args.train_mode == "sequential" else [args.epochs],
        "stage_starts": stage_starts if args.train_mode == "sequential" else [0],
        "fit_time_s": elapsed,
        "final_cpc": final_cpc,
        "final_diagnostic": final_diag,
        "rum_snapshot": full_snapshot,
        "v2_baseline_cpc": 0.485,
        "v3a_dm_nested_cpc": 0.534,
        "history": history,
        "residual_diag": residual_diag,
    }
    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Persist weights so downstream scenario / counterfactual scripts can re-load
    # the trained model without retraining. Sibling file to the JSON result.
    ckpt_path = out_path.with_suffix(".pt")
    ckpt_payload = {
        "encoder_state": encoder.state_dict(),
        "rum_state": rum.state_dict(),
        "args": vars(args),
        "final_cpc": float(final_cpc),
        "n_modes": int(M),
        "n_boroughs": int(n_boroughs),
        "static_dim": int(X_static.shape[1]),
        "dyn_dim": int(X_dynamic.shape[-1]),
        "N": int(N),
        "T": int(T),
        "encoder_kind": (
            "DualBranchPairEncoder" if args.use_dual_pair_encoder
            else "PairResidualNN" if args.use_pair_nn
            else "DualBranchGATEncoder" if args.use_gat
            else "DualBranchEncoder"
        ),
    }
    torch.save(ckpt_payload, ckpt_path)
    print(f"  weights saved to {ckpt_path}")

    print(f"\n[smoke] done in {elapsed:.0f}s ({len(history)} epochs)")
    print(f"  CPC            = {final_cpc:.4f}  (v2 baseline 0.485, v3a 0.534)")
    def _fmt(v, w=8):
        if isinstance(v, list):
            return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"
        return f"{v:+.{w-4}f}"
    if final_diag.get("use_tier_mixture"):
        print(f"  TIER MIXTURE (low / mid / high income, marginal: {income_tier_props.mean(dim=0).tolist()}):")
    print(f"  alpha_wage     = {_fmt(final_diag['alpha_wage'])}  (W_j; ≥0 by construction)")
    print(f"  gamma_M        = {_fmt(final_diag['gamma_M'])}  (M_j; ≥0 by construction)")
    print(f"  nu_D           = {_fmt(final_diag['nu_D'])}  (D_j; ≤0 by construction)")
    print(f"  delta_match    = {_fmt(final_diag['delta_match'])}  (match_prob; ≥0 by construction)")
    print(f"  beta_t,0 mean  = {final_diag['beta_t_mean']:+.4f}  (t intercept; ≤0 by construction)")
    print(f"  beta_t,1 mean  = {final_diag['beta_t_slope_mean']:+.4f}  (t slope on log_d; ≤0 by construction)")
    if final_diag.get("T_threshold_per_tier"):
        T = final_diag["T_threshold_per_tier"]
        bk = final_diag["beta_t_kink_per_tier"]
        ks = final_diag.get("k_sharpness_per_tier") or [0]*3
        print(f"  T_threshold per tier (min): low={T[0]:.1f}  mid={T[1]:.1f}  high={T[2]:.1f}")
        print(f"  β_kink     per tier        : low={bk[0]:+.4f}  mid={bk[1]:+.4f}  high={bk[2]:+.4f}")
        print(f"  k_sharpness per tier (min) : low={ks[0]:.2f}  mid={ks[1]:.2f}  high={ks[2]:.2f}  (smaller=sharper)")
    if final_diag.get("match_filter_thresh") is not None:
        print(f"  ----- Consideration filter (lexicographic) -----")
        print(f"  Layer 1 match: threshold = {final_diag['match_filter_thresh']:.4f}, "
              f"sharpness k = {final_diag['k_match_filter']:.2f}")
        budgets = final_diag["cost_budget_per_tier"]
        thrs = final_diag["cost_thresh_per_tier"]
        print(f"  Layer 2 cost:  θ_time={final_diag['theta_t_cost']:.4f}  "
              f"θ_dist={final_diag['theta_d_cost']:.4f}  k={final_diag['k_cost_filter']:.2f}")
        print(f"               budget per tier:  low={budgets[0]:.3f}  mid={budgets[1]:.3f}  high={budgets[2]:.3f}")
        print(f"               ratio thresh   :  low={thrs[0]:.3f}  mid={thrs[1]:.3f}  high={thrs[2]:.3f}")
    print(f"  theta_inc      = {final_diag.get('theta_inc_per_mode')}  (per-mode income coef)")
    print(f"  theta_kids     = {final_diag.get('theta_kids_per_mode')}  (per-mode pct_kids coef)")
    print(f"  theta_cars     = {final_diag.get('theta_cars_per_mode')}  (per-mode mean_cars coef)")
    print(f"  λ_b mean/std   = {final_diag['lambda_b_mean']:.3f} / {final_diag['lambda_b_std']:.3f}")
    if final_diag.get("use_push_pull"):
        print(f"  xi_push_pull   = {final_diag['xi_push_pull']:+.4f}  "
              f"(origin labor surplus × dest M_j; expected > 0 if push amplifies pull)")
    if final_diag.get("gnn_mode") == "residual":
        print(f"  w_NN (scale)   = {final_diag['gnn_residual_scale']:.4f}  (Wang TB-ResNet additive)")
        print(f"  δ_measured     = {final_diag['delta_measured']:.4f}  (=‖V_NN‖/(‖V_RUM‖+‖V_NN‖); <0.30 = Path A)")
        print(f"  RUM/NN RMS     = {final_diag['rum_rms']:.4f} / {final_diag['nn_rms']:.4f}")
    elif final_diag.get("gnn_mode") == "mult":
        print(f"  γ_mult         = {final_diag['gnn_mult_scale']:.4f}  (multiplicative gating: V=V_RUM·(1+γ·σ(V_NN)))")
        print(f"  δ_measured     = {final_diag['delta_measured']:.4f}  (=‖ΔV‖/(‖V_RUM‖+‖ΔV‖))")
        print(f"  RUM/ΔV RMS     = {final_diag['rum_rms']:.4f} / {final_diag['nn_rms']:.4f}")
    elif final_diag.get("gnn_mode") == "moe":
        print(f"  per-origin gate g_i mean/std = {final_diag['gate_per_origin_mean']:.4f} / {final_diag['gate_per_origin_std']:.4f}")
        print(f"  δ_measured     = {final_diag['delta_measured']:.4f}  (gated NN contribution share)")
        print(f"  RUM/NN RMS     = {final_diag['rum_rms']:.4f} / {final_diag['nn_rms']:.4f}")
    elif final_diag["blend"] is not None:
        print(f"  blend (δ_GNN)  = {final_diag['blend']:.4f}  (v3a 0.437, legacy convex)")
    # Attribute share table (paper interpretation)
    if "attribute_shares" in final_diag:
        print("  Attribute shares (variance share of total logit; mode_choice_IV dominates by scale):")
        for k, v in sorted(final_diag["attribute_shares"].items(), key=lambda kv: -kv[1]):
            rms = final_diag["attribute_rms"][k]
            print(f"    {k:<24s} {v*100:5.1f}%   (RMS {rms:.3f})")
    if "attribute_shares_dest" in final_diag:
        print("  Destination-attractor shares (excl. mode_choice_IV — interpretable in paper):")
        for k, v in sorted(final_diag["attribute_shares_dest"].items(), key=lambda kv: -kv[1]):
            rms = final_diag["attribute_rms"][k]
            print(f"    {k:<24s} {v*100:5.1f}%   (RMS {rms:.3f})")
    if "attribute_shares_across_j" in final_diag:
        print("  Across-j variance shares (TRUE destination-discriminating importance):")
        for k, v in sorted(final_diag["attribute_shares_across_j"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:<24s} {v*100:5.1f}%")
    if "ablation_drop" in final_diag:
        print("  Ablation CPC drop (mute each component, measure CPC loss — paper-grade importance):")
        for k, v in sorted(final_diag["ablation_drop"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:<24s} {v:+.4f}")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
