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
    hour_weights: Optional[torch.Tensor] = None,            # (T,) — Plan E hour weighting:
                                                            # weights * flow → loss focus on
                                                            # commute-pure hours (Toole 2015,
                                                            # Iqbal 2014). None = legacy all-hours.
    log_transit_access_z: Optional[torch.Tensor] = None,    # (N,) z-scored PTAL proxy
    log_commercial_z: Optional[torch.Tensor] = None,        # (N,) z-scored commercial land use
    quality_nvm_z: Optional[torch.Tensor] = None,           # (N, N) z-scored modal flexibility
    quality_ta_z: Optional[torch.Tensor] = None,            # (N, N) z-scored log_transit_advantage
    quality_wf_z: Optional[torch.Tensor] = None,            # (N, N) z-scored walk_feasibility
    log_time_mask_per_tier: Optional[torch.Tensor] = None,  # (K_tier, N, N) frozen feasibility mask
    strip_time_from_vdest_iv: bool = False,                 # 2026-05-22 reframing — Component 3:
                                                            # compute a second IV from V_lower_quality
                                                            # = ASC + agent_term (NO β_t·t) and use
                                                            # it (not the full IV_mode) inside V_dest.
                                                            # Mode probability P(m|i,j) STILL uses
                                                            # the full IV_mode with raw time.
    tier_commute_probe: bool = False,                       # if True, accumulate per-income-tier
                                                            # predicted mean commute time (min, off-
                                                            # diagonal, hour-weighted) from each
                                                            # class's exp(log pi_c + log P_c). Used
                                                            # by the income x commute moment injection.
    soc_dest_probe: bool = False,                           # if True, accumulate per-SOC predicted
                                                            # destination flow (N,), off-diagonal,
                                                            # hour-weighted, from each class's
                                                            # exp(log pi_c + log P_c). Used by the
                                                            # occupation x industry moment injection
                                                            # to re-identify delta_match.
    force_no_ckpt: bool = False,                            # disable gradient checkpointing in the
                                                            # class loop. Windows-CPU torch SEGFAULTs
                                                            # in checkpointed backward; the moment
                                                            # injections (few params) fit in RAM
                                                            # without checkpointing, so set True there.
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

        # Frozen time mask (Component 1 of 2026-05-22 reframing): hard feasibility
        # gate independent of (and stronger than) the soft use_time_gate. Same
        # injection path — adds to log_cost_pass_per_tier so the per-class loop
        # already picks it up via cost_term_c.
        if log_time_mask_per_tier is not None:
            # (K_tier, N, N) broadcast onto (K_tier, T, N, N)
            log_cost_pass_per_tier = log_cost_pass_per_tier + log_time_mask_per_tier.unsqueeze(1)

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

    log_iv_quality = None  # only used when strip_time_from_vdest_iv=True
    _scaled_list = []      # per-mode V_lower/λ, for exposing mode log-probs (identifiability diag)
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
        _scaled_list.append(scaled)
        if log_iv is None:
            log_iv = scaled
        else:
            log_iv = torch.logaddexp(log_iv, scaled)
        pi_m_ij = pi_m_pair[:, :, m_idx].unsqueeze(0)                  # (1, N, N)
        if ce_accum is None:
            ce_accum = pi_m_ij * scaled
        else:
            ce_accum = ce_accum + pi_m_ij * scaled

        # 2026-05-22 reframing — Component 3: strip raw time from the IV that
        # feeds V_dest. V_lower_q_m has only ASC + agent_term (no β_t·t_m), so
        # IV_quality has no time variance. The full IV_mode (with time) is
        # still used for mode probability P(m|i,j) below.
        if strip_time_from_vdest_iv:
            V_lower_q_m = (asc[m_idx] + agent_term)                    # (1, N, 1) — no j-dep
            scaled_q = V_lower_q_m / lam_view                          # (1, N, N) — j-dep only via λ
            if log_iv_quality is None:
                log_iv_quality = scaled_q
            else:
                log_iv_quality = torch.logaddexp(log_iv_quality, scaled_q)

    IV_mode = log_iv                                                   # (T, N, N)
    ce_mode_per_ijt = IV_mode - ce_accum                               # (T, N, N)
    log_P_m = torch.stack(_scaled_list, dim=-1) - IV_mode.unsqueeze(-1)  # (T,N,N,M) mode log-probs
    # IV used inside V_dest: full (with time) by default, quality-only when stripping.
    IV_for_V_dest = log_iv_quality if strip_time_from_vdest_iv else IV_mode

    # ---- V_upper (Cervero 1999 multiplicative + optional tier-mixture) ----
    # Single (legacy):  γ_eff = γ + δ · match;  V = γ_eff·log_M + α·log_W + ν·log_D
    # Tier mixture: each (α_k, γ_k, ν_k, δ_k) is tier-specific (k = 0/1/2 for low/mid/high)
    #               final P(j|i) = Σ_k π_k(i) · softmax_j(V_upper_k)
    alpha_w = rum.alpha_wage          # scalar (legacy) or (K,) tensor (tier mixture)
    gamma_M = rum.gamma_M
    nu_D = rum.nu_D
    delta_m = rum.delta_match

    # Attribute attention modulators (None if disabled).
    # Δ_attr(i) is per-origin; γ_M, α_W, δ_match use multiplicative exp(Δ);
    # ν_D uses additive Δ (sign-free with --nu-D-sign-free). At init the MLP
    # output is zeroed → exp(0)=1 → no perturbation vs baseline.
    d_g_attn, d_a_attn, d_n_attn, d_d_attn = rum.attribute_modulators()
    if d_g_attn is not None:
        exp_d_g_attn = torch.exp(d_g_attn)        # (N,) > 0
        exp_d_a_attn = torch.exp(d_a_attn)        # (N,)
        exp_d_d_attn = torch.exp(d_d_attn)        # (N,)
    else:
        exp_d_g_attn = exp_d_a_attn = exp_d_d_attn = None

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

    soc_dest_flow = None   # (S, N) per-SOC predicted dest flow when soc_dest_probe (else None)
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
        use_ckpt = (rum.training or any(p.requires_grad for p in rum.parameters())) and not force_no_ckpt

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
            V_rum_c = (V_M_c + V_other_c).unsqueeze(0) + lam_view * IV_for_V_dest
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
            weighted = log_pi_c_row.view(1, N, 1) + log_P_c
            # Per-tier commute-time probe contributions, computed INSIDE the checkpoint so
            # the (T,N,N) intermediates are recomputed in backward, not stored (bounds mem).
            if tier_commute_probe or soc_dest_probe:
                _pf = weighted.exp() * _otot * _offd * _hw                   # (T,N,N) flow contrib
                _ctn = (_pf * t_min_per_pair).sum() if tier_commute_probe else weighted.new_zeros(())
                _ctd = _pf.sum() if tier_commute_probe else weighted.new_zeros(())
                _df = _pf.sum(dim=(0, 1)) if soc_dest_probe else weighted.new_zeros(N)  # (N,)
                return weighted, _ctn, _ctd, _df
            return (weighted, weighted.new_zeros(()), weighted.new_zeros(()),
                    weighted.new_zeros(N))

        log_P_accum = None
        V_rum_dest_avg = torch.zeros(T, N, N, device=V_gnn.device)
        has_kink = rum.use_tier_threshold
        # Optional per-tier commute-time probe (differentiable, memory-bounded): for each
        # income tier, accumulate sum of predicted off-diagonal flow * t_min and the flow
        # normalizer, using each class's contribution exp(log pi_c + log P_c).
        # Optional per-SOC destination-flow probe (differentiable, memory-bounded):
        # for each occupation, accumulate predicted off-diagonal, hour-weighted flow to
        # each destination from each class's contribution exp(log pi_c + log P_c), grouped
        # by soc_idx (= c % S). The (T,N,N) flow tensor is built INSIDE the checkpoint (see
        # _class_logp_step) so it is recomputed in backward, not stored x27 (that was a
        # SEGFAULT). Used to build the predicted occupation->industry landing table for the
        # occupation x industry moment injection (delta_match).
        _tcp = bool(tier_commute_probe)
        _sdp = bool(soc_dest_probe)
        if _sdp:
            assert rum.use_soc_mixture, "soc_dest_probe requires --use-soc-mixture"
            _Ns = rum.n_soc
            _sf = [None] * _Ns
        if _tcp:
            _Kt = rum.n_income_tiers
            _tnum = [None] * _Kt
            _tden = [None] * _Kt
        if _tcp or _sdp:   # shared probe weights (off-diagonal, observed production, hour)
            _otot = observed_OD.sum(dim=2, keepdim=True)                     # (T,N,1)
            _offd = (1.0 - torch.eye(N, device=V_gnn.device)).view(1, N, N)  # zero diagonal
            _hw = (hour_weights.view(-1, 1, 1) if hour_weights is not None
                   else torch.ones(T, 1, 1, device=V_gnn.device))
        for c in range(K):
            # Structured-heterogeneity lookup:
            #   tier_idx selects tier-only params (α_W, ν_D, T, β_kink, k_sharp)
            #   soc_idx selects SOC-only params (δ_match)
            #   c selects tier×SOC params (γ_M)
            if rum.use_soc_mixture:
                tier_idx = c // S
                soc_idx = c % S
                log_dshare_for_c = log_demand_share[:, soc_idx]              # (N,)
                # Stoll coupling on cosine match + per-SOC refinement.
                # With attribute attention: γ_M[c] / δ_match[soc] become per-origin via
                # multiplicative modulator (≥0 sign preserved).
                if exp_d_g_attn is not None:
                    gamma_M_eff = gamma_M[c] * exp_d_g_attn                          # (N,)
                    delta_m_eff = delta_m[soc_idx] * exp_d_d_attn                    # (N,)
                    V_M_c = (gamma_M_eff.view(N, 1) * (log_M_j.view(1, N) + log_cosine_match)
                             + delta_m_eff.view(N, 1) * log_dshare_for_c.view(1, N))  # (N, N)
                elif getattr(rum, "use_occ_specific_mass", False):
                    # Occupation-specific job mass M_{j,s} = M_j × demand_share[s,j].
                    # log M_{j,s} = log_M_j + log_dshare_for_c. Occupation is baked INTO
                    # gravity with the full γ_M weight (no separate small δ_match add-on),
                    # so destinations attract worker class c by their OCCUPATION-MATCHED
                    # job count, not total job mass. Tests whether aggregate flows reward
                    # occupation matching when it carries gravity-strength weight.
                    V_M_c = gamma_M[c] * (log_M_j.view(1, N) + log_dshare_for_c.view(1, N))  # (N, N)
                else:
                    V_M_c = (gamma_M[c] * (log_M_j.view(1, N) + log_cosine_match)
                             + delta_m[soc_idx] * log_dshare_for_c.view(1, N))        # (N, N)
            else:
                tier_idx = c                                                  # plain tier mixture
                if exp_d_g_attn is not None:
                    gamma_M_eff_t = gamma_M[c] * exp_d_g_attn                # (N,)
                    delta_m_eff_t = delta_m[c] * exp_d_d_attn                # (N,)
                    if log_match_signal is not None:
                        V_M_c = gamma_M_eff_t.view(N, 1) * (log_M_j.view(1, N) + log_match_signal)
                    else:
                        # gamma_eff(i, j) = γ[c]·exp(Δγ(i)) + δ[c]·exp(Δδ(i))·match(i,j)
                        gamma_eff_c = (gamma_M_eff_t.view(N, 1)
                                       + delta_m_eff_t.view(N, 1) * match_signal)   # (N, N)
                        V_M_c = gamma_eff_c * log_M_j.view(1, N)
                else:
                    if log_match_signal is not None:
                        V_M_c = gamma_M[c] * (log_M_j.view(1, N) + log_match_signal)
                    else:
                        gamma_eff_c = gamma_M[c] + delta_m[c] * match_signal
                        V_M_c = gamma_eff_c * log_M_j.view(1, N)
            if exp_d_a_attn is not None:
                # α_W: multiplicative exp(Δα); ν_D: additive Δν (sign-free path).
                alpha_w_eff = alpha_w[tier_idx] * exp_d_a_attn                       # (N,)
                nu_D_eff = nu_D[tier_idx] + d_n_attn                                 # (N,)
                V_other_c = (alpha_w_eff.view(N, 1) * log_W_j.view(1, N)
                             + nu_D_eff.view(N, 1) * log_D_j.view(1, N))             # (N, N)
            else:
                V_other_c = (alpha_w[tier_idx] * log_W_j
                             + nu_D[tier_idx] * log_D_j).view(1, N)
            if rum.beta_transit is not None:
                V_other_c = V_other_c + (rum.beta_transit * log_transit_access_z
                                         + rum.beta_commercial * log_commercial_z).view(1, N)
            if rum.use_quality_features and quality_nvm_z is not None:
                # Pair-wise (N, N) quality residuals — promote V_other_c to (N, N)
                # via broadcasting. Same downstream path as attention's V_other_c
                # shape upgrade.
                V_other_c = V_other_c + (rum.beta_nvm * quality_nvm_z
                                         + rum.beta_ta * quality_ta_z
                                         + rum.beta_wf * quality_wf_z)
            # R7/R8 — Hansen decay (tier-specific scalar × log_d_ij (N, N))
            if rum.use_hansen_decay:
                gamma_decay_t = rum.gamma_decay_per_tier[tier_idx]
                V_other_c = V_other_c - gamma_decay_t * log_d_ij                          # (N, N)
            # R8 — t_min cost (tier-specific scalar × t_min(i, j) (N, N))
            if rum.use_tmin_cost:
                beta_tmin_t = rum.beta_tmin_per_tier[tier_idx]
                # t_min_per_pair is (T, N, N) but static across T (data fact);
                # take first slice for the per-pair cost.
                V_other_c = V_other_c - beta_tmin_t * t_min_per_pair[0]                   # (N, N)
            # R7/R8 — Agent × destination explicit interactions.
            # Round 13 fix (2026-05-22): all 3 terms pair with attractor (log_M / log_W)
            # NOT log_d, to avoid multicollinearity with IV's implicit distance signal.
            #   θ_inc · income · log_M : income-jobs sorting (Rosen 1974)
            #   θ_kids · kids · log_W  : family-wage attraction (financial security)
            #   θ_cars · cars · log_M  : car-enabled job access (Schwanen 2003 mobility)
            if rum.use_agent_dest_interaction:
                V_other_c = (V_other_c
                             + rum.theta_inc_dest * income_score_per_origin.view(N, 1) * log_M_j.view(1, N)
                             + rum.theta_kids_dest * pct_kids_per_origin.view(N, 1) * log_W_j.view(1, N)
                             + rum.theta_cars_dest * mean_cars_per_origin.view(N, 1) * log_M_j.view(1, N))
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
                weighted_c, _nc, _dc, _fc = _ckpt(_class_logp_step,
                                   V_M_c, V_other_c, log_joint_per_class[c],
                                   T_c, bk_c, ks_c, has_kink,
                                   filter_term_c, match_term_c, cost_term_c,
                                   use_reentrant=False)
            else:
                weighted_c, _nc, _dc, _fc = _class_logp_step(V_M_c, V_other_c,
                                              log_joint_per_class[c],
                                              T_c, bk_c, ks_c, has_kink,
                                              filter_term_c, match_term_c, cost_term_c)
            log_P_accum = (weighted_c if log_P_accum is None
                           else torch.logaddexp(log_P_accum, weighted_c))
            if _tcp:  # per-tier commute-time accumulation (scalars from inside checkpoint)
                _tnum[tier_idx] = _nc if _tnum[tier_idx] is None else _tnum[tier_idx] + _nc
                _tden[tier_idx] = _dc if _tden[tier_idx] is None else _tden[tier_idx] + _dc
            if _sdp:  # per-SOC destination flow (N,) — from inside checkpoint, memory-bounded
                _si = c % S
                _sf[_si] = _fc if _sf[_si] is None else _sf[_si] + _fc
            # Detached diagnostic accumulator
            with torch.no_grad():
                V_rum_c_diag = (V_M_c + V_other_c).unsqueeze(0) + lam_view * IV_for_V_dest
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
        if _tcp:
            tier_commute_time = torch.stack([_tnum[k] / _tden[k].clamp_min(1e-6)
                                             for k in range(_Kt)])           # (K_tier,) minutes
        else:
            tier_commute_time = None
        if _sdp:
            soc_dest_flow = torch.stack([_sf[s] for s in range(_Ns)])       # (S, N)
    else:
        tier_commute_time = None
        # single-RUM path (original)
        # gamma_M / alpha_w / nu_D / delta_m are scalars here (shape (1,) or 0-dim).
        # With attribute attention they become per-origin via modulator.
        if exp_d_g_attn is not None:
            gamma_eff_i = gamma_M * exp_d_g_attn                              # (N,)
            delta_m_eff_i = delta_m * exp_d_d_attn                            # (N,)
            if log_match_signal is not None:
                V_M = gamma_eff_i.view(N, 1) * (log_M_j.view(1, N) + log_match_signal)  # (N, N)
            else:
                gamma_eff = (gamma_eff_i.view(N, 1)
                             + delta_m_eff_i.view(N, 1) * match_signal)       # (N, N)
                V_M = gamma_eff * log_M_j.view(1, N)
        else:
            if log_match_signal is not None:
                V_M = gamma_M * (log_M_j.view(1, N) + log_match_signal)       # (N, N)
            else:
                gamma_effective = gamma_M + delta_m * match_signal            # (N, N)
                V_M = gamma_effective * log_M_j.view(1, N)                    # (N, N)
        if exp_d_a_attn is not None:
            alpha_w_eff_i = alpha_w * exp_d_a_attn                            # (N,)
            nu_D_eff_i = nu_D + d_n_attn                                      # (N,)
            V_other = (alpha_w_eff_i.view(1, N, 1) * log_W_j.view(1, 1, N)
                       + nu_D_eff_i.view(1, N, 1) * log_D_j.view(1, 1, N))   # (1, N, N)
        else:
            V_other = (alpha_w * log_W_j.view(1, 1, N)
                       + nu_D * log_D_j.view(1, 1, N))                       # (1, 1, N)
        if rum.beta_transit is not None:
            V_other = V_other + (rum.beta_transit * log_transit_access_z
                                 + rum.beta_commercial * log_commercial_z).view(1, 1, N)
        if rum.use_quality_features and quality_nvm_z is not None:
            V_other = V_other + (rum.beta_nvm * quality_nvm_z
                                 + rum.beta_ta * quality_ta_z
                                 + rum.beta_wf * quality_wf_z).view(1, N, N)
        V_rum_dest = (V_M.unsqueeze(0) + V_other + lam_view * IV_for_V_dest)   # (T, N, N)
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
    # Plan E (Toole 2015 / Iqbal 2014 fusion): hour_weights re-weights commute-pure
    # hours up and non-commute hours down (or zero them out for hard slicing).
    # Math equivalence: L_weighted = Σ_h w_h · CE_h, which is the
    # weighted-MLE form of IPF calibration when w_h = P(commute|hour=h).
    mask_f = train_mask.view(1, N, 1).float()
    if hour_weights is not None:
        mask_f = mask_f * hour_weights.view(T, 1, 1)
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
            "mode_choice_IV":     (lam_view * IV_for_V_dest).expand(T, N, N),
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
    iv_var = (lam_view * IV_for_V_dest).var(dim=-1, unbiased=False).mean()
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
        "ce_mode_per_ijt": ce_mode_per_ijt,
        "log_P_m": log_P_m,
        "flow": flow,
        "flow_sum": flow_sum,
        "nn_norm_sq": nn_norm_sq,
        "ortho_cos_sq": ortho_cos_sq,
        "match_ortho_cos_sq": match_ortho_cos_sq,
        "iv_share_across_j": iv_share_across_j,
        "diagnostic": diagnostic,
        "tier_commute_time": tier_commute_time,
        "soc_dest_flow": soc_dest_flow,
    }


def cpc(log_P: torch.Tensor, observed_OD: torch.Tensor, mask: torch.Tensor,
        hour_weights: Optional[torch.Tensor] = None) -> float:
    """CPC = Sørensen index over destination inflows.

    When hour_weights is provided (Plan E), only hours with non-zero weight
    contribute. This keeps CPC aligned with the training objective: if loss
    is computed on commute-hours only, CPC must be too — otherwise a model
    trained on commute peaks gets scored on midday shopping flows and looks
    artificially bad.
    """
    with torch.no_grad():
        P = log_P.exp()
        row_sum = observed_OD.sum(dim=2, keepdim=True)
        pred = P * row_sum
        if hour_weights is not None:
            keep_h = (hour_weights > 0).nonzero(as_tuple=True)[0]
            pred = pred[keep_h]
            obs = observed_OD[keep_h]
        else:
            obs = observed_OD
        num = 2.0 * torch.minimum(pred[:, mask], obs[:, mask]).sum()
        den = (pred[:, mask].sum() + obs[:, mask].sum()).clamp(min=1.0)
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
    ap.add_argument("--lambda-occ-penalty", type=float, default=0.0,
                    help="Ben-Akiva-Morikawa in-loss external-moment penalty: anchors per-SOC "
                         "delta_match (z-scored pattern) to WU07AUK occupation sorting. Sweep λ to "
                         "show external data narrows the (aggregate-OD-flat) heterogeneity solution set.")
    ap.add_argument("--occ-ext-delta", type=str,
                    default="0.033,0.130,0.077,-0.019,0.252,0.244,0.435,0.337,0.382",
                    help="WU07AUK per-SOC occupation-sorting delta (external moment target, soc1..9).")
    ap.add_argument("--lambda-class-penalty", type=float, default=0.0,
                    help="Ben-Akiva-Morikawa in-loss external-moment penalty on income/class: anchors "
                         "per-tier alpha_wage (z-scored) to NS-SeC class-wage sorting (monotone increasing: "
                         "higher income tier -> higher-wage destinations). Sweep λ to narrow.")
    ap.add_argument("--class-ext-pattern", type=str, default="mono",
                    help="Income/class external target: 'mono' = monotone increasing wage attraction "
                         "by tier (NS-SeC); or comma-sep z-target values matching n_income_tiers.")
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
    ap.add_argument("--use-transit-commercial-features", action="store_true",
                    help="Add β_transit·log_transit_access_z + β_commercial·log_commercial_z "
                         "to V_other (destination utility). PTAL proxy + commercial land use. "
                         "Both coefs sign-free real. Requires aux npz with log_transit_access_j_z "
                         "and log_commercial_j_z (1725,) — see build_transit_office_features.py.")
    ap.add_argument("--r7-warmup-epochs", type=int, default=0,
                    help="2026-05-22 Round 13 — Sequential training for R7 additions. "
                         "If N>0: first N epochs train Plan E baseline with R7 additions "
                         "(γ_decay, β_tmin, θ_*_dest) FROZEN at init. After ep N: freeze "
                         "Plan E params (all RUM + encoder), unfreeze R7 additions. "
                         "Prevents the multicollinearity train wreck (R7 100% IV inflation) "
                         "by letting baseline converge first, then R7 additions absorb residual. "
                         "Lit anchor: Wang TB-ResNet sequential training (Wang 2021 Essay 3).")
    ap.add_argument("--use-hansen-decay", action="store_true",
                    help="2026-05-22 R7/R8 — Hansen 1959 distance decay in V_dest. "
                         "Adds -γ_decay[tier]·log_d_ij. 3 tier-specific, ≥0 sign-constrained.")
    ap.add_argument("--gamma-decay-init", type=float, default=0.05,
                    help="Hansen decay init per tier (≥0 via softplus).")
    ap.add_argument("--use-tmin-cost", action="store_true",
                    help="2026-05-22 R8 — scalar best-mode time cost in V_dest. "
                         "Adds -β_tmin[tier]·t_min(i, j). 3 tier-specific, ≥0. "
                         "Designed to partially replace nested IV when paired with "
                         "--strip-time-from-vdest-iv (Path B / Wang MTL style).")
    ap.add_argument("--beta-tmin-init", type=float, default=0.01,
                    help="t_min cost init per tier.")
    ap.add_argument("--use-agent-dest-interaction", action="store_true",
                    help="2026-05-22 R7/R8 — agent × destination explicit interactions. "
                         "Adds θ_inc·income_i·log_M_j + θ_kids·kids_i·log_d_ij + θ_cars·cars_i·log_d_ij "
                         "to V_dest. 3 sign-free scalars. Sub-population heterogeneity in V_dest.")
    ap.add_argument("--strip-time-from-vdest-iv", action="store_true",
                    help="2026-05-22 reframing — Component 3 (the real V_lower surgery). "
                         "Compute a 2nd IV from V_lower_quality (= ASC + agent_term, NO β_t·t) "
                         "and use IT (not the time-bearing IV_mode) inside V_dest. Mode "
                         "probability P(m|i,j) STILL uses full IV_mode with time. Decouples "
                         "destination choice from raw travel time entirely — feasibility now "
                         "handled only by --use-frozen-time-mask, IV_quality only carries "
                         "ASC + agent-mode interaction signal (small magnitude → no across-j "
                         "dominance). Pair with --use-frozen-time-mask + --use-quality-features.")
    ap.add_argument("--use-frozen-time-mask", action="store_true",
                    help="2026-05-22 reframing pivot — Component 1 (Feasibility gate). "
                         "Add a HARD per-tier log-mask based on t_min(i, j) vs T_max[tier]. "
                         "Pairs with t_min > T_max[tier] get log_penalty (default -50, near-hard). "
                         "Decouples 'can I reach there' (this mask) from 'how much do I like getting "
                         "there' (residual IV + quality features). Requires --use-consideration-filter "
                         "(piggybacks on log_cost_pass_per_tier path). Uses --T-max-low/mid/high.")
    ap.add_argument("--time-mask-log-penalty", type=float, default=-50.0,
                    help="log-penalty for INFEASIBLE pairs in frozen time mask. Default -50 "
                         "(near-hard, ~e-22 weight). Use -1e9 for fully hard.")
    ap.add_argument("--lambda-max", type=float, default=1.0,
                    help="2026-05-22 reframing pivot — Component 2 (IV down-weight). "
                         "Upper bound on λ_borough ∈ [lambda_eps_min, lambda_max]. "
                         "Default 1.0 keeps Plan E behaviour. Set 0.15 to suppress IV from "
                         "89%% across-j variance to ~15%%, opening V_dest space for A_j + quality.")
    ap.add_argument("--use-quality-features", action="store_true",
                    help="2026-05-22 reframing sanity check: add 3 pair-wise (N, N) quality "
                         "features to V_other (additive, sign-free): n_viable_modes_z + "
                         "log_transit_advantage_z + walk_feasibility_z. Tests whether modal-"
                         "flex / transit-network / walk-access signals carry across-j variance "
                         "beyond what IV (raw travel time) captures. Requires aux npz built by "
                         "data/scripts/build_quality_features.py. Expected: β_nvm > 0 "
                         "(more options = better), β_ta > 0 (transit-favored = better), "
                         "β_wf > 0 (walkable = better).")
    ap.add_argument("--use-attribute-attention", action="store_true",
                    help="Origin-conditioned attribute modulator: per-origin MLP from "
                         "i_emb produces 4 deltas applied to γ_M / α_W / ν_D / δ_match. "
                         "γ_M, α_W, δ_match use multiplicative exp(Δ) (keeps ≥0); ν_D "
                         "uses additive Δ — requires --nu-D-sign-free for true sign-free "
                         "ν_D. Targets multicollinearity ceiling found in direction 1 "
                         "additive features (PTAL/commercial) experiment.")
    ap.add_argument("--attn-emb-dim", type=int, default=32,
                    help="Per-origin embedding width for attribute attention. Default 32.")
    ap.add_argument("--attn-hidden", type=int, default=64,
                    help="MLP hidden width for attribute attention (2 hidden layers). Default 64.")
    ap.add_argument("--attn-dropout", type=float, default=0.1,
                    help="Dropout between MLP layers in attribute attention. Default 0.1.")
    ap.add_argument("--attn-weight-decay", type=float, default=1e-4,
                    help="Weight decay on attribute attention params only (separate group). "
                         "Default 1e-4 — moderate regularization on ~62k attention params.")
    ap.add_argument("--attn-max-log-mul", type=float, default=1.0,
                    help="Bound on multiplicative modulator for γ_M / α_W / δ_match: "
                         "exp(Δ) ∈ [exp(-X), exp(+X)] via tanh(Δ_raw)·X. Default 1.0 → "
                         "[0.37, 2.72]. Set higher only if you want more amplification; "
                         "2026-05-22 unbounded run hit exp(60)=1e26 and collapsed.")
    ap.add_argument("--attn-max-add", type=float, default=1.0,
                    help="Bound on additive modulator for ν_D (sign-free): Δ ∈ [-X, +X]. "
                         "Default 1.0 → ν_D can shift by ±1, enough to flip sign from "
                         "the typical ν ≈ -0.1 base.")
    ap.add_argument("--nu-D-sign-free", action="store_true",
                    help="Bypass -softplus on ν_D and use a raw learnable parameter so the "
                         "competition coefficient can flip sign if data demands. Cervero-Shen "
                         "lit-anchor (ν<0) is relaxed. Typical pairing with attribute attention "
                         "for the 'let data decide' design choice.")
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
    # --- Plan E: commute-hour slicing + NTS-weighted loss ---
    # GEODS 2019 OD has no trip purpose. To approximate commute-only training,
    # we (a) zero-mask non-commute hours and (b) re-weight remaining hours by
    # NTS commute-departure share (P(h|commute) as proxy for P(commute|h)).
    # Math equivalence: L_weighted = Σ_h w_h · CE_h. Lit anchors:
    #   - Iqbal et al. 2014 (TRC) — AM-peak slicing
    #   - Toole et al. 2015 (TRC) — survey-fusion calibration
    ap.add_argument("--commute-hours", type=str, default="",
                    help="Comma-separated hour list to keep in loss (e.g. '7,8,9,17,18'). "
                         "Other hours get weight=0 (hard slice). Empty string = keep all 24h.")
    ap.add_argument("--use-nts-weighted-loss", action="store_true",
                    help="Within kept commute_hours, weight each hour by NTS commute-departure "
                         "share (P(start hour = h | trip = commute)). Without this flag, all "
                         "kept hours get weight 1/|kept| (uniform).")
    ap.add_argument("--nts-share-csv", type=str,
                    default=str(V3_ROOT / "data" / "processed" / "nts_commute_departure_time.csv"),
                    help="Path to NTS commute departure-time share CSV. "
                         "Source: DfT NTS0502 (assets.publishing.service.gov.uk).")
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--kl-warmup-epochs", type=int, default=15)
    ap.add_argument("--lr-theta", type=float, default=1e-3)
    ap.add_argument("--lr-rum", type=float, default=1e-2)
    ap.add_argument("--lambda-init", type=float, default=0.95)
    ap.add_argument("--lambda-eps-min", type=float, default=0.05)
    ap.add_argument("--aux-path", default="data/processed/paperA_v3_aux_cervero.npz")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="evaluation_outputs/paper_a/cervero_shen_smoke.json")
    # --- Profile-likelihood / eval-only mode (2026-05-26) ---
    # Load a trained .pt, skip training, scan each parameter on a grid and record
    # how val NLL / CPC respond. Sharp response = data-identified; flat = not.
    # Default-off: when --load-pt is unset, training behaviour is unchanged.
    ap.add_argument("--load-pt", type=str, default=None,
                    help="Profile mode: path to trained .pt. Skips training, runs "
                         "profile-likelihood scan, writes --profile-out.")
    ap.add_argument("--profile-out", type=str, default=None,
                    help="Output JSON path for profile-likelihood results "
                         "(relative to v3 root). Required with --load-pt.")
    ap.add_argument("--profile-refit-epochs", type=int, default=0,
                    help="If >0, after fixing the scanned param, re-optimise all OTHER "
                         "params for this many epochs (true profile likelihood). "
                         "0 = cheap slice (no refit).")
    ap.add_argument("--profile-params", type=str, default="all",
                    help="Comma-separated param tags to scan (T_max,alpha_W,nu_D,"
                         "gamma_decay,delta_match,match_thresh,gamma_M_tier), or 'all'.")
    ap.add_argument("--use-occ-specific-mass", action="store_true",
                    help="Replace γ_M·log_M_j + δ_match·log_dshare with γ_M·log(M_j·dshare_s) "
                         "= occupation-specific job mass as the gravity attractor. Requires "
                         "--use-soc-mixture. Tests whether occupation matching, given full "
                         "gravity weight, helps or hurts aggregate flow prediction.")
    ap.add_argument("--profile-zero-nn", action="store_true",
                    help="Before profiling, set the GNN residual/blend scale to ~0 "
                         "(remove V_NN from V_dest). Tests whether the NN was absorbing "
                         "the structural signal that would otherwise identify RUM params.")
    # --- Income x commute MOMENT INJECTION (2026-06-18, SIGSPATIAL positive result) ---
    # Aggregate OD leaves per-tier commute-tolerance T_max FLAT (unidentified). Inject an
    # EXTERNAL income x commute-time moment (NTS, 3 numbers = mean commute minutes per
    # income tier) and re-identify raw_T_max_per_tier from it: freeze everything else
    # (incl. gamma_M and the NN), fit only T_max so predicted per-tier commute time
    # matches the target. Constructive half of the conditions-for-identifiability story.
    ap.add_argument("--inject-income-moment", type=str, default=None,
                    help="Path to .npz with key 'target_min' (K_tier mean commute minutes "
                         "by income tier). Freezes all params except raw_T_max_per_tier and "
                         "fits it to match predicted per-tier commute time. Requires --load-pt.")
    ap.add_argument("--inject-param", type=str, default="decay",
                    choices=["gammaM", "decay", "tmin", "kink", "tmax"],
                    help="Per-tier lever to fit to the moment: decay=raw_gamma_decay "
                         "(Hansen distance decay, continuous, default), tmin=raw_beta_tmin, "
                         "kink=raw_beta_t_kink_per_tier, tmax=raw_T_max_per_tier "
                         "(saturated hard gate, diagnostic only).")
    ap.add_argument("--inject-steps", type=int, default=300)
    ap.add_argument("--inject-lr", type=float, default=0.05)
    ap.add_argument("--inject-out", type=str, default=None,
                    help="Output JSON for injection result (relative to v3 root).")
    # --- Occupation x industry MOMENT INJECTION (2026-06-22, SIGSPATIAL positive result) ---
    # Aggregate OD leaves occupation matching delta_match FLAT (occupation-blind flows).
    # Inject an EXTERNAL, published, AGGREGATE statistic the OD does not contain -- the
    # national P(industry|occupation) from ONS APS (build_occ_industry_target.py) -- and
    # re-identify delta_match: freeze everything else (incl. gamma_M, T_max, the NN), fit
    # only raw_delta_match so the model's predicted occupation->industry landing
    # distribution matches the target. Occupation analogue of --inject-income-moment.
    ap.add_argument("--inject-occ-moment", type=str, default=None,
                    help="Path to .npz with key 'target_pio' (S x n_ind, P(industry|occupation)). "
                         "Freezes all params except raw_delta_match and fits it to match the "
                         "predicted per-SOC destination-industry distribution. Requires --load-pt "
                         "+ --use-soc-mixture.")
    ap.add_argument("--inject-occ-steps", type=int, default=300)
    ap.add_argument("--inject-occ-lr", type=float, default=0.05)
    ap.add_argument("--inject-occ-loss", type=str, default="kl", choices=["kl", "relmse"],
                    help="Moment-matching loss: kl = sum_o KL(target_o || pred_o) (default), "
                         "relmse = sum ((pred-target)/target)^2.")
    ap.add_argument("--inject-occ-thresh", action="store_true",
                    help="Two-lever recovery: also free the per-SOC consideration-filter "
                         "threshold (raw_match_filter_thresh_per_soc) alongside delta_match. "
                         "Gives moderate, interpretable magnitudes and a lower residual than "
                         "delta_match alone.")
    ap.add_argument("--inject-occ-out", type=str, default=None,
                    help="Output JSON for occupation injection result (relative to v3 root).")
    # Hessian spectrum: second-order identifiability diagnostic at the optimum.
    ap.add_argument("--hessian-spectrum", action="store_true",
                    help="Compute Hessian of val NLL w.r.t. behavioral coefficients "
                         "(V_NN cached/detached) and report eigenvalue spectrum + per-coef "
                         "diagonal curvature. Requires --load-pt.")
    ap.add_argument("--hessian-params", type=str,
                    default="raw_gamma_M,raw_delta_match,raw_gamma_decay,raw_alpha_wage",
                    help="Comma-separated raw RUM param names to include in the Hessian block.")
    ap.add_argument("--hessian-mask", type=str, default="val", choices=["val", "train"],
                    help="Which split to evaluate NLL on for the Hessian (default val).")
    ap.add_argument("--hessian-eps", type=float, default=1e-3,
                    help="Relative finite-difference step for the Hessian (central diff).")
    ap.add_argument("--hessian-out", type=str, default=None,
                    help="Output JSON for Hessian spectrum (relative to v3 root).")
    # Synthetic collinearity sweep: identifiability threshold for occupation matching.
    ap.add_argument("--synth-collin-sweep", action="store_true",
                    help="Sweep cross-SOC demand-profile collinearity; at each level "
                         "generate OD from a known delta_match and recover it. Requires --load-pt.")
    ap.add_argument("--synth-betas", type=str, default="0.2,0.4,0.6,1.0,1.6,2.6",
                    help="Comma-separated cross-SOC spread multipliers (beta<1 = more "
                         "collinear, >1 = less). beta=1 reproduces the dataset's collinearity.")
    ap.add_argument("--synth-recover-steps", type=int, default=120,
                    help="Adam steps to recover delta_match from random init per collinearity level.")
    ap.add_argument("--synth-out", type=str, default=None,
                    help="Output JSON for the synthetic sweep (relative to v3 root).")
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

    # Optional destination features: transit access proxy (PTAL) + commercial land use
    if args.use_transit_commercial_features:
        if "log_transit_access_j_z" not in aux.files or "log_commercial_j_z" not in aux.files:
            raise RuntimeError(
                "--use-transit-commercial-features requires log_transit_access_j_z and "
                "log_commercial_j_z in aux npz. Run data/scripts/build_transit_office_features.py."
            )
        log_transit_access_z = torch.from_numpy(aux["log_transit_access_j_z"]).float()
        log_commercial_z = torch.from_numpy(aux["log_commercial_j_z"]).float()
        print(f"  transit/commercial features loaded: log_transit_access_j_z "
              f"(std {float(log_transit_access_z.std()):.3f}), "
              f"log_commercial_j_z (std {float(log_commercial_z.std()):.3f})")
    else:
        log_transit_access_z = None
        log_commercial_z = None

    # Pair-wise quality features (2026-05-22 reframing sanity check)
    if args.use_quality_features:
        for k in ("quality_n_viable_modes_z", "quality_log_transit_adv_z",
                  "quality_walk_feasibility_z"):
            assert k in aux.files, (
                f"--use-quality-features requires {k} in aux npz. "
                "Run data/scripts/build_quality_features.py."
            )
        quality_nvm_z = torch.from_numpy(aux["quality_n_viable_modes_z"]).float()
        quality_ta_z = torch.from_numpy(aux["quality_log_transit_adv_z"]).float()
        quality_wf_z = torch.from_numpy(aux["quality_walk_feasibility_z"]).float()
        print(f"  quality features loaded: n_viable_modes (std {float(quality_nvm_z.std()):.3f}), "
              f"log_transit_adv (std {float(quality_ta_z.std()):.3f}), "
              f"walk_feasibility (std {float(quality_wf_z.std()):.3f})")
    else:
        quality_nvm_z = None
        quality_ta_z = None
        quality_wf_z = None

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
        # Destination industry composition (N, n_ind): used by the occupation x industry
        # moment injection to map per-SOC predicted dest flow -> predicted P(industry|occ).
        grid_industry_prop_t = torch.from_numpy(grid_ind_np.astype(np.float32))
        print(f"  per-SOC: soc_props {tuple(soc_props_per_origin.shape)}, "
              f"demand_share_j {tuple(per_soc_demand_share_j.shape)}; "
              f"demand_share range [{per_soc_demand_share_j.min():.3f}, {per_soc_demand_share_j.max():.3f}], "
              f"mean per SOC = {per_soc_demand_share_j.mean(0).tolist()}")
    else:
        soc_props_per_origin = None
        per_soc_demand_share_j = None
        grid_industry_prop_t = None
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
    if log_transit_access_z is not None:
        log_transit_access_z = log_transit_access_z.to(device)
        log_commercial_z = log_commercial_z.to(device)
    if quality_nvm_z is not None:
        quality_nvm_z = quality_nvm_z.to(device)
        quality_ta_z = quality_ta_z.to(device)
        quality_wf_z = quality_wf_z.to(device)
    income_tier_props = income_tier_props.to(device)
    if soc_props_per_origin is not None:
        soc_props_per_origin = soc_props_per_origin.to(device)
        per_soc_demand_share_j = per_soc_demand_share_j.to(device)
        if grid_industry_prop_t is not None:
            grid_industry_prop_t = grid_industry_prop_t.to(device)
    pi_m_pair = pair_mode_share.to(device).float()
    grid_borough_idx = grid_borough_idx.to(device)

    t_per_mode = {}
    for m in mode_names:
        key = "t_" + m
        tens = d[key].to(device).float()
        if tens.dim() == 2:
            tens = tens.unsqueeze(0).expand(T, N, N).contiguous()
        t_per_mode[m] = tens

    # ---- Frozen time mask (2026-05-22 reframing Component 1) ----
    # Hard / near-hard feasibility gate: pairs with t_min(i, j) > T_max[tier] are
    # excluded (log_mask = -50 → softmax ~ e-22 weight). Decouples "can I reach"
    # from "do I want to". Static t_per_mode → mask is also static (K_tier, N, N).
    log_time_mask_per_tier = None
    if args.use_frozen_time_mask:
        assert args.use_consideration_filter, \
            "--use-frozen-time-mask requires --use-consideration-filter (merges into log_cost_pass_per_tier path)"
        with torch.no_grad():
            t_min_static = torch.stack(
                [t_per_mode[m][0] for m in mode_names], dim=0
            ).min(dim=0).values                                              # (N, N)
            T_max_per_tier_t = torch.tensor(
                [args.T_max_low, args.T_max_mid, args.T_max_high][:args.n_income_tiers],
                device=device, dtype=torch.float32,
            )
            is_feasible = t_min_static.unsqueeze(0) <= T_max_per_tier_t.view(-1, 1, 1)  # (K_tier, N, N)
            log_time_mask_per_tier = torch.where(
                is_feasible,
                torch.zeros_like(t_min_static).unsqueeze(0).expand_as(is_feasible),
                torch.full_like(t_min_static, float(args.time_mask_log_penalty))
                     .unsqueeze(0).expand_as(is_feasible),
            ).contiguous()                                                   # (K_tier, N, N)
            print(f"        frozen time mask (Component 1):")
            for k in range(log_time_mask_per_tier.shape[0]):
                frac = float(is_feasible[k].float().mean())
                t_med = float(t_min_static[is_feasible[k]].median()) if is_feasible[k].any() else 0.0
                print(f"          tier {k}: T_max={float(T_max_per_tier_t[k]):.0f} min, "
                      f"feasible {frac*100:.1f}%, median t_min within set {t_med:.1f} min")

    # ---- Plan E: hour_weights tensor for commute-pure loss focus ----
    # Empty --commute-hours and no --use-nts-weighted-loss → None → legacy
    # behaviour (uniform all-hours, exactly equivalent to pre-Plan-E training).
    hour_weights = None
    commute_hours_list = []
    if args.commute_hours.strip():
        commute_hours_list = [int(h) for h in args.commute_hours.split(",") if h.strip()]
        assert all(0 <= h < T for h in commute_hours_list), \
            f"--commute-hours must be in [0, {T}); got {commute_hours_list}"
    if commute_hours_list or args.use_nts_weighted_loss:
        hour_weights_np = np.zeros(T, dtype=np.float32)
        if args.use_nts_weighted_loss:
            import csv as _csv
            nts_share_by_hour = {}
            with open(args.nts_share_csv, "r") as f:
                for ln in f:
                    if ln.startswith("#") or ln.startswith("mode"):
                        continue
                    parts = ln.strip().split(",")
                    if len(parts) != 3:
                        continue
                    mode_tag, h_str, share_str = parts
                    if mode_tag != "all":
                        continue
                    nts_share_by_hour[int(h_str)] = float(share_str)
            assert len(nts_share_by_hour) == 24, \
                f"--nts-share-csv must have 24 hours for mode=all; got {len(nts_share_by_hour)}"
            kept = commute_hours_list if commute_hours_list else list(range(T))
            for h in kept:
                hour_weights_np[h] = nts_share_by_hour[h]
            # Re-normalize so weights sum to len(kept) — preserves loss scale.
            s = hour_weights_np.sum()
            assert s > 0, "NTS weights summed to 0 over kept commute_hours"
            hour_weights_np *= len(kept) / s
        else:
            for h in commute_hours_list:
                hour_weights_np[h] = 1.0   # uniform within kept hours

        hour_weights = torch.from_numpy(hour_weights_np).to(device).float()
        nz = hour_weights_np > 0
        print(f"  Plan E hour weights: kept={int(nz.sum())}/24 hours, "
              f"weights[active]={hour_weights_np[nz].tolist()}")
        # Diagnostic: what fraction of total OD flow is kept?
        with torch.no_grad():
            flow_total = observed_OD.sum().item()
            flow_kept = (observed_OD * hour_weights.view(T, 1, 1)).sum().item()
            print(f"    weighted OD flow fraction: {flow_kept/flow_total:.3f} "
                  f"(raw kept-hour fraction: {observed_OD[nz].sum().item()/flow_total:.3f})")

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
        lambda_max=args.lambda_max,
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
        use_transit_commercial_features=args.use_transit_commercial_features,
        use_quality_features=args.use_quality_features,
        use_hansen_decay=args.use_hansen_decay,
        gamma_decay_init=args.gamma_decay_init,
        use_tmin_cost=args.use_tmin_cost,
        beta_tmin_init=args.beta_tmin_init,
        use_agent_dest_interaction=args.use_agent_dest_interaction,
        use_attribute_attention=args.use_attribute_attention,
        attn_n_origins=N if args.use_attribute_attention else 0,
        attn_emb_dim=args.attn_emb_dim,
        attn_hidden=args.attn_hidden,
        attn_dropout=args.attn_dropout,
        attn_max_log_mul=args.attn_max_log_mul,
        attn_max_add=args.attn_max_add,
        nu_D_sign_free=args.nu_D_sign_free,
    ).to(device)
    # Occupation-specific job mass reformulation (set as attribute; forward_cs reads it).
    rum.use_occ_specific_mass = bool(args.use_occ_specific_mass)
    if rum.use_occ_specific_mass:
        assert args.use_soc_mixture, "--use-occ-specific-mass requires --use-soc-mixture"
        print("        occupation-specific mass: V_M = γ_M·log(M_j·demand_share_s) "
              "(occupation baked into gravity)")

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

    # Split RUM params: attribute-attention params get their own weight_decay group,
    # all other RUM params remain wd=0 (preserves prior behaviour exactly when
    # use_attribute_attention=False — rum_attn is empty in that case).
    def _split_rum_params(only_requires_grad=False):
        attn_ids = set()
        if getattr(rum, "attribute_attention", None) is not None:
            attn_ids = {id(p) for p in rum.attribute_attention.parameters()}
        rum_other, rum_attn = [], []
        for p in rum.parameters():
            if only_requires_grad and not p.requires_grad:
                continue
            (rum_attn if id(p) in attn_ids else rum_other).append(p)
        return rum_other, rum_attn

    _rum_other, _rum_attn = _split_rum_params()
    _opt_groups = [
        {"params": encoder.parameters(), "lr": args.lr_theta, "weight_decay": 1e-4},
        {"params": _rum_other, "lr": args.lr_rum, "weight_decay": 0.0},
    ]
    if _rum_attn:
        _opt_groups.append({"params": _rum_attn, "lr": args.lr_rum,
                            "weight_decay": float(args.attn_weight_decay)})
        print(f"        attribute attention: {sum(p.numel() for p in _rum_attn)} params, "
              f"wd={args.attn_weight_decay}")
    optimizer = torch.optim.AdamW(_opt_groups)

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

    # R7 staged training infra (2026-05-22 Round 13).
    # If --r7-warmup-epochs N > 0: stage 0 (eps 0..N) trains Plan E baseline with R7
    # additions frozen at init values. Stage 1 (eps N..end) freezes Plan E baseline
    # and unfreezes only R7 additions. Prevents multicollinearity train wreck.
    R7_PARAM_NAMES = (
        "raw_gamma_decay",
        "raw_beta_tmin",
        "theta_inc_dest",
        "theta_kids_dest",
        "theta_cars_dest",
        "beta_nvm", "beta_ta", "beta_wf",  # quality features (if on)
    )
    def _is_r7_addition(param_name: str) -> bool:
        return any(tag in param_name for tag in R7_PARAM_NAMES)

    def _apply_r7_stage(stage: str):
        """stage = 'baseline' (Plan E learn, R7 frozen) OR 'r7' (Plan E frozen, R7 learn)."""
        if stage == "baseline":
            for name, p in rum.named_parameters():
                p.requires_grad = (not _is_r7_addition(name))
            for p in encoder.parameters():
                p.requires_grad = True
            print(f"        [r7-staging] Stage BASELINE: R7 additions frozen, Plan E + encoder trainable")
        elif stage == "r7":
            for name, p in rum.named_parameters():
                p.requires_grad = _is_r7_addition(name)
            for p in encoder.parameters():
                p.requires_grad = False
            print(f"        [r7-staging] Stage R7-ONLY: Plan E + encoder frozen, R7 additions trainable")
        n_trainable = sum(p.numel() for p in rum.parameters() if p.requires_grad)
        print(f"        [r7-staging] trainable RUM params: {n_trainable}")

    if args.r7_warmup_epochs > 0:
        assert args.train_mode == "joint", \
            "--r7-warmup-epochs requires --train-mode joint (mutually exclusive with --train-mode sequential)"
        _apply_r7_stage("baseline")

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
    # ===================================================================
    # Profile-likelihood / eval-only mode (2026-05-26).
    # Load trained .pt, skip training, scan each param on a grid in EFFECTIVE
    # space (multiplicative around fitted value), record val NLL / CPC response.
    # Cheap "slice" (refit-epochs=0): hold all other params fixed. A FLAT slice
    # is strong evidence of non-identification; a SHARP slice is necessary but
    # not sufficient (others may compensate) -> confirm with --profile-refit-epochs.
    # ===================================================================
    if args.load_pt is not None:
        import json as _pjson
        assert (args.profile_out or args.inject_income_moment or args.inject_occ_moment
                or args.hessian_spectrum or args.synth_collin_sweep), \
            "--load-pt requires --profile-out, --inject-income-moment, --inject-occ-moment, " \
            "--hessian-spectrum, or --synth-collin-sweep"
        _ckpt = torch.load(args.load_pt, map_location=device, weights_only=False)
        encoder.load_state_dict(_ckpt["encoder_state"])
        rum.load_state_dict(_ckpt["rum_state"])
        encoder.eval(); rum.eval()

        if args.profile_zero_nn:
            # Remove V_NN from V_dest so we can see whether RUM params regain
            # leverage on NLL absent the NN (tests NN-absorption hypothesis).
            _off = math.log(math.exp(1e-4) - 1.0)  # inv_softplus(1e-4) ~= -9.21
            with torch.no_grad():
                if getattr(rum, "raw_gnn_residual_scale", None) is not None:
                    rum.raw_gnn_residual_scale.fill_(_off)
                if getattr(rum, "raw_gnn_blend", None) is not None:
                    rum.raw_gnn_blend.fill_(-15.0)
            print("[profile] ZERO-NN mode: V_NN removed from V_dest", flush=True)

        # Cache V_NN once: the encoder (GNN) output does NOT depend on RUM params,
        # so during profiling (encoder frozen) we compute it a single time and a
        # stub encoder returns the cached tensor — each forward then skips the
        # costly GraphSAGE pass. Exact, and ~50x faster on CPU.
        with torch.no_grad():
            _vnn_cache = encoder(X_static, X_dynamic, edge_index).detach()

        class _StubEncoder:
            def __call__(self, *a, **k):
                return _vnn_cache

            def parameters(self):
                return iter(())

        _enc = _StubEncoder()

        def _profile_fwd(mask=val_mask, tier_commute_probe=False, soc_dest_probe=False,
                         force_no_ckpt=False):
            return forward_cs(
                _enc, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
                tier_commute_probe=tier_commute_probe,
                soc_dest_probe=soc_dest_probe,
                force_no_ckpt=force_no_ckpt,
            )

        def _profile_eval():
            with torch.no_grad():
                _o = _profile_fwd(val_mask)
                return (float(cpc(_o["log_P_D"], observed_OD, val_mask, hour_weights=hour_weights)),
                        float(_o["nll_dest"]))

        def _train_loss_from(out):
            l = (out["nll_dest"]
                 + args.lambda_kl * out["ce_mode"]
                 + args.lambda_nn_norm * out["nn_norm_sq"]
                 + args.lambda_ortho * out["ortho_cos_sq"]
                 + args.lambda_match_ortho * out["match_ortho_cos_sq"]
                 + args.lambda_iv_balance * (out["iv_share_across_j"] - args.target_iv_share) ** 2)
            if rum.use_self_loop_boost and args.self_loop_l2 > 0:
                l = l + args.self_loop_l2 * (rum.self_loop_boost ** 2).mean()
            if rum.n_busy_dest > 0 and args.busy_dest_l2 > 0:
                l = l + args.busy_dest_l2 * (rum.busy_dest_boost ** 2).mean()
            return l

        _REFIT = int(args.profile_refit_epochs)

        def _refit_eval(p, idxs, kind, effs):
            # TRUE profile likelihood: pin the scanned indices at `effs`, then
            # re-optimize ALL OTHER rum params (V_NN held fixed) for _REFIT steps,
            # and evaluate on val. A param that stays sharp after others are free
            # to compensate is identified; one that flattens is not.
            snap = {n: t.detach().clone() for n, t in rum.named_parameters()}
            for k, i in enumerate(idxs):
                _set_eff(p, i, kind, effs[k])
            pinned = [p.data[i].item() for i in idxs]
            if _REFIT > 0:
                rum.train()
                opt = torch.optim.AdamW([q for q in rum.parameters() if q.requires_grad],
                                        lr=args.lr_rum, weight_decay=0.0)
                for _it in range(_REFIT):
                    opt.zero_grad()
                    loss = _train_loss_from(_profile_fwd(train_mask))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(rum.parameters()), 1.0)
                    opt.step()
                    with torch.no_grad():  # re-pin scanned indices after each step
                        for k, i in enumerate(idxs):
                            p.data[i] = pinned[k]
                rum.eval()
            c, nll = _profile_eval()
            with torch.no_grad():
                for n, t in rum.named_parameters():
                    t.data.copy_(snap[n])
            return c, nll

        def _inv_softplus_t(y):  # y > 0 (tensor or float) -> raw such that softplus(raw)=y
            import math as _m
            yv = float(y)
            return yv if yv > 20.0 else _m.log(_m.expm1(yv))

        import time as _time
        _t0 = _time.time()
        base_cpc, base_nll = _profile_eval()
        _fwd_s = _time.time() - _t0
        print(f"[profile] loaded {args.load_pt}", flush=True)
        print(f"[profile] BASELINE reproduce: CPC={base_cpc:.4f} (ckpt {_ckpt['final_cpc']:.4f}, "
              f"diff {base_cpc-_ckpt['final_cpc']:+.4f}) | NLL={base_nll:.4f} | "
              f"forward={_fwd_s:.1f}s", flush=True)

        # ---- HESSIAN SPECTRUM (identifiability via local curvature) ----
        # Second-order companion to the profile likelihood. At the fitted optimum,
        # form the Hessian of the validation NLL w.r.t. the behavioral coefficients
        # (V_NN cached & detached, so no GNN double-backward). Near-zero eigenvalues
        # are flat directions = non-identified combinations; the per-coefficient
        # diagonal curvature maps onto the profile-likelihood table (identified =
        # sharp curvature, flat ~ 0). Coordinates are the raw (unconstrained)
        # parametrization; flat directions are coordinate-covariant (a genuinely
        # flat direction has ~0 curvature in any smooth reparametrization).
        if getattr(args, "hessian_spectrum", False):
            import json as _hjson
            groups = [g.strip() for g in (args.hessian_params or
                      "raw_gamma_M,raw_delta_match,raw_gamma_decay,raw_alpha_wage").split(",") if g.strip()]
            sel, blocks = [], []   # (name, start, len)
            off = 0
            for g in groups:
                t = getattr(rum, g, None)
                if t is None:
                    print(f"[hessian] skip missing param '{g}'", flush=True); continue
                sel.append(t); blocks.append((g, off, t.numel())); off += t.numel()
            assert sel, "no behavioral params found for hessian"
            for p in rum.parameters():
                p.requires_grad_(False)
            for t in sel:
                t.requires_grad_(True)
            rum.eval()
            _hmask = train_mask if getattr(args, "hessian_mask", "val") == "train" else val_mask
            n = int(off)
            print(f"[hessian] params={[b[0] for b in blocks]} dim={n} mask={args.hessian_mask} "
                  f"(finite-diff gradient, memory-frugal: no create_graph)", flush=True)
            # global flat index -> (tensor, local linear index)
            idxmap = []
            for t in sel:
                for j in range(t.numel()):
                    idxmap.append((t, j))

            def _grad_vec():
                for t in sel:
                    if t.grad is not None:
                        t.grad = None
                nll = _profile_fwd(_hmask)["nll_dest"]
                g = torch.autograd.grad(nll, sel)            # single backward, no create_graph
                return torch.cat([gi.reshape(-1) for gi in g]).detach()

            EPS = float(getattr(args, "hessian_eps", 1e-3))
            H = torch.zeros(n, n, device=device)
            for i in range(n):
                t, j = idxmap[i]
                orig = t.data.view(-1)[j].item()
                h = EPS * (1.0 + abs(orig))                   # relative step
                with torch.no_grad():
                    t.data.view(-1)[j] = orig + h
                gp = _grad_vec()
                with torch.no_grad():
                    t.data.view(-1)[j] = orig - h
                gm = _grad_vec()
                with torch.no_grad():
                    t.data.view(-1)[j] = orig            # restore exactly
                H[:, i] = (gp - gm) / (2.0 * h)
                if (i + 1) % 10 == 0:
                    print(f"[hessian]   col {i+1}/{n}", flush=True)
            H = 0.5 * (H + H.T)                              # symmetrize (FD noise)
            evals = torch.linalg.eigvalsh(H)                       # ascending
            evals_l = [float(x) for x in evals]
            diag = torch.diagonal(H)
            # per-named-coefficient summary (mean & max diagonal curvature within block)
            FLAT_EIG = 1e-3 * max(1.0, float(evals.max()))         # relative flat threshold
            n_flat = int((evals < FLAT_EIG).sum())
            # top eigenvector: which raw coefficients dominate the stiffest direction
            evecs = torch.linalg.eigh(H).eigenvectors
            top = evecs[:, -1].abs()
            coef_diag, coef_top = {}, {}
            for (gname, s, ln) in blocks:
                blk = diag[s:s + ln]
                coef_diag[gname] = {"mean": float(blk.mean()), "max": float(blk.max()),
                                    "min": float(blk.min())}
                coef_top[gname] = float(top[s:s + ln].pow(2).sum())   # share of stiff direction
                # gamma_M is tier-major (tier*S + soc): report per-tier diagonal
                if gname == "raw_gamma_M":
                    S = rum.n_soc
                    coef_diag[gname]["per_tier_mean"] = [float(blk.view(-1, S)[k].mean())
                                                         for k in range(blk.numel() // S)]
            res = {"ckpt": args.load_pt, "mask": args.hessian_mask, "dim": n,
                   "base_nll": float(base_nll), "base_cpc": float(base_cpc),
                   "eigenvalues": evals_l,
                   "eig_max": float(evals.max()), "eig_min": float(evals.min()),
                   "n_flat_directions": n_flat, "flat_eig_threshold": float(FLAT_EIG),
                   "condition_number": float(evals.max() / evals.clamp_min(1e-12).min()),
                   "coef_diag_curvature": coef_diag,
                   "coef_share_of_stiff_direction": coef_top,
                   "blocks": [(b[0], b[1], b[2]) for b in blocks]}
            print(f"[hessian] eig range [{evals.min():.3e}, {evals.max():.3e}]  "
                  f"n_flat(<{FLAT_EIG:.2e})={n_flat}/{n}  cond={res['condition_number']:.2e}", flush=True)
            for gname in coef_diag:
                print(f"[hessian]   {gname}: diag mean={coef_diag[gname]['mean']:.3e} "
                      f"max={coef_diag[gname]['max']:.3e}  stiff-share={coef_top[gname]:.3f}", flush=True)
            if getattr(args, "hessian_out", None):
                _hp = V3_ROOT / args.hessian_out
                _hp.parent.mkdir(parents=True, exist_ok=True)
                _hjson.dump(res, open(_hp, "w"), indent=2)
                print(f"[hessian] wrote {_hp}", flush=True)
            return

        # ---- SYNTHETIC COLLINEARITY SWEEP (identifiability threshold) ----
        # Constructive condition for OCCUPATION heterogeneity: control the
        # cross-SOC demand-profile collinearity, generate OD from a KNOWN
        # delta_match, then recover delta_match from a random init. Recovery
        # succeeds (low error) only below a collinearity threshold -> turns the
        # "necessary but not sufficient" resolution result into a sufficient,
        # quantified condition: identifiable iff demand-profile collinearity < X.
        if getattr(args, "synth_collin_sweep", False):
            import json as _sjson
            assert per_soc_demand_share_j is not None, "synth sweep needs per_soc_demand_share_j"
            S = rum.n_soc

            def _set_delta(vals):
                with torch.no_grad():
                    raw = torch.log(torch.expm1(vals.clamp(min=1e-4)))   # inv-softplus
                    rum.raw_delta_match.copy_(raw)

            # designed TRUE occupation-matching gradient: clear, monotone, >0
            true_vals = torch.linspace(0.3, 1.7, S, device=device)
            _set_delta(true_vals)
            delta_true = rum.delta_match.detach().clone()

            base_ds = per_soc_demand_share_j.detach().clone()            # (N, S)
            mbar = base_ds.mean(dim=1, keepdim=True)                     # (N, 1) cross-SOC mean

            def _collin(ds):                # max |off-diag corr| between SOC dest profiles (cols)
                X = ds.t()                  # (S, N)
                Xc = X - X.mean(dim=1, keepdim=True)
                C = Xc @ Xc.t()
                d = torch.sqrt(torch.diag(C).clamp_min(1e-12))
                R = C / (d.view(-1, 1) * d.view(1, -1))
                return float((R - torch.eye(S, device=ds.device)).abs().max())

            def _synth_fwd(ds, od, mask):
                return forward_cs(
                    _enc, rum, X_static, X_dynamic, edge_index,
                    t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
                    income_score, pct_kids, mean_cars, income_tier_props,
                    pi_m_pair, grid_borough_idx, od, mask,
                    soc_props_per_origin=soc_props_per_origin,
                    per_soc_demand_share_j=ds, hour_weights=hour_weights,
                    log_transit_access_z=log_transit_access_z,
                    log_commercial_z=log_commercial_z,
                    quality_nvm_z=quality_nvm_z, quality_ta_z=quality_ta_z,
                    quality_wf_z=quality_wf_z,
                    log_time_mask_per_tier=log_time_mask_per_tier,
                    strip_time_from_vdest_iv=args.strip_time_from_vdest_iv)

            # Collinearity knob: blend each SOC's destination profile toward a COMMON
            # reference column R (= cross-SOC mean profile). lambda->1 makes all SOC
            # profiles identical (collinear); lambda<0 spreads them apart (less collinear).
            # NB: do NOT row-renormalize -- that turns the blend into a global scale that
            # cancels in the correlation (the v1 bug). Demand share enters as a per-SOC
            # log attractiveness, so the destination softmax tolerates an unnormalized level.
            R_ref = base_ds.mean(dim=1, keepdim=True)                   # (N,1) common profile
            lambdas = [float(b) for b in (args.synth_betas or
                       "-0.6,0.0,0.5,0.8,0.94,0.99").split(",")]
            OD_tot = observed_OD.sum(dim=2, keepdim=True)               # (T,N,1) origin totals
            R_STEPS = int(args.synth_recover_steps)
            # destination-utility params free to COMPENSATE during recovery (mirrors the
            # profile-with-refit test): collinear profiles let these absorb delta_match.
            FREE = [a for a in ["raw_delta_match", "raw_gamma_M", "raw_alpha_wage",
                                "raw_nu_D", "raw_gamma_decay"] if getattr(rum, a, None) is not None]
            _set_delta(true_vals)
            truth_state = {n: t.detach().clone() for n, t in rum.named_parameters()}
            sweep = []
            for lam in lambdas:
                ds_b = (R_ref + (1.0 - lam) * (base_ds - R_ref)).clamp(min=1e-6)
                collin = _collin(ds_b)
                # restore all params to truth, then generate synthetic OD from TRUE delta
                with torch.no_grad():
                    for n, t in rum.named_parameters():
                        t.data.copy_(truth_state[n])
                    P = _synth_fwd(ds_b, observed_OD, train_mask)["log_P_D"].exp()
                    synth_OD = torch.poisson(P * OD_tot).detach()        # finite-sample noise
                # recover: random-init delta_match, free destination-utility params to
                # compensate, fit to noisy synthetic OD
                for p in rum.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    rum.raw_delta_match.copy_(torch.randn_like(rum.raw_delta_match) * 0.5)
                free_params = []
                for a in FREE:
                    getattr(rum, a).requires_grad_(True)
                    free_params.append(getattr(rum, a))
                opt = torch.optim.Adam(free_params, lr=0.05)
                rum.train()
                for _it in range(R_STEPS):
                    opt.zero_grad()
                    loss = _synth_fwd(ds_b, synth_OD, train_mask)["nll_dest"]
                    loss.backward()
                    opt.step()
                rum.eval()
                rec = rum.delta_match.detach().clone()
                mae = float((rec - delta_true).abs().mean())
                relmae = float(((rec - delta_true).abs()
                                / delta_true.abs().clamp_min(1e-3)).mean())
                sweep.append({"lambda": lam, "collinearity": collin, "recover_mae": mae,
                              "recover_relmae": relmae,
                              "recovered": [round(float(x), 4) for x in rec]})
                print(f"[synth] lambda={lam:+.2f} collin={collin:.3f} "
                      f"MAE={mae:.4f} relMAE={relmae:.3f}", flush=True)
            # restore truth at the end
            with torch.no_grad():
                for n, t in rum.named_parameters():
                    t.data.copy_(truth_state[n])
            res = {"ckpt": args.load_pt, "metric": "delta_match_recovery_vs_collinearity",
                   "true_delta": [round(float(x), 4) for x in delta_true],
                   "recover_steps": R_STEPS, "free_params": FREE,
                   "noise": "poisson", "sweep": sweep}
            if getattr(args, "synth_out", None):
                _op = V3_ROOT / args.synth_out
                _op.parent.mkdir(parents=True, exist_ok=True)
                _sjson.dump(res, open(_op, "w"), indent=2)
                print(f"[synth] wrote {_op}", flush=True)
            return

        # ---- Income x commute MOMENT INJECTION (constructive identifiability) ----
        # Reuses _profile_fwd (cached V_NN). Freeze all RUM params except the chosen
        # per-tier time-disutility lever (--inject-param); fit it so predicted per-tier
        # commute time matches an external income x commute-time target. Reports whether
        # the lever pins, predicted time -> target, CPC stays put, gamma_M untouched.
        #   tmin = raw_beta_tmin  (linear per-tier t_min cost; continuous leverage)  [default]
        #   kink = raw_beta_t_kink_per_tier  (Bhat long-commute kink penalty)
        #   tmax = raw_T_max_per_tier  (hard time gate; SATURATED-OPEN -> no leverage; diagnostic only)
        # Observable EXCLUDES self-loops (i==j): NTS measures actual commute journeys,
        # and the model's diagonal (self_loop_boost) otherwise dominates the mean (~5 min).
        if getattr(args, "inject_income_moment", None) is not None:
            import json as _ijson
            _PMAP = {"gammaM": "raw_gamma_M", "decay": "raw_gamma_decay",
                     "tmin": "raw_beta_tmin", "kink": "raw_beta_t_kink_per_tier",
                     "tmax": "raw_T_max_per_tier"}
            _DISP = {"decay": "gamma_decay_per_tier", "tmin": "beta_tmin_per_tier",
                     "kink": "beta_t_kink_per_tier", "tmax": "T_max_per_tier"}
            _pname = args.inject_param
            _raw_attr = _PMAP[_pname]
            _raw = getattr(rum, _raw_attr, None)
            assert _raw is not None, f"model has no {_raw_attr} (inject-param={_pname})"
            K = int(income_tier_props.shape[1])
            _gm_mode = (_pname == "gammaM")   # gravity lever (strong); fit low+mid, freeze high tier
            if _gm_mode:
                _S = rum.n_soc                                  # gamma_M is (K_tier*n_soc,), tier-major
                _hi = list(range((K - 1) * _S, K * _S))         # high-tier class indices (re-pinned)
                _disp = lambda: [round(float(rum.gamma_M.view(K, -1)[k].mean()), 3) for k in range(K)]
            else:
                _disp = lambda: [round(float(x), 3) for x in getattr(rum, _DISP[_pname])]
            _mm = np.load(args.inject_income_moment)
            target = torch.tensor(_mm["target_min"], dtype=torch.float32, device=device)  # (K_tier,)
            assert target.numel() == K, f"target has {target.numel()} entries, model has {K} tiers"
            # Predicted per-tier mean commute time uses the model's PER-CLASS destination
            # distributions (forward_cs tier_commute_probe), NOT the tier-marginal P --
            # the marginal cannot express tier differences. Off-diagonal, hour-weighted.
            def _pred_time_by_tier():
                out = _profile_fwd(train_mask, tier_commute_probe=True)
                tct = out["tier_commute_time"]
                assert tct is not None, "tier_commute_probe returned None (model not tier-mixture?)"
                return tct                                      # (K_tier,) minutes

            for p in rum.parameters():                          # freeze all ...
                p.requires_grad_(False)
            _raw.requires_grad_(True)                            # ... except the chosen lever
            gM0 = rum.gamma_M.detach().clone()
            if _gm_mode:                                         # protect OD-identified high tier
                _gm_hi_orig = _raw.data[_hi].clone()
            with torch.no_grad():
                pt0 = _pred_time_by_tier()
            lev0 = _disp()
            print(f"[inject] param={_pname} ({_raw_attr})  target min (low/mid/high) = "
                  f"{[round(float(x),1) for x in target]}", flush=True)
            print(f"[inject] BEFORE: {_pname}={lev0}  "
                  f"pred_time={[round(float(x),1) for x in pt0]}  CPC={base_cpc:.4f}", flush=True)

            opt = torch.optim.Adam([_raw], lr=float(args.inject_lr))
            rum.train()
            for s in range(int(args.inject_steps)):
                opt.zero_grad()
                pt = _pred_time_by_tier()
                loss = (((pt - target) / target) ** 2).sum()
                loss.backward()
                opt.step()
                if _gm_mode:                       # re-pin high tier (keep identified gamma_M[high])
                    with torch.no_grad():
                        _raw.data[_hi] = _gm_hi_orig
                if (s + 1) % 50 == 0:
                    print(f"[inject] step {s+1}: loss={float(loss):.4f} "
                          f"{_pname}={_disp()}  pred={[round(float(x),1) for x in pt.detach()]}", flush=True)
            rum.eval()
            with torch.no_grad():
                pt1 = _pred_time_by_tier()
            cpc1, nll1 = _profile_eval()
            dG = float((rum.gamma_M.detach() - gM0).abs().max())
            # In gammaM mode, the high tier is re-pinned: report its drift separately
            # (should be ~0 = OD-identified gamma_M[high] protected) vs the fitted low/mid.
            dG_hi = (float((rum.gamma_M.detach().view(K, -1)[K - 1]
                            - gM0.view(K, -1)[K - 1]).abs().max()) if _gm_mode else dG)
            lev1 = _disp()
            print(f"[inject] AFTER:  {_pname}={lev1}  pred_time={[round(float(x),1) for x in pt1]}  "
                  f"CPC={cpc1:.4f} (d{cpc1-base_cpc:+.4f})  "
                  f"gamma_M max|d|={dG:.2e}  high-tier|d|={dG_hi:.2e}", flush=True)
            res = {"ckpt": args.load_pt, "metric": "commute_time_min_offdiag",
                   "inject_param": _pname, "raw_attr": _raw_attr,
                   "target": [float(x) for x in target],
                   "pred_before": [float(x) for x in pt0], "pred_after": [float(x) for x in pt1],
                   "lever_before": lev0, "lever_after": lev1,
                   "cpc_before": float(base_cpc), "cpc_after": float(cpc1),
                   "nll_after": float(nll1), "gamma_M_max_abs_delta": dG,
                   "gamma_M_high_abs_delta": dG_hi, "steps": int(args.inject_steps)}
            if args.inject_out:
                _op = V3_ROOT / args.inject_out
                _op.parent.mkdir(parents=True, exist_ok=True)
                _ijson.dump(res, open(_op, "w"), indent=2)
                print(f"[inject] wrote {_op}", flush=True)
            _ok = (abs(cpc1 - base_cpc) < 0.01) and (dG_hi < 1e-6 if _gm_mode else True)
            print(f"[inject] check: pred {[round(float(x),1) for x in pt0]} -> "
                  f"{[round(float(x),1) for x in pt1]} (target {[round(float(x),1) for x in target]}); "
                  f"CPC {base_cpc:.4f}->{cpc1:.4f}; "
                  f"{'OK: low/mid fit, high+CPC protected' if _ok else 'PARTIAL: inspect deltas'}",
                  flush=True)
            return

        # --- Occupation x industry moment injection: re-identify delta_match ---
        # Aggregate OD leaves delta_match FLAT (occupation-blind). Freeze everything
        # else (incl. gamma_M, T_max, the NN) and fit ONLY raw_delta_match so the model's
        # predicted per-SOC destination-industry distribution P(industry|occupation)
        # matches the external ONS APS moment. Constructive occupation analogue of the
        # income x commute moment injection above.
        if getattr(args, "inject_occ_moment", None) is not None:
            import json as _ojson
            assert rum.use_soc_mixture, "--inject-occ-moment requires --use-soc-mixture"
            _raw_dm = getattr(rum, "raw_delta_match", None)
            assert _raw_dm is not None, "model has no raw_delta_match"
            SOC9 = ["mgr", "prof", "assoc", "admin", "trades", "care", "sales", "oper", "elem"]
            _mm = np.load(args.inject_occ_moment, allow_pickle=True)
            tgt = torch.tensor(_mm["target_pio"], dtype=torch.float32, device=device)   # (S, n_ind)
            S_occ = rum.n_soc
            assert tgt.shape[0] == S_occ, f"target has {tgt.shape[0]} occ rows, model has {S_occ} SOC"
            assert grid_industry_prop_t is not None and grid_industry_prop_t.shape[1] == tgt.shape[1], \
                "grid_industry_prop n_ind must match target_pio n_ind (use the bres18 aux + bres18 target)"
            _gip = grid_industry_prop_t                                                  # (N, n_ind)

            def _pred_occ_industry():
                # Per-SOC predicted dest flow (S, N) -> map to industry via grid mix ->
                # row-normalise to predicted P(industry | occupation) (S, n_ind).
                out = _profile_fwd(train_mask, soc_dest_probe=True, force_no_ckpt=True)
                sf = out["soc_dest_flow"]                                                # (S, N)
                assert sf is not None, "soc_dest_probe returned None (model not soc-mixture?)"
                occ_ind = sf @ _gip                                                      # (S, n_ind)
                return occ_ind / occ_ind.sum(dim=1, keepdim=True).clamp_min(1e-9)

            def _moment_loss(pred):
                if args.inject_occ_loss == "kl":
                    return (tgt * (tgt.clamp_min(1e-9).log() - pred.clamp_min(1e-9).log())).sum()
                return (((pred - tgt) / tgt.clamp_min(1e-3)) ** 2).sum()

            # Freeze everything; optimise the 9 delta_match GRADIENT-FREE over no_grad
            # forward evaluations. (Windows-CPU autograd backward through the 27-class
            # mixture SEGFAULTs — confirmed for both income and occ injections — but the
            # forward is fine, and 9 smooth params optimise quickly with scipy L-BFGS-B.)
            for p in rum.parameters():
                p.requires_grad_(False)
            rum.eval()
            # Levers: delta_match always; + per-SOC consideration-filter threshold if asked.
            # Two occupation knobs let each settle at a moderate, interpretable magnitude
            # (one knob alone gets blown up to ~30) and reach a lower moment residual.
            _use_thresh = bool(getattr(args, "inject_occ_thresh", False))
            _levers = [("raw_delta_match", _raw_dm)]
            if _use_thresh:
                _rt = getattr(rum, "raw_match_filter_thresh_per_soc", None)
                assert _rt is not None, \
                    "--inject-occ-thresh needs raw_match_filter_thresh_per_soc (per-SOC filter)"
                _levers.append(("raw_match_filter_thresh_per_soc", _rt))
            _sizes = [t.numel() for _, t in _levers]
            x0 = np.concatenate([t.detach().cpu().numpy().astype(np.float64).ravel()
                                 for _, t in _levers])
            dm0 = rum.delta_match.detach().clone()
            th0 = rum.match_filter_thresh_per_soc.detach().clone() if _use_thresh else None

            def _set_and_loss(x):
                with torch.no_grad():
                    off = 0
                    for (nm, t), sz in zip(_levers, _sizes):
                        t.data = torch.tensor(x[off:off + sz], dtype=t.dtype,
                                              device=device).view_as(t)
                        off += sz
                    return float(_moment_loss(_pred_occ_industry()))

            loss0 = _set_and_loss(x0)
            print(f"[inject-occ] target P(ind|occ) {tuple(tgt.shape)}  loss={args.inject_occ_loss}  "
                  f"levers={[n for n, _ in _levers]} ({x0.size} params)", flush=True)
            print(f"[inject-occ] BEFORE: delta_match={[round(float(x), 3) for x in dm0]}  "
                  f"moment_loss={loss0:.4f}  CPC={base_cpc:.4f}", flush=True)
            if th0 is not None:
                print(f"[inject-occ] BEFORE: thresh_per_soc={[round(float(x), 4) for x in th0]}",
                      flush=True)

            from scipy.optimize import minimize as _spmin
            _ev = {"n": 0}

            def _obj(x):
                _ev["n"] += 1
                v = _set_and_loss(x)
                if _ev["n"] % 10 == 0:
                    print(f"[inject-occ] eval {_ev['n']}: loss={v:.4f} "
                          f"delta_match={[round(float(z), 3) for z in rum.delta_match.detach()]}",
                          flush=True)
                return v

            _spres = _spmin(_obj, x0, method="L-BFGS-B",
                            options={"eps": 1e-3, "maxiter": int(args.inject_occ_steps),
                                     "maxfun": int(args.inject_occ_steps) * 12, "ftol": 1e-7})
            loss1 = _set_and_loss(_spres.x)
            cpc1, nll1 = _profile_eval()
            dm1 = rum.delta_match.detach().clone()
            th1 = rum.match_filter_thresh_per_soc.detach().clone() if _use_thresh else None
            print(f"[inject-occ] AFTER:  delta_match={[round(float(x), 3) for x in dm1]}  "
                  f"moment_loss={loss1:.4f}  CPC={cpc1:.4f} (d{cpc1-base_cpc:+.4f})  "
                  f"evals={_ev['n']}  converged={bool(_spres.success)}", flush=True)
            if th1 is not None:
                print(f"[inject-occ] AFTER:  thresh_per_soc={[round(float(x), 4) for x in th1]}",
                      flush=True)
            res = {"ckpt": args.load_pt, "metric": "P(industry|occupation)",
                   "loss_kind": args.inject_occ_loss, "soc_labels": SOC9,
                   "levers": [n for n, _ in _levers],
                   "delta_match_before": [float(x) for x in dm0],
                   "delta_match_after": [float(x) for x in dm1],
                   "thresh_before": ([float(x) for x in th0] if th0 is not None else None),
                   "thresh_after": ([float(x) for x in th1] if th1 is not None else None),
                   "moment_loss_before": loss0, "moment_loss_after": loss1,
                   "cpc_before": float(base_cpc), "cpc_after": float(cpc1),
                   "nll_after": float(nll1), "maxiter": int(args.inject_occ_steps),
                   "n_evals": _ev["n"], "scipy_success": bool(_spres.success)}
            if args.inject_occ_out:
                _op = V3_ROOT / args.inject_occ_out
                _op.parent.mkdir(parents=True, exist_ok=True)
                _ojson.dump(res, open(_op, "w"), indent=2)
                print(f"[inject-occ] wrote {_op}", flush=True)
            _ok = abs(cpc1 - base_cpc) < 0.01
            print(f"[inject-occ] check: moment_loss {loss0:.4f}->{loss1:.4f}; "
                  f"CPC {base_cpc:.4f}->{cpc1:.4f}; "
                  f"{'OK: delta_match identified at small CPC cost' if _ok else 'PARTIAL: inspect'}",
                  flush=True)
            return

        # transform kind per raw attr: 'sp' = softplus, 'neg_sp' = -softplus
        SOC9 = ["mgr","prof","assoc","admin","trades","care","sales","oper","elem"]
        TIER3 = ["low","mid","high"]
        # (raw_attr, kind, n, labels, tag, group_size)  group_size=1 => per-index
        specs = []
        def _maybe(attr, kind, n, labels, tag, group=1):
            if getattr(rum, attr, None) is not None:
                specs.append((attr, kind, n, labels, tag, group))
        _maybe("raw_T_max_per_tier", "sp", 3, TIER3, "T_max")
        _maybe("raw_alpha_wage", "sp", 3, TIER3, "alpha_W")
        _maybe("raw_nu_D", "neg_sp", 3, TIER3, "nu_D")
        _maybe("raw_gamma_decay", "sp", 3, TIER3, "gamma_decay")
        _maybe("raw_delta_match", "sp", 9, SOC9, "delta_match")
        _maybe("raw_match_filter_thresh_per_soc", "sp", 9, SOC9, "match_thresh")
        _maybe("raw_gamma_M", "sp", 27, [f"{TIER3[i//9]}.{SOC9[i%9]}" for i in range(27)],
               "gamma_M_tier", group=9)  # scan 3 tier-blocks of 9 together

        if args.profile_params != "all":
            _want = set(args.profile_params.split(","))
            specs = [s for s in specs if s[4] in _want]
            print(f"[profile] filtered to tags: {sorted(_want)} -> {len(specs)} param groups", flush=True)

        mults = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        results = {
            "ckpt": args.load_pt, "ckpt_cpc": float(_ckpt["final_cpc"]),
            "refit_epochs": int(args.profile_refit_epochs),
            "baseline": {"cpc": base_cpc, "nll": base_nll},
            "mults": mults, "scans": [],
        }

        def _set_eff(p, idx, kind, eff):
            with torch.no_grad():
                if kind == "sp":
                    p.data[idx] = _inv_softplus_t(max(eff, 1e-6))
                elif kind == "neg_sp":
                    p.data[idx] = _inv_softplus_t(max(-eff, 1e-6))

        for attr, kind, n, labels, tag, group in specs:
            p = getattr(rum, attr)
            n_scans = n // group
            for gi in range(n_scans):
                idxs = list(range(gi*group, gi*group+group))
                glabel = f"{tag}[{labels[idxs[0]]}]" if group == 1 else f"{tag}[{TIER3[gi]}]"
                orig = [float(p.data[i].item()) for i in idxs]
                # fitted effective per index
                if kind == "sp":
                    fitted_eff = [float(torch.nn.functional.softplus(torch.tensor(o))) for o in orig]
                else:
                    fitted_eff = [-float(torch.nn.functional.softplus(torch.tensor(o))) for o in orig]
                pts = []
                for m in mults:
                    if _REFIT > 0:
                        c, nll = _refit_eval(p, idxs, kind, [fitted_eff[k] * m for k in range(len(idxs))])
                    else:
                        for k, i in enumerate(idxs):
                            _set_eff(p, i, kind, fitted_eff[k] * m)
                        c, nll = _profile_eval()
                    pts.append({"mult": m, "eff_idx0": fitted_eff[0]*m, "cpc": c, "nll": nll})
                    print(f"[profile]      {glabel:>20s} mult={m:.2f} -> cpc={c:.4f} nll={nll:.4f}", flush=True)
                for k, i in enumerate(idxs):  # restore
                    with torch.no_grad():
                        p.data[i] = orig[k]
                nlls = [pt["nll"] for pt in pts]
                cpcs = [pt["cpc"] for pt in pts]
                rec = {"param": glabel, "raw_attr": attr, "idxs": idxs,
                       "fitted_eff0": fitted_eff[0],
                       "nll_span": max(nlls)-min(nlls), "cpc_span": max(cpcs)-min(cpcs),
                       "points": pts}
                results["scans"].append(rec)
                print(f"[profile] {glabel:>20s}  fitted_eff={fitted_eff[0]:+.4f}  "
                      f"NLL span={rec['nll_span']:.4f}  CPC span={rec['cpc_span']:.4f}", flush=True)

        _outp = V3_ROOT / args.profile_out
        _outp.parent.mkdir(parents=True, exist_ok=True)
        with open(_outp, "w") as _f:
            _pjson.dump(results, _f, indent=2)
        # Rank by NLL span (smallest = least identified)
        ranked = sorted(results["scans"], key=lambda r: r["nll_span"])
        print("\n[profile] === IDENTIFICATION RANKING (NLL span, smallest=flattest=least identified) ===", flush=True)
        for r in ranked:
            flag = "FLAT (not identified)" if r["nll_span"] < 0.01 else ("weak" if r["nll_span"] < 0.05 else "identified")
            print(f"  {r['param']:>20s}  NLL span {r['nll_span']:.4f}  CPC span {r['cpc_span']:.4f}  -> {flag}", flush=True)
        print(f"\n[profile] wrote {_outp}", flush=True)
        return

    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []
    current_stage = 0
    stage_t0 = t0

    r7_stage_switched = False
    for ep in range(total_epochs):
        # R7 staged training transition: at ep == r7_warmup_epochs, switch from
        # baseline-only to R7-additions-only training.
        if args.r7_warmup_epochs > 0 and ep == args.r7_warmup_epochs and not r7_stage_switched:
            prev_cpc = history[-1]["cpc"] if history else 0.0
            print(f"\n[r7-staging] Baseline stage done at ep {ep}, CPC={prev_cpc:.4f}. Switching to R7-only.")
            _apply_r7_stage("r7")
            # Rebuild optimizer with only currently-trainable params
            _rum_other_s, _rum_attn_s = _split_rum_params(only_requires_grad=True)
            _opt_groups_s = [
                {"params": [p for p in encoder.parameters() if p.requires_grad],
                 "lr": args.lr_theta, "weight_decay": 1e-4},
                {"params": _rum_other_s, "lr": args.lr_rum,
                 "weight_decay": 1e-3},  # stronger wd on R7 additions to prevent runaway
            ]
            if _rum_attn_s:
                _opt_groups_s.append({"params": _rum_attn_s, "lr": args.lr_rum,
                                      "weight_decay": float(args.attn_weight_decay)})
            optimizer = torch.optim.AdamW(_opt_groups_s)
            r7_stage_switched = True
            no_improve = 0   # reset patience for fresh stage

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
                _rum_other_s, _rum_attn_s = _split_rum_params(only_requires_grad=True)
                _opt_groups_s = [
                    {"params": [p for p in encoder.parameters() if p.requires_grad],
                     "lr": args.lr_theta, "weight_decay": 1e-4},
                    {"params": _rum_other_s, "lr": args.lr_rum, "weight_decay": 0.0},
                ]
                if _rum_attn_s:
                    _opt_groups_s.append({"params": _rum_attn_s, "lr": args.lr_rum,
                                          "weight_decay": float(args.attn_weight_decay)})
                optimizer = torch.optim.AdamW(_opt_groups_s)
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
            hour_weights=hour_weights,
            log_transit_access_z=log_transit_access_z,
            log_commercial_z=log_commercial_z,
            quality_nvm_z=quality_nvm_z,
            quality_ta_z=quality_ta_z,
            quality_wf_z=quality_wf_z,
            log_time_mask_per_tier=log_time_mask_per_tier,
            strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
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
        # Ben-Akiva-Morikawa in-loss external occupation moment (z-scored pattern anchor).
        # Aggregate OD is flat in delta_match; this penalty narrows it to the WU07AUK pattern.
        if args.lambda_occ_penalty > 0 and getattr(rum, "use_soc_mixture", False):
            dm = rum.delta_match
            dm_z = (dm - dm.mean()) / (dm.std() + 1e-6)
            _ext = torch.tensor([float(x) for x in args.occ_ext_delta.split(",")],
                                device=dm.device, dtype=dm.dtype)
            ext_z = (_ext - _ext.mean()) / (_ext.std() + 1e-6)
            loss = loss + args.lambda_occ_penalty * ((dm_z - ext_z) ** 2).mean()
        # Income/class external moment (NS-SeC): per-tier wage attraction monotone increasing.
        if args.lambda_class_penalty > 0 and getattr(rum, "use_tier_mixture", False):
            aw = torch.nn.functional.softplus(rum.raw_alpha_wage)
            aw_z = (aw - aw.mean()) / (aw.std() + 1e-6)
            if args.class_ext_pattern == "mono":
                _ct = torch.linspace(0.0, 1.0, aw.numel(), device=aw.device, dtype=aw.dtype)
            else:
                _ct = torch.tensor([float(x) for x in args.class_ext_pattern.split(",")],
                                   device=aw.device, dtype=aw.dtype)
            ct_z = (_ct - _ct.mean()) / (_ct.std() + 1e-6)
            loss = loss + args.lambda_class_penalty * ((aw_z - ct_z) ** 2).mean()
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
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
            )
            val_nll = float(out_e["nll_dest"])
            val_cpc = cpc(out_e["log_P_D"], observed_OD, val_mask,
                          hour_weights=hour_weights)
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
            hour_weights=hour_weights,
            log_transit_access_z=log_transit_access_z,
            log_commercial_z=log_commercial_z,
            quality_nvm_z=quality_nvm_z,
            quality_ta_z=quality_ta_z,
            quality_wf_z=quality_wf_z,
            log_time_mask_per_tier=log_time_mask_per_tier,
            strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
        )
        final_cpc = cpc(out_final["log_P_D"], observed_OD, val_mask,
                        hour_weights=hour_weights)
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
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
            )
            abl_cpc = cpc(out_abl["log_P_D"], observed_OD, val_mask,
                          hour_weights=hour_weights)
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
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
            )
            cpc_no_filter = cpc(out_no_filter["log_P_D"], observed_OD, val_mask,
                                hour_weights=hour_weights)
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
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
            )
        cpc_no_cosine_match = cpc(out_no_match["log_P_D"], observed_OD, val_mask,
                                  hour_weights=hour_weights)
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
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
            )
            cpc_no_nn = cpc(out_no_nn["log_P_D"], observed_OD, val_mask,
                            hour_weights=hour_weights)
            # Joint: NN off AND cosine_match off
            out_no_both = forward_cs(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, torch.ones_like(match_prob),
                log_M_j, log_W_j, log_D_j,
                income_score, pct_kids, mean_cars, income_tier_props,
                pi_m_pair, grid_borough_idx, observed_OD, val_mask,
                soc_props_per_origin=soc_props_per_origin,
                per_soc_demand_share_j=per_soc_demand_share_j,
                hour_weights=hour_weights,
                log_transit_access_z=log_transit_access_z,
                log_commercial_z=log_commercial_z,
                quality_nvm_z=quality_nvm_z,
                quality_ta_z=quality_ta_z,
                quality_wf_z=quality_wf_z,
                log_time_mask_per_tier=log_time_mask_per_tier,
                strip_time_from_vdest_iv=args.strip_time_from_vdest_iv,
            )
            cpc_no_both = cpc(out_no_both["log_P_D"], observed_OD, val_mask,
                              hour_weights=hour_weights)
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
