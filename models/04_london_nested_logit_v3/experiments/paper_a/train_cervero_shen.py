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
):
    V_gnn_jt = encoder(X_static, X_dynamic, edge_index)              # (T, N)
    T, N = V_gnn_jt.shape
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
        # ---- Tier mixture path: compute V_upper_k per tier, then marginalize ----
        K = rum.n_income_tiers                                          # 3
        # alpha_w/gamma_M/nu_D/delta_m: each (K,)
        # build V_upper_k: (K, T, N, N)
        V_uppers = []
        for k in range(K):
            gamma_eff_k = gamma_M[k] + delta_m[k] * match_signal        # (N, N)
            V_M_k = gamma_eff_k * log_M_j.view(1, N)                    # (N, N)
            V_other_k = (alpha_w[k] * log_W_j + nu_D[k] * log_D_j).view(1, N)  # (1, N)
            V_rum_k = (V_M_k + V_other_k).unsqueeze(0) + lam_view * IV_mode    # (T, N, N)
            if V_push_pull is not None:
                V_rum_k = V_rum_k + V_push_pull                          # broadcast (1, N, N) → (T, N, N)
            if V_self_loop is not None:
                V_rum_k = V_rum_k + V_self_loop                          # (T, N, N) self-loop boost
            if V_busy_dest is not None:
                V_rum_k = V_rum_k + V_busy_dest                          # (T, 1, N) broadcast on i
            V_uppers.append(V_rum_k)
        V_uppers_stacked = torch.stack(V_uppers, dim=0)                  # (K, T, N, N)
        V_rum_dest = V_uppers_stacked                                    # downstream knows tier dim
    else:
        # single-RUM path (original)
        gamma_effective = gamma_M + delta_m * match_signal               # (N, N)
        V_M = gamma_effective * log_M_j.view(1, N)                       # (N, N)
        V_other = (alpha_w * log_W_j.view(1, 1, N)
                   + nu_D * log_D_j.view(1, 1, N))                       # (1, 1, N)
        V_rum_dest = (V_M.unsqueeze(0) + V_other + lam_view * IV_mode)   # (T, N, N)
        if V_push_pull is not None:
            V_rum_dest = V_rum_dest + V_push_pull
        if V_self_loop is not None:
            V_rum_dest = V_rum_dest + V_self_loop
        if V_busy_dest is not None:
            V_rum_dest = V_rum_dest + V_busy_dest

    V_gnn = V_gnn_jt.view(T, 1, N).expand(T, N, N)                     # (T, N, N)

    # Compute V_dest per tier (if tier mixture) or directly (single RUM)
    if rum.gnn_mode == "residual":
        w_nn = rum.gnn_residual_scale
        if w_nn is None:
            V_nn_scaled = torch.zeros(T, N, N, device=V_gnn.device)
        else:
            V_nn_scaled = w_nn * V_gnn                                  # (T, N, N)
        if rum.use_tier_mixture:
            # V_rum_dest is (K, T, N, N), add same NN to each
            V_dest = V_rum_dest + V_nn_scaled.unsqueeze(0)              # (K, T, N, N)
        else:
            V_dest = V_rum_dest + V_nn_scaled                            # (T, N, N)
    else:
        blend = rum.gnn_blend
        if blend is None:
            V_nn_scaled = torch.zeros(T, N, N, device=V_gnn.device)
        else:
            V_nn_scaled = blend * V_gnn
        if rum.use_tier_mixture:
            if blend is None:
                V_dest = V_rum_dest
            else:
                V_dest = (1.0 - blend) * V_rum_dest + V_nn_scaled.unsqueeze(0)
        else:
            if blend is None:
                V_dest = V_rum_dest
            else:
                V_dest = (1.0 - blend) * V_rum_dest + V_nn_scaled

    # Tier-mixture: log P(j|i) = logsumexp_k(log π_k + log softmax_j(V_upper_k))
    if rum.use_tier_mixture:
        # V_dest shape: (K, T, N, N)
        log_P_per_tier = F.log_softmax(V_dest, dim=-1)                  # (K, T, N, N)
        # income_tier_props shape (N, K) → log_pi (K, 1, N, 1) broadcast on T, j-axis
        log_pi = torch.log(income_tier_props.t().unsqueeze(1).unsqueeze(-1) + 1e-9)  # (K, 1, N, 1)
        log_P_D = torch.logsumexp(log_pi + log_P_per_tier, dim=0)       # (T, N, N)
    else:
        log_P_D = torch.log_softmax(V_dest, dim=-1)                     # (T, N, N)

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

    # NN-norm regularization (training only): pushes ‖V_NN‖² down to enforce theory-dominance
    nn_norm_sq = (V_nn_scaled ** 2).mean()

    # ---- diagnostic ----
    with torch.no_grad():
        # tier-mixture: alpha/gamma/nu/delta are (K,) tensors → store as lists
        def _diag_val(t):
            return t.tolist() if t.dim() > 0 else float(t)
        diagnostic = {
            "use_tier_mixture": rum.use_tier_mixture,
            "alpha_wage": _diag_val(rum.alpha_wage),
            "gamma_M": _diag_val(rum.gamma_M),
            "nu_D": _diag_val(rum.nu_D),
            "delta_match": _diag_val(rum.delta_match),
            "beta_t_mean": float(rum.beta_t_per_mode.mean()),
            "beta_t_slope_mean": float(rum.beta_t_slope_per_mode.mean()),
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
            "gnn_mode": rum.gnn_mode,
            "delta_measured": delta_measured,
            "rum_rms": float(rum_rms),
            "nn_rms": float(nn_rms),
            "use_push_pull": rum.use_push_pull,
            "xi_push_pull": float(rum.xi_push_pull) if rum.xi_push_pull is not None else None,
        }

    return {
        "log_P_D": log_P_D,
        "nll_dest": nll_dest,
        "ce_mode": ce_mode,
        "nn_norm_sq": nn_norm_sq,
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
    ap.add_argument("--gnn-mode", choices=["convex", "residual"], default="convex",
                    help="convex: V=(1-δ)V_RUM+δ V_NN (legacy v3b/c). "
                         "residual: V=V_RUM+w·V_NN (Wang TB-ResNet original).")
    ap.add_argument("--gnn-residual-scale-init", type=float, default=0.1)
    ap.add_argument("--lambda-nn-norm", type=float, default=0.0,
                    help="L2 penalty on ‖V_NN‖² to enforce Wang Path A (δ_measured < 0.30)")
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
                    help="Sequential mode: epochs for Phase 1 (RUM only).")
    ap.add_argument("--rum-checkpoint", type=str, default=None,
                    help="Sequential mode: load Phase 1 RUM checkpoint instead of retraining.")
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
    pi_m_pair = pair_mode_share.to(device).float()
    grid_borough_idx = grid_borough_idx.to(device)

    t_per_mode = {}
    for m in mode_names:
        key = "t_" + m
        tens = d[key].to(device).float()
        if tens.dim() == 2:
            tens = tens.unsqueeze(0).expand(T, N, N).contiguous()
        t_per_mode[m] = tens

    if args.use_gat:
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
    ).to(device)

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
    # Wang TB-ResNet sequential training (W4 §3.2 perspective 3)
    # Phase 1: train RUM only (force blend=0, encoder frozen)
    # Phase 2: freeze RUM, set blend=fixed_blend, train encoder only
    # ===================================================================
    if args.train_mode == "sequential":
        print(f"\n[sequential] Phase 1: train RUM only ({args.epochs_phase1} epochs)")
        # Force blend ≈ 0 (sigmoid(-15) ≈ 3e-7)
        if rum.raw_gnn_blend is not None:
            with torch.no_grad():
                rum.raw_gnn_blend.fill_(-15.0)
            rum.raw_gnn_blend.requires_grad = False
        # Freeze encoder
        for p in encoder.parameters():
            p.requires_grad = False

    t0 = time.time()
    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []

    phase1_epochs = args.epochs_phase1 if args.train_mode == "sequential" else 0
    total_epochs = (args.epochs_phase1 + args.epochs) if args.train_mode == "sequential" else args.epochs
    phase2_started = False

    for ep in range(total_epochs):
        # Phase transition for sequential training
        if args.train_mode == "sequential" and ep == phase1_epochs and not phase2_started:
            phase2_started = True
            phase1_time = time.time() - t0
            phase1_cpc = history[-1]["cpc"] if history else 0.0
            print(f"\n[sequential] Phase 1 done in {phase1_time:.0f}s, CPC={phase1_cpc:.4f}")
            # Set blend to fixed value
            if args.fixed_blend is not None and rum.raw_gnn_blend is not None:
                fb = max(min(float(args.fixed_blend) / max(args.blend_max, 1e-6), 1-1e-4), 1e-4)
                raw_b = math.log(fb / (1 - fb))
                with torch.no_grad():
                    rum.raw_gnn_blend.fill_(raw_b)
                rum.raw_gnn_blend.requires_grad = False
                print(f"[sequential] Phase 2: freeze RUM, train GNN at fixed blend={args.fixed_blend} ({args.epochs} epochs)")
            else:
                # No fixed_blend: keep blend learnable for Phase 2 only
                if rum.raw_gnn_blend is not None:
                    rum.raw_gnn_blend.requires_grad = True
                print(f"[sequential] Phase 2: freeze RUM (except blend), train GNN ({args.epochs} epochs)")
            # Freeze all RUM params except blend
            for name, p in rum.named_parameters():
                if "gnn_blend" not in name:
                    p.requires_grad = False
            # Unfreeze encoder
            for p in encoder.parameters():
                p.requires_grad = True
            # Reset optimizer to only train unfrozen params (encoder + maybe blend)
            optimizer = torch.optim.AdamW(
                [{"params": [p for p in encoder.parameters() if p.requires_grad],
                  "lr": args.lr_theta, "weight_decay": 1e-4},
                 {"params": [p for p in rum.parameters() if p.requires_grad],
                  "lr": args.lr_rum, "weight_decay": 0.0}]
            )
            best_val = float("inf")  # reset early stop for Phase 2
            no_improve = 0

        # ---- adjust epoch index for kl_warmup (relative to phase start) ----
        if args.train_mode == "sequential":
            eff_ep = ep if ep < phase1_epochs else (ep - phase1_epochs)
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
        )
        loss = (out["nll_dest"]
                + kl_weight * out["ce_mode"]
                + args.lambda_nn_norm * out["nn_norm_sq"])
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
            )
            val_nll = float(out_e["nll_dest"])
            val_cpc = cpc(out_e["log_P_D"], observed_OD, val_mask)
            diag = out_e["diagnostic"]
            ce_val = float(out_e["ce_mode"])

        history.append({
            "ep": ep, "tnll": float(out["nll_dest"].item()),
            "ce_train": float(out["ce_mode"].item()),
            "vnll": val_nll, "ce_val": ce_val, "cpc": val_cpc,
            "kl_weight": kl_weight, **{k: v for k, v in diag.items()
                                       if k not in ("asc_per_mode", "theta_inc_per_mode")},
        })

        if args.verbose or ep % 5 == 0 or ep == args.epochs - 1:
            if diag['gnn_mode'] == 'residual':
                gnn_str = (f"w_NN={diag['gnn_residual_scale']:.3f} "
                           f"δ_meas={diag['delta_measured']:.3f} "
                           f"(rum/nn rms {diag['rum_rms']:.2f}/{diag['nn_rms']:.2f})")
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
        # Don't early-stop during sequential Phase 1 — must reach phase transition
        in_phase1 = (args.train_mode == "sequential" and ep < phase1_epochs)
        if not in_phase1 and no_improve >= args.patience:
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
        )
        final_cpc = cpc(out_final["log_P_D"], observed_OD, val_mask)
        final_diag = out_final["diagnostic"]
        full_snapshot = rum.snapshot()

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
    elif final_diag["blend"] is not None:
        print(f"  blend (δ_GNN)  = {final_diag['blend']:.4f}  (v3a 0.437, legacy convex)")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
