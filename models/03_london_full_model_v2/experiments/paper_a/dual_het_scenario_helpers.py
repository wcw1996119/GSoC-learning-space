"""Shared helpers for DUAL_HET (DS-SIGNN) scenario forward.

Used by:
  - scenario_A_dual_het.py  (spatial intervention + Hansen accessibility)
  - scenario_B_dual_het.py  (temporal intervention + per-hour BPR + h08 access)

Centralises:
  - ckpt loading (rebuild encoder + head from state dicts)
  - data loading (delegates to compare_four_variants.load_data)
  - distance-aware pair_mode_share rebuild (must match training recipe)
  - forward log P(j | i, t) under arbitrary X_static / X_dynamic / t_per_mode
  - BPR loop where only the car-mode t_ij is updated (transit/walk untouched)
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum.dual_branch_encoder import DualBranchEncoder
from models_lib.inverse_rum.mixture_head import MixtureRUMHead
from models_lib.inverse_rum.dual_branch_mixture_trainer import (
    make_distance_aware_mode_share,
)
from models_lib.inverse_rum.bpr_layer import bpr_multiplier

from experiments.paper_a.compare_four_variants import load_data


def load_dual_het_ckpt(ckpt_path: Path) -> Tuple[DualBranchEncoder, MixtureRUMHead, dict]:
    """Rebuild encoder + head from a ckpt produced by train_dual_het_local.py."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = DualBranchEncoder(
        static_dim=cfg["F_static"], dyn_dim=cfg["F_dynamic"],
        hidden_dim=cfg["hidden_dim"], gru_hidden=cfg["gru_hidden"],
        n_sage_layers=cfg["n_sage_layers"], tcn_kernels=tuple(cfg["tcn_kernels"]),
    )
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()

    head = MixtureRUMHead(
        n_hours=cfg["n_hours"], n_modes=cfg["n_modes"], n_tiers=cfg["n_tiers"],
        tier_specific_delta=cfg["tier_specific_delta"],
    )
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return enc, head, ckpt


def build_scenario_inputs(walk_threshold_km: float = 5.0) -> dict:
    """Load all inputs needed for forward + BPR + Hansen.

    Returns a dict with the same fields as compare_four_variants.load_data plus:
      - pair_mode_share : (N, N, M) distance-aware
      - log_pi_mk_pair  : (N, N, M, K) precomputed log mixture prior
      - dist_km         : (N, N) Euclidean km
      - cache_static_mu / sd : raw → z round-trip stats
      - grid_ids, pop_w (for inequality weights)
    """
    import pandas as pd
    d = load_data()
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)

    coords = torch.from_numpy(cache["coords_bng"].astype(np.float32))
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_km = (torch.linalg.norm(diff, dim=-1) / 1000.0).clamp(min=0.1)

    mode_names = ["car", "transit", "walk"]
    pair_mode_share = make_distance_aware_mode_share(
        d["mode_share"], dist_km, mode_names,
        walk_threshold_km=walk_threshold_km,
    )                                                                    # (N, N, M)
    income_tier_props = d["income_tier_props"]                           # (N, K)

    N, _, M = pair_mode_share.shape
    K = income_tier_props.shape[1]
    log_pi_m = torch.log(pair_mode_share.clamp(min=1e-9))                # (N, N, M)
    log_pi_k = torch.log(income_tier_props.clamp(min=1e-9))              # (N, K)
    log_pi_mk_pair = log_pi_m.unsqueeze(-1) + log_pi_k.view(N, 1, 1, K)  # (N, N, M, K)

    grid_ids = cache["grid_ids"].tolist()
    df_feats = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    df_feats = df_feats.set_index("grid_id").reindex(grid_ids)
    pop_w = df_feats["population"].fillna(0.0).to_numpy().astype(np.float64)

    d.update({
        "pair_mode_share": pair_mode_share, "income_tier_props": income_tier_props,
        "log_pi_mk_pair": log_pi_mk_pair,
        "dist_km": dist_km, "mode_names": mode_names,
        "cache_static_mu": np.asarray(cache["static_mean"]),
        "cache_static_sd": np.asarray(cache["static_std"]),
        "grid_ids": grid_ids, "pop_w": pop_w,
    })
    return d


def forward_log_p(
    encoder: DualBranchEncoder,
    head: MixtureRUMHead,
    X_static: torch.Tensor,
    X_dynamic: torch.Tensor,
    edge_index: torch.Tensor,
    t_per_mode: Dict[str, torch.Tensor],
    log_d_ij: torch.Tensor,
    log_pi_mk_pair: torch.Tensor,
    occ_match: Optional[torch.Tensor] = None,
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    mode_names: Tuple[str, ...] = ("car", "transit", "walk"),
) -> torch.Tensor:
    """Compute log P(j | i, t) of shape (T, N, N) under given inputs.

    Mirrors DualBranchMixtureTrainer._per_hour_log_p but works with arbitrary
    inputs (so we can swap X_static for spatial intervention, X_dynamic for
    temporal, or t_per_mode for BPR).
    """
    encoder.eval(); head.eval()
    with torch.no_grad():
        V_jt = encoder(X_static, X_dynamic, edge_index, norm_stats=norm_stats)   # (T, N)
        T, N = V_jt.shape
        K = head.n_tiers
        alpha = head.alpha
        gamma = head.gamma
        beta_per_mode = head.beta_t_per_mode                                       # (T, M)
        kappa = head.kappa                                                          # (K,)
        delta_k = head.delta_per_tier() if occ_match is not None else None

        V_b = V_jt.view(T, 1, N).expand(T, N, N)                                   # (T, N, N)
        d_b = log_d_ij.view(1, N, N)                                                # (1, N, N)

        log_p = None
        for m, name in enumerate(mode_names):
            t_m = t_per_mode[name]
            if t_m.dim() == 2:
                t_m = t_m.unsqueeze(0).expand(T, N, N)
            for k in range(K):
                beta_eff = (beta_per_mode[:, m] * kappa[k]).view(T, 1, 1)
                logits = alpha * V_b + beta_eff * t_m + gamma * d_b
                if delta_k is not None and occ_match is not None:
                    logits = logits + delta_k[k] * occ_match.view(1, N, N)
                log_p_mk = torch.log_softmax(logits, dim=-1)                       # (T, N, N)
                log_w = log_pi_mk_pair[:, :, m, k].view(1, N, N)
                weighted = log_p_mk + log_w
                log_p = weighted if log_p is None else torch.logaddexp(log_p, weighted)
        return log_p                                                                 # (T, N, N)


def predict_flow(
    encoder, head, X_static, X_dynamic, edge_index, t_per_mode,
    log_d_ij, log_pi_mk_pair, observed_OD, occ_match=None,
    norm_stats=None, mode_names=("car", "transit", "walk"),
) -> torch.Tensor:
    """Predict flow F(j|i,t) × row sum, returning (T, N, N) flow tensor."""
    log_p = forward_log_p(
        encoder, head, X_static, X_dynamic, edge_index, t_per_mode,
        log_d_ij, log_pi_mk_pair, occ_match=occ_match, norm_stats=norm_stats,
        mode_names=mode_names,
    )
    P = log_p.exp()
    row_sum = observed_OD.sum(dim=2, keepdim=True)
    return P * row_sum                                                              # (T, N, N)


def apply_bpr_to_car_only(
    t_per_mode: Dict[str, torch.Tensor],
    flow_tn: torch.Tensor,
    capacity_j: torch.Tensor,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    mult_cap: float = 2.5,
) -> Dict[str, torch.Tensor]:
    """Apply BPR to car-mode t_ij only. Transit and walk are unaffected.

    Parameters
    ----------
    t_per_mode : dict {car / transit / walk: (N,N) or (T,N,N)} — free-flow times
    flow_tn    : (N, N) flow used to compute inflow per j (sum over origins)
    capacity_j : (N,) destination capacity proxy.

    Returns
    -------
    new_t_per_mode : same dict structure, but t_per_mode['car'] is congested.
    """
    inflow_j = flow_tn.sum(dim=-2)                                                  # (N,) or (T, N)
    mult_j = bpr_multiplier(inflow_j, capacity_j, alpha=bpr_alpha,
                             beta=bpr_beta, mult_cap=mult_cap)                       # same shape as inflow_j
    out = dict(t_per_mode)
    t_car = t_per_mode["car"]
    if t_car.dim() == 2 and mult_j.dim() == 1:
        out["car"] = t_car * mult_j.view(1, -1)                                     # broadcast over origins
    elif t_car.dim() == 2 and mult_j.dim() == 2:
        # mult is per-hour but t_car is static — broadcast: (T, 1, N) × (N, N)
        # Result becomes (T, N, N).
        out["car"] = t_car.unsqueeze(0) * mult_j.unsqueeze(-2)
    elif t_car.dim() == 3 and mult_j.dim() == 1:
        out["car"] = t_car * mult_j.view(1, 1, -1)
    elif t_car.dim() == 3 and mult_j.dim() == 2:
        out["car"] = t_car * mult_j.unsqueeze(-2)
    return out
