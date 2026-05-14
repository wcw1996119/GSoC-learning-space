"""v3 Destination→Mode nested logit training (Phase 3b'.E entry point).

End-to-end training script for the v3 redesign:
  - V_dest with non-linear δ(d) and GNN blend
  - V_mode with non-linear β_m(d) and mode-level features θ_m·X_{i,m}
  - Borough-level λ_b coupling upper and lower nest (33 boroughs by default)
  - KL constraint loss using Census pair-level π_m(i, j) as mode-share supervisor
  - Linear warmup on λ_kl over first ``--kl-warmup-epochs``

Reads v2 data + the new mode-level features built by
``build_mode_level_features.py``.

Compared to v2 baseline (CPC 0.485 ± 0.001, δ_free 0.446) the diagnostic
verdict at the end of training will tell us whether D→M nested + non-linear V
moves δ down (Path A < 0.30, Path B 0.30-0.40, Path C > 0.40 — see
methodology/02_dm_nested_design.md §7).

Run:
    python experiments/paper_a/train_dm_nested.py \
        --epochs 200 --patience 30 --seed 0 \
        --lambda-kl 1.0 --kl-warmup-epochs 15 \
        --blend-max 1.0 \
        --device cuda --verbose
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"

# v2 owns the `models_lib` namespace (encoder, dataloader, helpers).
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

# Load v3 head by absolute path (avoid models_lib package collision with v2).
_spec = importlib.util.spec_from_file_location(
    "v3_dm_nested_logit_head",
    V3_ROOT / "models_lib" / "inverse_rum" / "dm_nested_logit_head.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DMNestedLogitHead = _mod.DMNestedLogitHead

from models_lib.inverse_rum.dual_branch_encoder import DualBranchEncoder  # v2
from models_lib.inverse_rum.dual_branch_mixture_trainer import (  # v2
    make_distance_aware_mode_share,
)
from experiments.paper_a.compare_four_variants import load_data  # v2 loader


# =============================================================================
# Forward computation
# =============================================================================

def forward_dm_nested(
    encoder: DualBranchEncoder,
    rum: DMNestedLogitHead,
    X_static: torch.Tensor,
    X_dynamic: torch.Tensor,
    edge_index: torch.Tensor,
    t_per_mode: Dict[str, torch.Tensor],
    mode_names: list,
    log_d_ij: torch.Tensor,
    occ_match: torch.Tensor,
    pi_m_pair: torch.Tensor,
    mode_level_X: torch.Tensor,
    grid_borough_idx: torch.Tensor,
    observed_OD: torch.Tensor,
    train_mask: torch.Tensor,
):
    """Run one full forward through encoder + DM nested logit head.

    Returns
    -------
    dict with:
        log_P_D       : (T, N, N) — log P(j | i, t)
        nll_dest      : scalar    — destination NLL (only on origins in train_mask)
        ce_mode_per_ijt : (T, N, N) — per-cell cross-entropy of P_M against π_m(i, j)
        ce_mode       : scalar    — flow-weighted CE_mode summary
        diagnostic    : dict      — λ_b mean/std, β_m(d_ref) values, blend, etc.
    """
    V_jt = encoder(X_static, X_dynamic, edge_index)                  # (T, N)
    T, N = V_jt.shape
    M = rum.n_modes

    # ---- precompute per-destination λ_{b(j)} ----
    lambda_per_j = rum.lambda_for_destination(grid_borough_idx)      # (N,)
    lam_view = lambda_per_j.view(1, 1, N)                            # broadcast on j-axis

    # ---- precompute β_m(d_ij) and θ_m^T·X_{i,m} ----
    beta_per_m_d = rum.beta_at_log_d(log_d_ij)                       # (N, N, M)
    theta_X = rum.theta_dot_X(mode_level_X)                          # (N, M) per origin per mode
    alpha_m = rum.alpha_per_mode                                     # (M,)

    # ---- iterate over modes: build IV_mode + cross-entropy accumulator ----
    # IV_mode(i,j,t) = log Σ_m exp( V_mode_m / λ_{b(j)} )
    # ce_accum(i,j,t) = Σ_m π_m(i,j) · V_mode_m / λ_{b(j)}
    # cross-entropy = IV_mode - ce_accum (= -Σ_m π_m log P_M)
    log_iv = None
    ce_accum = None

    log_d_T = log_d_ij.unsqueeze(0)                                  # (1, N, N)

    for m_idx, name in enumerate(mode_names):
        t_m = t_per_mode[name]                                       # (T, N, N)
        beta_m_d = beta_per_m_d[..., m_idx]                          # (N, N)
        # V_mode_m(i, j, t) = β_m(d) · t_{ij,m} + θ_m·X_{i,m} + α_m
        # θ_m·X_{i,m} depends only on i (origin) → broadcast as (1, N, 1)
        # α_m is scalar (broadcast)
        V_mode_m = (beta_m_d.unsqueeze(0) * t_m
                    + theta_X[:, m_idx].view(1, N, 1)
                    + alpha_m[m_idx])                                # (T, N, N)

        scaled = V_mode_m / lam_view                                 # (T, N, N)
        if log_iv is None:
            log_iv = scaled
        else:
            log_iv = torch.logaddexp(log_iv, scaled)

        pi_m_ij = pi_m_pair[:, :, m_idx].unsqueeze(0)                # (1, N, N)
        if ce_accum is None:
            ce_accum = pi_m_ij * scaled
        else:
            ce_accum = ce_accum + pi_m_ij * scaled
        # V_mode_m goes out of scope, freed by autograd's release strategy

    IV_mode = log_iv                                                  # (T, N, N)
    ce_mode_per_ijt = IV_mode - ce_accum                              # (T, N, N) cross-entropy

    # ---- V_dest: upper nest destination utility ----
    gamma = rum.gamma                                                 # scalar
    delta_d = rum.delta_at_log_d(log_d_ij)                            # (N, N)
    V_gnn = V_jt.view(T, 1, N).expand(T, N, N)                        # (T, N, N) - destination only

    # V_RUM-part of destination utility (RUM = utility-theory components)
    V_rum_dest = (gamma * log_d_T                                     # γ · log_d_ij
                  + delta_d.unsqueeze(0) * occ_match.view(1, N, N)    # δ(d) · OccMatch
                  + lam_view * IV_mode)                               # λ_{b(j)} · IV_mode

    # Wang TB-ResNet blend (V_total = (1-blend)·V_RUM + blend·V_GNN)
    blend = rum.gnn_blend
    if blend is None:
        V_dest = V_rum_dest + rum.alpha * V_gnn                       # legacy additive
    else:
        V_dest = (1.0 - blend) * V_rum_dest + blend * V_gnn           # TB-ResNet blend

    log_P_D = torch.log_softmax(V_dest, dim=-1)                       # (T, N, N)

    # ---- losses ----
    mask_f = train_mask.view(1, N, 1).float()                         # train origin mask
    flow = observed_OD * mask_f                                       # (T, N, N)
    flow_sum = flow.sum().clamp(min=1.0)

    nll_dest = -(flow * log_P_D).sum() / flow_sum

    # CE_mode weighted by flow (so we focus on real commuting cells)
    ce_mode = (flow * ce_mode_per_ijt).sum() / flow_sum

    # ---- diagnostic snapshot ----
    with torch.no_grad():
        diagnostic = {
            "lambda_b_mean": float(rum.lambda_per_borough.mean()),
            "lambda_b_std": float(rum.lambda_per_borough.std()),
            "lambda_b_min": float(rum.lambda_per_borough.min()),
            "lambda_b_max": float(rum.lambda_per_borough.max()),
            "blend": float(blend) if blend is not None else None,
            "beta_intercept_mean": float(rum.beta_intercept_per_mode.mean()),
            "beta_slope_mean": float(rum.beta_slope.mean()),
            "gamma": float(rum.gamma),
            "delta_intercept": float(rum.delta_intercept),
            "delta_slope": float(rum.delta_slope),
            "alpha_per_mode": rum.alpha_per_mode.detach().cpu().tolist(),
        }

    return {
        "log_P_D": log_P_D,
        "nll_dest": nll_dest,
        "ce_mode": ce_mode,
        "diagnostic": diagnostic,
    }


def cpc(log_P: torch.Tensor, observed_OD: torch.Tensor, mask: torch.Tensor) -> float:
    """Common Part of Commuters (CPC) ∈ [0, 1]."""
    with torch.no_grad():
        P = log_P.exp()
        row_sum = observed_OD.sum(dim=2, keepdim=True)
        pred = P * row_sum
        num = 2.0 * torch.minimum(pred[:, mask], observed_OD[:, mask]).sum()
        den = (pred[:, mask].sum() + observed_OD[:, mask].sum()).clamp(min=1.0)
        return float(num / den)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blend-max", type=float, default=1.0,
                    help="Hard cap on Wang TB-ResNet GNN blend. 1.0=free, 0.3=theory-dominant, 0=pure RUM.")
    ap.add_argument("--lambda-kl", type=float, default=1.0,
                    help="Weight on KL mode-share constraint term.")
    ap.add_argument("--kl-warmup-epochs", type=int, default=15,
                    help="Linear warmup epochs for λ_kl.")
    ap.add_argument("--lr-theta", type=float, default=1e-3)
    ap.add_argument("--lr-rum", type=float, default=1e-2)
    ap.add_argument("--lambda-init", type=float, default=0.95)
    ap.add_argument("--lambda-eps-min", type=float, default=0.05)
    ap.add_argument("--mode-features-path", default="data/processed/mode_level_features.npz")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="evaluation_outputs/paper_a/dm_nested_smoke.json")
    args = ap.parse_args()

    # ---- load v2 data ----
    print(f"Loading v2 data from {V2_ROOT / 'data' / 'processed'} ...")
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

    # ---- load mode-level features ----
    mfeat_path = V3_ROOT / args.mode_features_path
    if not mfeat_path.exists():
        raise FileNotFoundError(
            f"Mode-level features not found at {mfeat_path}. "
            f"Run experiments/paper_a/build_mode_level_features.py first."
        )
    mfeat = np.load(mfeat_path, allow_pickle=True)
    mode_level_X = torch.from_numpy(mfeat["X"]).float()                # (N, M, F_mode)
    print(f"Mode-level features: shape={tuple(mode_level_X.shape)}  features={list(mfeat['feature_names'])}")

    # ---- borough lookup (from v2 demo_cache) ----
    grid_borough_idx = d["grid_borough_idx"].long()                    # (N,)
    n_boroughs = int(grid_borough_idx.max().item()) + 1
    print(f"Borough count: {n_boroughs}  (each will get its own λ_b)")

    # ---- send to device ----
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N, T = d["N"], d["T"]
    M = pair_mode_share.shape[2]
    F_mode = mode_level_X.shape[2]
    mode_names = ["car", "transit", "walk"]

    X_static = d["X_static"].to(device).float()
    X_dynamic = d["X_dynamic"].to(device).float()
    edge_index = d["edge_index"].to(device)
    observed_OD = d["F_ij_t"].to(device).float()
    log_d_ij = d["log_d"].to(device).float()
    train_mask = d["train_mask"].to(device).bool()
    val_mask = d["val_mask"].to(device).bool()
    occ_match = d["occ_match"].to(device).float()
    pi_m_pair = pair_mode_share.to(device).float()                     # (N, N, M)
    mode_level_X = mode_level_X.to(device)
    grid_borough_idx = grid_borough_idx.to(device)

    t_per_mode = {}
    for m in mode_names:
        key = "t_" + m
        tens = d[key].to(device).float()
        if tens.dim() == 2:
            tens = tens.unsqueeze(0).expand(T, N, N).contiguous()
        t_per_mode[m] = tens

    # ---- build encoder + head ----
    encoder = DualBranchEncoder(
        static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
        hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
    ).to(device)

    rum = DMNestedLogitHead(
        n_modes=M, n_tiers=1,                                          # tier mixture dropped for v3 smoke
        n_boroughs=n_boroughs,
        n_mode_features=F_mode,
        beta_intercept_init=-0.07, beta_slope_init=0.0,
        gamma_init=-0.5,
        delta_intercept_init=0.0, delta_slope_init=0.0,
        lambda_init=args.lambda_init, lambda_eps_min=args.lambda_eps_min,
        use_gnn_blend=True, gnn_blend_init=0.5, blend_max=args.blend_max,
    ).to(device)

    print(f"\n[train] starting: seed={args.seed} epochs={args.epochs} "
          f"blend_max={args.blend_max} λ_kl={args.lambda_kl} kl_warmup={args.kl_warmup_epochs}")
    print(f"        params: encoder={sum(p.numel() for p in encoder.parameters()):,}  "
          f"rum={sum(p.numel() for p in rum.parameters()):,}")
    print(f"        rum component sizes:  β_intercept(M={M}) β_slope(M={M}) α_m(M={M}) "
          f"θ(M×F_mode={M}×{F_mode}) γ δ_0 δ_1 λ_b(B={n_boroughs}) blend")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.lr_theta, "weight_decay": 1e-4},
            {"params": rum.parameters(), "lr": args.lr_rum, "weight_decay": 0.0},
        ]
    )

    # ---- train loop ----
    t0 = time.time()
    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []

    for ep in range(args.epochs):
        # ---- λ_kl warmup ----
        if args.kl_warmup_epochs > 0:
            kl_weight = args.lambda_kl * min(1.0, (ep + 1) / args.kl_warmup_epochs)
        else:
            kl_weight = args.lambda_kl

        encoder.train(); rum.train()
        optimizer.zero_grad()
        out = forward_dm_nested(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, occ_match,
            pi_m_pair, mode_level_X, grid_borough_idx,
            observed_OD, train_mask,
        )
        loss = out["nll_dest"] + kl_weight * out["ce_mode"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(rum.parameters()), 1.0
        )
        optimizer.step()

        encoder.eval(); rum.eval()
        with torch.no_grad():
            out_e = forward_dm_nested(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, occ_match,
                pi_m_pair, mode_level_X, grid_borough_idx,
                observed_OD, val_mask,
            )
            val_nll = float(out_e["nll_dest"])
            val_cpc = cpc(out_e["log_P_D"], observed_OD, val_mask)
            diag = out_e["diagnostic"]
            ce_val = float(out_e["ce_mode"])

        history.append({
            "ep": ep,
            "tnll": float(out["nll_dest"].item()),
            "ce_train": float(out["ce_mode"].item()),
            "vnll": val_nll,
            "ce_val": ce_val,
            "cpc": val_cpc,
            "kl_weight": kl_weight,
            **diag,
        })

        if args.verbose or ep % 5 == 0 or ep == args.epochs - 1:
            lam_str = (f"λ_b=[μ={diag['lambda_b_mean']:.3f} σ={diag['lambda_b_std']:.3f} "
                       f"min={diag['lambda_b_min']:.3f} max={diag['lambda_b_max']:.3f}]")
            beta_str = (f"β=[i={diag['beta_intercept_mean']:+.3f} "
                        f"s={diag['beta_slope_mean']:+.4f}]")
            blend_str = f"δ={diag['blend']:.3f}" if diag['blend'] is not None else "δ=n/a"
            print(f"ep {ep:3d} | tnll {out['nll_dest'].item():.3f} | "
                  f"ce_t {out['ce_mode'].item():.3f} | "
                  f"vnll {val_nll:.3f} | cpc {val_cpc:.3f} | klw {kl_weight:.2f} | "
                  f"{blend_str} | {lam_str} | {beta_str}")

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            no_improve = 0
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "rum": {k: v.detach().clone() for k, v in rum.state_dict().items()},
            }
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"early stop ep {ep}")
            break

    # ---- restore best state ----
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        rum.load_state_dict(best_state["rum"])

    # ---- final eval at best state ----
    encoder.eval(); rum.eval()
    with torch.no_grad():
        out_final = forward_dm_nested(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, occ_match,
            pi_m_pair, mode_level_X, grid_borough_idx,
            observed_OD, val_mask,
        )
        final_cpc = cpc(out_final["log_P_D"], observed_OD, val_mask)
        final_diag = out_final["diagnostic"]
        full_snapshot = rum.snapshot()

    elapsed = time.time() - t0
    result = {
        "config": "dm_nested_smoke",
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
        "v2_baseline_blend_free": 0.446,
        "history": history,
    }
    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[smoke] done in {elapsed:.0f}s ({len(history)} epochs)")
    print(f"  CPC            = {final_cpc:.4f}  (v2 baseline 0.485)")
    print(f"  gnn_blend (δ)  = {final_diag['blend']:.4f}  (v2 free 0.446)" if final_diag["blend"] is not None else "  blend disabled")
    print(f"  λ_b mean       = {final_diag['lambda_b_mean']:.3f}  std={final_diag['lambda_b_std']:.3f}")
    print(f"  λ_b min/max    = {final_diag['lambda_b_min']:.3f} / {final_diag['lambda_b_max']:.3f}")
    print(f"  β_intercept    = {final_diag['beta_intercept_mean']:+.3f}")
    print(f"  β_slope        = {final_diag['beta_slope_mean']:+.4f} (slope on log_d)")
    print(f"  γ              = {final_diag['gamma']:+.3f}")
    print(f"  δ_0, δ_1       = {final_diag['delta_intercept']:+.3f}, {final_diag['delta_slope']:+.4f}")
    print(f"  saved to {out_path}")

    # ---- verdict ----
    print("\n[verdict]")
    blend_final = final_diag["blend"]
    if blend_final is None:
        print("  blend disabled — no GNN comparison available")
    elif blend_final < 0.30:
        print(f"  δ_free = {blend_final:.3f} < 0.30 → **Path A (theory-dominant)**: nested+non-linear V displaced GNN")
    elif blend_final < 0.40:
        print(f"  δ_free = {blend_final:.3f} ∈ [0.30, 0.40) → **Path B (spatial extension)**: partial improvement")
    else:
        print(f"  δ_free = {blend_final:.3f} ≥ 0.40 → **Path C (data finding)**: spatial structure intrinsic")

    lam_std = final_diag["lambda_b_std"]
    if lam_std > 0.1:
        print(f"  λ_b std = {lam_std:.3f} > 0.10 → borough heterogeneity present → paper figure")
    else:
        print(f"  λ_b std = {lam_std:.3f} ≤ 0.10 → borough heterogeneity weak → single-λ may suffice")


if __name__ == "__main__":
    main()
