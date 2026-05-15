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
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

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
    log_d_T = log_d_ij.unsqueeze(0)                                    # (1, N, N) broadcast

    log_iv = None
    ce_accum = None

    for m_idx, name in enumerate(mode_names):
        t_m = t_per_mode[name]                                        # (T, N, N)
        # β_t_m(d) = β_t_m,0 + β_t_m,1 · log_d_ij  → per-pair time-disutility coeff
        beta_t_m_at_d = beta_t[m_idx] + beta_t_slope[m_idx] * log_d_T  # (1, N, N)
        # V_lower_m(i, j, t) = β_t_m(d) · t + ASC_m + θ_inc_m · income_i
        V_lower_m = (beta_t_m_at_d * t_m
                     + asc[m_idx]
                     + theta_inc[m_idx] * income_score_per_origin.view(1, N, 1))  # (T, N, N)
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

    # ---- V_upper ----
    alpha_w = rum.alpha_wage
    gamma_M = rum.gamma_M
    nu_D = rum.nu_D
    delta_m = rum.delta_match

    # destination-level terms: (N,) → (1, 1, N)
    V_dest_destonly = (alpha_w * log_W_j.view(1, 1, N)
                       + gamma_M * log_M_j.view(1, 1, N)
                       + nu_D * log_D_j.view(1, 1, N))                 # (1, 1, N)

    # match_prob is (N, N) → (1, N, N)
    V_match = delta_m * match_prob.view(1, N, N)                       # (1, N, N)

    V_rum_dest = (V_dest_destonly
                  + V_match
                  + lam_view * IV_mode)                                # (T, N, N)

    V_gnn = V_gnn_jt.view(T, 1, N).expand(T, N, N)                     # (T, N, N)

    blend = rum.gnn_blend
    if blend is None:
        V_dest = V_rum_dest                                            # pure RUM
    else:
        V_dest = (1.0 - blend) * V_rum_dest + blend * V_gnn

    log_P_D = torch.log_softmax(V_dest, dim=-1)                        # (T, N, N)

    # ---- losses ----
    mask_f = train_mask.view(1, N, 1).float()
    flow = observed_OD * mask_f
    flow_sum = flow.sum().clamp(min=1.0)

    nll_dest = -(flow * log_P_D).sum() / flow_sum
    ce_mode = (flow * ce_mode_per_ijt).sum() / flow_sum

    # ---- diagnostic ----
    with torch.no_grad():
        diagnostic = {
            "alpha_wage": float(rum.alpha_wage),
            "gamma_M": float(rum.gamma_M),
            "nu_D": float(rum.nu_D),
            "delta_match": float(rum.delta_match),
            "beta_t_mean": float(rum.beta_t_per_mode.mean()),
            "beta_t_slope_mean": float(rum.beta_t_slope_per_mode.mean()),
            "asc_per_mode": rum.asc_per_mode.detach().cpu().tolist(),
            "theta_inc_per_mode": rum.theta_inc_per_mode.detach().cpu().tolist(),
            "lambda_b_mean": float(rum.lambda_per_borough.mean()),
            "lambda_b_std": float(rum.lambda_per_borough.std()),
            "lambda_b_min": float(rum.lambda_per_borough.min()),
            "lambda_b_max": float(rum.lambda_per_borough.max()),
            "blend": float(blend) if blend is not None else None,
        }

    return {
        "log_P_D": log_P_D,
        "nll_dest": nll_dest,
        "ce_mode": ce_mode,
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
        match_prob = torch.from_numpy(aux["match_z"]).float()
        log_M_j = torch.from_numpy(aux["log_M_z"]).float()
        log_W_j = torch.from_numpy(aux["log_W_z"]).float()
        log_D_j = torch.from_numpy(aux["log_D_z"]).float()
        income_score = torch.from_numpy(aux["income_z"]).float()
        print(f"v3 aux loaded (Z-SCORED features): all mean≈0 std≈1")
        print(f"  raw scaling factors stored as log_W_mean/std etc. for paper reporting")
    else:
        match_prob = torch.from_numpy(aux["match_prob"]).float()
        log_M_j = torch.from_numpy(aux["log_M_j"]).float()
        log_W_j = torch.from_numpy(aux["log_W_j"]).float()
        log_D_j = torch.from_numpy(aux["log_D_j"]).float()
        income_score = torch.from_numpy(aux["income_score_per_origin"]).float()
        print(f"v3 aux loaded (RAW features, no z-score): coefficients NOT directly comparable")

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
    match_prob = match_prob.to(device)
    log_M_j = log_M_j.to(device)
    log_W_j = log_W_j.to(device)
    log_D_j = log_D_j.to(device)
    income_score = income_score.to(device)
    pi_m_pair = pair_mode_share.to(device).float()
    grid_borough_idx = grid_borough_idx.to(device)

    t_per_mode = {}
    for m in mode_names:
        key = "t_" + m
        tens = d[key].to(device).float()
        if tens.dim() == 2:
            tens = tens.unsqueeze(0).expand(T, N, N).contiguous()
        t_per_mode[m] = tens

    encoder = DualBranchEncoder(
        static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
        hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
    ).to(device)
    rum = CerveroShenHead(
        n_modes=M, n_boroughs=n_boroughs,
        lambda_init=args.lambda_init, lambda_eps_min=args.lambda_eps_min,
        use_gnn_blend=True, gnn_blend_init=0.5, blend_max=args.blend_max,
    ).to(device)

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

    t0 = time.time()
    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []

    for ep in range(args.epochs):
        if args.kl_warmup_epochs > 0:
            kl_weight = args.lambda_kl * min(1.0, (ep + 1) / args.kl_warmup_epochs)
        else:
            kl_weight = args.lambda_kl

        encoder.train(); rum.train()
        optimizer.zero_grad()
        out = forward_cs(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
            income_score, pi_m_pair, grid_borough_idx, observed_OD, train_mask,
        )
        loss = out["nll_dest"] + kl_weight * out["ce_mode"]
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
                income_score, pi_m_pair, grid_borough_idx, observed_OD, val_mask,
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
            blend_str = f"δ={diag['blend']:.3f}" if diag['blend'] is not None else "δ=n/a"
            print(f"ep {ep:3d} | tnll {out['nll_dest'].item():.3f} | "
                  f"vnll {val_nll:.3f} | cpc {val_cpc:.3f} | klw {kl_weight:.2f} | "
                  f"α={diag['alpha_wage']:+.3f} γ={diag['gamma_M']:+.3f} ν={diag['nu_D']:+.3f} "
                  f"δ_m={diag['delta_match']:+.3f} | β_t,0={diag['beta_t_mean']:+.3f} "
                  f"β_t,1={diag['beta_t_slope_mean']:+.4f} | "
                  f"λ_μ={diag['lambda_b_mean']:.3f} | {blend_str}")

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

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        rum.load_state_dict(best_state["rum"])

    encoder.eval(); rum.eval()
    with torch.no_grad():
        out_final = forward_cs(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, match_prob, log_M_j, log_W_j, log_D_j,
            income_score, pi_m_pair, grid_borough_idx, observed_OD, val_mask,
        )
        final_cpc = cpc(out_final["log_P_D"], observed_OD, val_mask)
        final_diag = out_final["diagnostic"]
        full_snapshot = rum.snapshot()

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
    }
    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[smoke] done in {elapsed:.0f}s ({len(history)} epochs)")
    print(f"  CPC            = {final_cpc:.4f}  (v2 baseline 0.485, v3a 0.534)")
    print(f"  alpha_wage     = {final_diag['alpha_wage']:+.4f}  (W_j; ≥0 by construction)")
    print(f"  gamma_M        = {final_diag['gamma_M']:+.4f}  (M_j; ≥0 by construction)")
    print(f"  nu_D           = {final_diag['nu_D']:+.4f}  (D_j; ≤0 by construction)")
    print(f"  delta_match    = {final_diag['delta_match']:+.4f}  (match_prob; ≥0 by construction)")
    print(f"  beta_t,0 mean  = {final_diag['beta_t_mean']:+.4f}  (t intercept; ≤0 by construction)")
    print(f"  beta_t,1 mean  = {final_diag['beta_t_slope_mean']:+.4f}  (t slope on log_d; ≤0 by construction)")
    print(f"  λ_b mean/std   = {final_diag['lambda_b_mean']:.3f} / {final_diag['lambda_b_std']:.3f}")
    if final_diag["blend"] is not None:
        print(f"  blend (δ_GNN)  = {final_diag['blend']:.4f}  (v3a 0.437)")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
