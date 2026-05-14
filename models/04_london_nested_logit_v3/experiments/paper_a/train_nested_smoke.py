"""v3 nested-logit smoke train (Phase 3 minimal).

Validates the hypothesis: does adding λ_m (per-mode destination softmax
temperature) increase V_RUM expressiveness and reduce GNN's learned blend
share δ?

Loads v2 data verbatim (relative path), uses v2's encoder + v2's distance-aware
mode share, but swaps the RUM head to v3's NestedLogitRUMHead and divides
per-mode logits by λ_m before destination softmax.

Compared to v2 baseline (3 seeds × 200ep, blend_max=1.0):
    v2 CPC = 0.485 ± 0.001
    v2 δ_free = 0.446 (learned, no cap)
    v2 has no λ_m (effectively λ_m = 1)

Run:
    python experiments/paper_a/train_nested_smoke.py
        [--epochs 50] [--seed 0] [--blend-max 1.0]
        [--lambda-init 0.99] [--lambda-eps-min 0.05]
        [--device cpu] [--verbose]

Output: evaluation_outputs/paper_a/nested_logit_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
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

# v2 owns the `models_lib` package namespace (since v3 borrows v2's encoder,
# data loader, helpers). Add v2 to sys.path so v2 imports resolve normally.
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

# Load v3-specific nested_logit_head by absolute path to avoid the
# `models_lib.inverse_rum` package collision between v2 and v3 trees.
import importlib.util
_nested_path = V3_ROOT / "models_lib" / "inverse_rum" / "nested_logit_head.py"
_spec = importlib.util.spec_from_file_location("v3_nested_logit_head", _nested_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NestedLogitRUMHead = _mod.NestedLogitRUMHead

from models_lib.inverse_rum.dual_branch_encoder import DualBranchEncoder  # v2
from models_lib.inverse_rum.dual_branch_mixture_trainer import (  # v2
    make_distance_aware_mode_share,
)
from experiments.paper_a.compare_four_variants import load_data  # v2 loader


def build_pair_mode_share(d):
    """Same recipe v2's run_dual_het_ablation uses."""
    coords_path = V2_ROOT / "data" / "processed" / "demo_cache.npz"
    coords = torch.from_numpy(
        np.load(coords_path, allow_pickle=True)["coords_bng"].astype(np.float32)
    )
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_km = (torch.linalg.norm(diff, dim=-1) / 1000.0).clamp(min=0.1)
    return make_distance_aware_mode_share(
        d["mode_share"], dist_km, ["car", "transit", "walk"], walk_threshold_km=5.0,
    )


# -------------------------------------------------------------- forward / loss

def per_hour_log_p_nested(
    encoder: DualBranchEncoder,
    rum: NestedLogitRUMHead,
    X_static: torch.Tensor,
    X_dynamic: torch.Tensor,
    edge_index: torch.Tensor,
    t_per_mode: Dict[str, torch.Tensor],
    mode_names: list,
    log_d_ij: torch.Tensor,
    occ_match: Optional[torch.Tensor],
    log_pi_mk_pair: torch.Tensor,
    enable_occ_match: bool,
) -> torch.Tensor:
    """log P(j | i, t) under nested logit (per-mode λ_m on destination softmax).

    Mirrors v2's _per_hour_log_p with one structural change:
        log_p_mk = log_softmax_j(V_total / λ_m)   instead of   log_softmax_j(V_total)

    Mixture step (logsumexp over (m, k) weighted by log π_m(i,j) + log π_k(i))
    is unchanged.
    """
    V_jt = encoder(X_static, X_dynamic, edge_index)
    T, N = V_jt.shape
    M = rum.n_modes
    K = rum.n_tiers

    alpha = rum.alpha
    gamma_raw = rum.gamma
    if gamma_raw.dim() == 0:
        gamma_vec = gamma_raw.unsqueeze(0).expand(M)
    else:
        gamma_vec = gamma_raw
    beta_per_mode = rum.beta_t_per_mode                 # (T, M)
    kappa = rum.kappa                                    # (K,)
    lambda_m = rum.lambda_per_mode                       # (M,)
    delta_k = rum.delta_per_tier() if enable_occ_match else None
    blend = rum.gnn_blend                                # scalar or None

    V_b = V_jt.view(T, 1, N).expand(T, N, N)
    d_b = log_d_ij.view(1, N, N)

    log_p = None
    for m, name in enumerate(mode_names):
        t_m = t_per_mode[name]                           # (T, N, N)
        for k in range(K):
            beta_eff = (beta_per_mode[:, m] * kappa[k]).view(T, 1, 1)
            V_rum = beta_eff * t_m + gamma_vec[m] * d_b
            if delta_k is not None and occ_match is not None:
                V_rum = V_rum + delta_k[k] * occ_match.view(1, N, N)
            V_gnn = alpha * V_b
            if blend is None:
                V_total = V_gnn + V_rum
            else:
                V_total = (1.0 - blend) * V_rum + blend * V_gnn

            # NESTED LOGIT: scale by 1/λ_m before softmax
            log_p_mk = torch.log_softmax(V_total / lambda_m[m], dim=-1)

            log_w_ij = log_pi_mk_pair[:, :, m, k].view(1, N, N)
            log_p_mk_weighted = log_p_mk + log_w_ij
            if log_p is None:
                log_p = log_p_mk_weighted
            else:
                log_p = torch.logaddexp(log_p, log_p_mk_weighted)
    return log_p


def nll(log_p: torch.Tensor, observed_OD: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    T, N, _ = log_p.shape
    mask_f = mask.view(1, N, 1).float()
    flow = observed_OD * mask_f
    denom = max(int((flow.sum(dim=2) > 0).sum()), 1)
    return -(flow * log_p).sum() / denom


def cpc(log_p: torch.Tensor, observed_OD: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        P = log_p.exp()
        row_sum = observed_OD.sum(dim=2, keepdim=True)
        pred = P * row_sum
        num = 2.0 * torch.minimum(pred[:, mask], observed_OD[:, mask]).sum()
        den = (pred[:, mask].sum() + observed_OD[:, mask].sum()).clamp(min=1.0)
        return float(num / den)


# -------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blend-max", type=float, default=1.0)
    ap.add_argument("--lambda-init", type=float, default=0.99)
    ap.add_argument("--lambda-eps-min", type=float, default=0.05)
    ap.add_argument("--lr-theta", type=float, default=1e-3)
    ap.add_argument("--lr-rum", type=float, default=1e-2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cache", default="demo_cache.npz")
    args = ap.parse_args()

    t_load = time.time()
    print(f"Loading v2 data from {V2_ROOT / 'data' / 'processed' / args.cache} ...")
    # v2 load_data uses ROOT-relative paths internally; we already sys.path'd v2
    # but it constructs ROOT from __file__ of compare_four_variants — that's v2.
    # So calling load_data() will read v2 paths correctly.
    d = load_data(cache_name=args.cache)
    pair_mode_share = build_pair_mode_share(d)
    print(f"  loaded in {time.time()-t_load:.1f}s  N={d['N']} T={d['T']}")

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N, T = d["N"], d["T"]
    M, K = pair_mode_share.shape[2], d["income_tier_props"].shape[1]
    mode_names = ["car", "transit", "walk"]

    X_static = d["X_static"].to(device).float()
    X_dynamic = d["X_dynamic"].to(device).float()
    edge_index = d["edge_index"].to(device)
    observed_OD = d["F_ij_t"].to(device).float()
    log_d_ij = d["log_d"].to(device).float()
    train_mask = d["train_mask"].to(device).bool()
    val_mask = d["val_mask"].to(device).bool()
    t_per_mode = {
        "car": d["t_car"].to(device).float(),
        "transit": d["t_transit"].to(device).float(),
        "walk": d["t_walk"].to(device).float(),
    }
    pi_m_pair = pair_mode_share.to(device).float()         # (N, N, M)
    pi_k = d["income_tier_props"].to(device).float()       # (N, K)
    occ_match = d["occ_match"].to(device).float()

    # Broadcast t to (T,N,N)
    for name, t in list(t_per_mode.items()):
        if t.dim() == 2:
            t_per_mode[name] = t.unsqueeze(0).expand(T, N, N).contiguous()

    # log priors per (i, j, m, k): (N, N, M, K)
    log_pi_m = torch.log(pi_m_pair.clamp(min=1e-9))
    log_pi_k = torch.log(pi_k.clamp(min=1e-9))
    log_pi_mk_pair = log_pi_m.unsqueeze(-1) + log_pi_k.view(N, 1, 1, K)

    encoder = DualBranchEncoder(
        static_dim=X_static.shape[1], dyn_dim=X_dynamic.shape[-1],
        hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
    ).to(device)

    rum = NestedLogitRUMHead(
        n_hours=T, n_modes=M, n_tiers=K,
        tier_specific_delta=True,
        use_gnn_blend=True,
        gnn_blend_init=0.5,
        blend_max=args.blend_max,
        lambda_init=args.lambda_init,
        lambda_eps_min=args.lambda_eps_min,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.lr_theta, "weight_decay": 1e-4},
            {"params": rum.parameters(), "lr": args.lr_rum, "weight_decay": 0.0},
        ]
    )

    print(f"\n[smoke] starting train: seed={args.seed} epochs={args.epochs} "
          f"blend_max={args.blend_max} λ_init={args.lambda_init}")
    print(f"        params: encoder {sum(p.numel() for p in encoder.parameters()):,} | "
          f"rum {sum(p.numel() for p in rum.parameters()):,}")

    t0 = time.time()
    best_vnll = float("inf")
    no_improve = 0
    best_state = None

    history = []
    for ep in range(args.epochs):
        encoder.train(); rum.train()
        optimizer.zero_grad()
        log_p = per_hour_log_p_nested(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, occ_match,
            log_pi_mk_pair, enable_occ_match=True,
        )
        loss = nll(log_p, observed_OD, train_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(rum.parameters()), 1.0
        )
        optimizer.step()

        encoder.eval(); rum.eval()
        with torch.no_grad():
            log_p_e = per_hour_log_p_nested(
                encoder, rum, X_static, X_dynamic, edge_index,
                t_per_mode, mode_names, log_d_ij, occ_match,
                log_pi_mk_pair, enable_occ_match=True,
            )
            vnll = float(nll(log_p_e, observed_OD, val_mask))
            cpc_val = cpc(log_p_e, observed_OD, val_mask)
            lam_now = rum.lambda_per_mode.detach().cpu().tolist()
            blend_now = float(rum.gnn_blend.detach()) if rum.gnn_blend is not None else None
            bm = rum.beta_t_per_mode.mean(dim=0).detach().cpu().tolist()

        history.append({
            "ep": ep,
            "tnll": float(loss.item()),
            "vnll": vnll,
            "cpc": cpc_val,
            "lambda_m": lam_now,
            "blend": blend_now,
            "beta_mean_per_mode": bm,
        })

        if args.verbose or ep % 5 == 0 or ep == args.epochs - 1:
            lam_str = "[" + ",".join(f"{x:.3f}" for x in lam_now) + "]"
            bm_str = "[" + ",".join(f"{x:+.3f}" for x in bm) + "]"
            blend_str = f"{blend_now:.3f}" if blend_now is not None else "n/a"
            print(f"ep {ep:3d} | tnll {loss.item():.3f} | vnll {vnll:.3f} | "
                  f"cpc {cpc_val:.3f} | λ_m={lam_str} | δ={blend_str} | β_mean={bm_str}")

        if vnll < best_vnll - 1e-4:
            best_vnll = vnll
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

    # ---- final eval at best state
    encoder.eval(); rum.eval()
    with torch.no_grad():
        log_p_final = per_hour_log_p_nested(
            encoder, rum, X_static, X_dynamic, edge_index,
            t_per_mode, mode_names, log_d_ij, occ_match,
            log_pi_mk_pair, enable_occ_match=True,
        )
        final_cpc = cpc(log_p_final, observed_OD, val_mask)
        final_lambda = rum.lambda_per_mode.detach().cpu().tolist()
        final_blend = float(rum.gnn_blend.detach()) if rum.gnn_blend is not None else None
        final_beta = rum.beta_t_per_mode.mean(dim=0).detach().cpu().tolist()
        final_gamma = float(rum.gamma.mean().detach())
        final_kappa = rum.kappa.detach().cpu().tolist()
        final_delta = rum.delta_per_tier().detach().cpu().tolist()

    elapsed = time.time() - t0
    out = {
        "config": "nested_logit_smoke",
        "seed": args.seed,
        "epochs_actual": len(history),
        "epochs_max": args.epochs,
        "blend_max": args.blend_max,
        "lambda_init": args.lambda_init,
        "lambda_eps_min": args.lambda_eps_min,
        "fit_time_s": elapsed,
        # final metrics
        "cpc": final_cpc,
        "lambda_per_mode": {m: l for m, l in zip(mode_names, final_lambda)},
        "gnn_blend": final_blend,
        "beta_per_mode_mean": {m: b for m, b in zip(mode_names, final_beta)},
        "gamma_mean": final_gamma,
        "kappa": final_kappa,
        "delta_per_tier": final_delta,
        # comparison anchor (v2 baseline)
        "v2_baseline_cpc": 0.485,
        "v2_baseline_blend_free": 0.446,
        "history": history,
    }

    out_dir = V3_ROOT / "evaluation_outputs" / "paper_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nested_logit_smoke.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[smoke] done in {elapsed:.0f}s ({len(history)} epochs)")
    print(f"  cpc            = {final_cpc:.4f}  (v2 baseline 0.485)")
    print(f"  λ_per_mode     = " + ", ".join(f"{m}:{l:.3f}" for m, l in zip(mode_names, final_lambda)))
    print(f"  gnn_blend (δ)  = {final_blend:.4f}  (v2 free baseline 0.446)")
    print(f"  β_per_mode     = " + ", ".join(f"{m}:{b:+.3f}" for m, b in zip(mode_names, final_beta)))
    print(f"  saved to {out_path}")

    # ---- hypothesis verdict ----
    print("\n[verdict]")
    lam_all_near_1 = all(abs(l - 1.0) < 0.05 for l in final_lambda)
    delta_dropped = (final_blend is not None) and (final_blend < 0.40)
    if lam_all_near_1:
        print("  λ_m all ≈ 1.0 → nested structure NOT doing meaningful work on this data")
    elif delta_dropped:
        print(f"  some λ_m < 1.0 AND δ dropped from 0.446 to {final_blend:.3f}")
        print("  → HYPOTHESIS SUPPORTED: nested logit increases RUM expressiveness,")
        print("    reduces GNN residual share")
    else:
        print(f"  some λ_m < 1.0 but δ = {final_blend:.3f} (not clearly lower than 0.446)")
        print("  → mixed signal: nested adds structure but doesn't displace GNN")


if __name__ == "__main__":
    main()
