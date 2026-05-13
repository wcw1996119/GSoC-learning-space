"""Frozen-component ablation for DUAL_HET (paper-A counterfactual prerequisite).

Question we answer: in the full DUAL_HET (encoder + RUM head), who is doing the
heavy lifting? If the RUM head carries the signal, counterfactual predictions
are economically grounded (β, γ map to a do(X) shock). If the GNN encoder
carries it, the model is closer to Mozolin 2000's "NN cannot extrapolate beyond
training sample" warning and the counterfactual stance is fragile.

Three configs:
  baseline    encoder + head both trained                                 (reuse distance_aware_results.json: 0.484±0.005)
  frozen_gnn  encoder.requires_grad=False at random init; only head trains
  frozen_rum  β = -0.05 / γ = -0.5 / κ = 1 / δ = 0 (literature priors,
              all RUM params requires_grad=False); only encoder trains

Reads:
  data/processed/{demo_cache.npz, grid_hourly_od_2019.npz,
                  grid_dynamic_features.npz, mode_inputs.npz,
                  paperA_v23_aux.npz, car_freeflow_t_ij.npy}

Writes:
  evaluation_outputs/paper_a/ablation_<config>.json   (per-seed metrics + recovered params)

Run:
  python experiments/paper_a/run_dual_het_ablation.py --config frozen_gnn --seeds 0
  python experiments/paper_a/run_dual_het_ablation.py --config frozen_rum --seeds 0
  python experiments/paper_a/run_dual_het_ablation.py --config frozen_gnn --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum.dual_branch_mixture_trainer import (  # noqa: E402
    DualBranchMixtureTrainer,
    make_distance_aware_mode_share,
)
from models_lib.inverse_rum.flow_metrics import all_metrics  # noqa: E402

from experiments.paper_a.compare_four_variants import load_data  # noqa: E402


LIT_BETA = -0.05          # literature time sensitivity (VOT-anchored)
LIT_GAMMA_NEG = -0.5      # literature distance decay (gravity tradition)
KAPPA_RAW_FOR_UNIT = -10.0  # softplus(-10) ≈ 0 ⇒ κ ≈ 1 for all tiers (no tier scaling)


def build_pair_mode_share(d):
    coords_path = ROOT / "data" / "processed" / "demo_cache.npz"
    coords = torch.from_numpy(
        np.load(coords_path, allow_pickle=True)["coords_bng"].astype(np.float32)
    )
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_km = (torch.linalg.norm(diff, dim=-1) / 1000.0).clamp(min=0.1)
    return make_distance_aware_mode_share(
        d["mode_share"], dist_km, ["car", "transit", "walk"], walk_threshold_km=5.0,
    )


def build_trainer(d, pair_mode_share, seed, config, epochs, patience, device="cpu",
                  irm_variant="none", irm_lambda=0.0, irm_warmup_epochs=0):
    fixed_beta = None
    if config == "frozen_rum":
        fixed_beta = {"car": LIT_BETA, "transit": LIT_BETA, "walk": LIT_BETA}

    use_gnn_blend = config in ("baseline_with_delta",)
    mode_specific_gamma = config in ("baseline_mode_gamma",)

    # IRM configs forward to underlying trainer regardless of base config
    # so we can stack 'baseline + IRM' or 'frozen_gnn + IRM' etc.
    trainer = DualBranchMixtureTrainer(
        X_static=d["X_static"], X_dynamic=d["X_dynamic"],
        edge_index=d["edge_index"], observed_OD=d["F_ij_t"],
        t_ij_per_mode={"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]},
        mode_share_per_origin=pair_mode_share,
        income_tier_props=d["income_tier_props"],
        log_d_ij=d["log_d"],
        occ_match=d["occ_match"],
        train_mask=d["train_mask"], val_mask=d["val_mask"],
        device=device, seed=seed,
        hidden_dim=32, gru_hidden=32, n_sage_layers=2, tcn_kernels=(3, 5, 7),
        lr_theta=1e-3, lr_rum=1e-2, weight_decay=1e-4,
        epochs=epochs, patience=patience,
        tier_specific_delta=True,
        fixed_beta_per_mode=fixed_beta,
        use_gnn_blend=use_gnn_blend,
        gnn_blend_init=0.5,
        mode_specific_gamma=mode_specific_gamma,
        irm_variant=irm_variant,
        irm_lambda=irm_lambda,
        irm_warmup_epochs=irm_warmup_epochs,
        origin_env_idx=d.get("grid_borough_idx"),
        verbose=False,
    )

    if config in ("baseline", "baseline_with_delta", "baseline_mode_gamma"):
        return trainer

    if config == "frozen_gnn":
        for p in trainer.encoder.parameters():
            p.requires_grad = False
        trainer.optimizer = torch.optim.AdamW(
            trainer.rum.parameters(), lr=1e-2, weight_decay=0.0,
        )
        return trainer

    if config == "frozen_rum":
        # γ default init is already inv_softplus(0.5) ⇒ γ ≈ -0.5 (LIT_GAMMA_NEG). OK.
        # δ default init = 0; raw_delta zeros ⇒ δ stays at 0. OK.
        # raw_kappa default init = 0 gives κ = 1 + softplus(0) ≈ 1.69 — NOT what we want.
        # Force κ = 1 for all tiers (no tier scaling) by pushing raw_kappa very negative.
        if trainer.rum.raw_kappa is not None:
            with torch.no_grad():
                trainer.rum.raw_kappa.data.fill_(KAPPA_RAW_FOR_UNIT)
        for p in trainer.rum.parameters():
            p.requires_grad = False
        trainer.optimizer = torch.optim.AdamW(
            trainer.encoder.parameters(), lr=1e-3, weight_decay=1e-4,
        )
        return trainer

    raise ValueError(f"unknown config: {config}")


def run_one_seed(d, pair_mode_share, seed, config, epochs, patience, device="cpu",
                 irm_variant="none", irm_lambda=0.0, irm_warmup_epochs=0):
    trainer = build_trainer(d, pair_mode_share, seed, config, epochs, patience, device=device,
                            irm_variant=irm_variant, irm_lambda=irm_lambda,
                            irm_warmup_epochs=irm_warmup_epochs)
    t0 = time.time()
    encoder, head, log = trainer.fit()
    fit_time = time.time() - t0
    pred = trainer.predict_OD()
    metrics = all_metrics(pred, d["F_ij_t"], d["val_mask"], K_top=10)
    bm = head.beta_t_per_mode.mean(dim=0).tolist()
    gnn_blend_val = float(head.gnn_blend.item()) if head.gnn_blend is not None else None
    gamma_val = head.gamma
    if gamma_val.dim() == 0:
        gamma_out = float(gamma_val.item())
    else:
        gamma_out = [float(x) for x in gamma_val.tolist()]

    # Diagnostic: cpc trajectory snapshots — verify whether IRM phase actually
    # moves the model away from the warm-up ERM checkpoint.
    cpc_trace = log.cpc_val if hasattr(log, "cpc_val") else []
    cpc_at_warmup_end = None
    if irm_warmup_epochs > 0 and len(cpc_trace) >= irm_warmup_epochs:
        cpc_at_warmup_end = float(cpc_trace[irm_warmup_epochs - 1])
    cpc_final = float(cpc_trace[-1]) if cpc_trace else None
    cpc_max = float(max(cpc_trace)) if cpc_trace else None
    cpc_max_epoch = int(cpc_trace.index(max(cpc_trace))) if cpc_trace else -1

    return {
        "seed": seed,
        "config": config,
        "cpc": float(metrics["cpc"]),
        "rmse": float(metrics["rmse"]),
        "pearson_r": float(metrics["pearson_r"]),
        "topK_acc_K10": float(metrics["topK_acc_K10"]),
        "beta_car": float(bm[0]),
        "beta_transit": float(bm[1]),
        "beta_walk": float(bm[2]),
        "gamma": gamma_out,
        "kappa": [float(x) for x in head.kappa.tolist()],
        "delta_per_tier": [float(x) for x in head.delta_per_tier().tolist()],
        "gnn_blend": gnn_blend_val,
        # IRM diagnostics — read these to tell whether penalty did anything:
        # If cpc_final ≈ cpc_at_warmup_end and cpc_max_epoch < warmup_end → IRM no-op.
        # If cpc_final differs from cpc_at_warmup_end → IRM moved the model.
        "cpc_at_warmup_end": cpc_at_warmup_end,
        "cpc_final": cpc_final,
        "cpc_max": cpc_max,
        "cpc_max_epoch": cpc_max_epoch,
        "fit_time_s": float(fit_time),
        "epochs_actual": int(len(log.train_nll)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        choices=["baseline", "baseline_with_delta",
                                 "baseline_mode_gamma",
                                 "frozen_gnn", "frozen_rum"],
                        required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--out_dir", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"))
    parser.add_argument("--cache_name", type=str, default="demo_cache.npz",
                        help="Which cache file under data/processed/ to load. "
                        "Use demo_cache_ext27.npz for extended features.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device: 'cpu' or 'cuda'. Colab T4 → 'cuda'.")
    # ---- IRM (Module 1, Paper A 2026-05-13) ----
    parser.add_argument("--irm_variant", type=str, default="none",
                        choices=["none", "rex", "irmv1"],
                        help="V-REx (Krueger 2021) or IRM-v1 (Arjovsky 2019). "
                        "'none' = standard ERM training.")
    parser.add_argument("--irm_lambda", type=float, default=0.0,
                        help="IRM penalty weight. 0 disables. Sweep "
                        "[0.1, 1, 10, 100, 1000] to find optimum.")
    parser.add_argument("--irm_warmup_epochs", type=int, default=50,
                        help="Epochs of λ=0 ERM warm-up before penalty kicks in. "
                        "Krueger 2021 recommends ~25%% of total epochs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Differentiate IRM runs to avoid clobbering when sweeping λ.
    irm_tag = ""
    if args.irm_variant != "none" and args.irm_lambda > 0:
        irm_tag = f"_irm_{args.irm_variant}_lam{args.irm_lambda:g}"
    out_path = out_dir / f"ablation_{args.config}{irm_tag}.json"

    print(f"[ablation] config={args.config}  seeds={args.seeds}  "
          f"epochs={args.epochs}  patience={args.patience}")
    print(f"[ablation] loading data from {args.cache_name} ...")
    d = load_data(cache_name=args.cache_name)
    pair_mode_share = build_pair_mode_share(d)
    print(f"[ablation] N={d['N']} T={d['T']} edges={d['edge_index'].shape[1]}")

    if args.irm_variant != "none" and args.irm_lambda > 0:
        print(f"[ablation] IRM ON: variant={args.irm_variant} λ={args.irm_lambda} "
              f"warmup={args.irm_warmup_epochs}ep envs={int(d['grid_borough_idx'].max()+1)}")

    results = []
    for seed in args.seeds:
        print(f"\n--- seed={seed} ---")
        r = run_one_seed(d, pair_mode_share, seed, args.config,
                         args.epochs, args.patience, device=args.device,
                         irm_variant=args.irm_variant,
                         irm_lambda=args.irm_lambda,
                         irm_warmup_epochs=args.irm_warmup_epochs)
        results.append(r)
        print(f"  CPC={r['cpc']:.4f}  RMSE={r['rmse']:.3f}  "
              f"Pearson r={r['pearson_r']:.3f}  topK={r['topK_acc_K10']:.3f}")
        blend_str = f"  δ_blend={r['gnn_blend']:.4f}" if r.get('gnn_blend') is not None else ""
        if isinstance(r['gamma'], list):
            gamma_str = f"γ=({','.join(f'{g:+.3f}' for g in r['gamma'])})"
        else:
            gamma_str = f"γ={r['gamma']:+.4f}"
        print(f"  β=({r['beta_car']:+.4f}, {r['beta_transit']:+.4f}, "
              f"{r['beta_walk']:+.4f})  {gamma_str}  "
              f"κ=[{', '.join(f'{k:.2f}' for k in r['kappa'])}]{blend_str}  "
              f"time={r['fit_time_s']:.0f}s")

    # Summary
    cpcs = [r["cpc"] for r in results]
    summary = {
        "config": args.config,
        "epochs": args.epochs,
        "patience": args.patience,
        "n_seeds": len(results),
        "cpc_mean": float(np.mean(cpcs)),
        "cpc_std": float(np.std(cpcs)) if len(cpcs) > 1 else 0.0,
        "irm_variant": args.irm_variant,
        "irm_lambda": args.irm_lambda,
        "irm_warmup_epochs": args.irm_warmup_epochs,
        "cache_name": args.cache_name,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ablation] CPC = {summary['cpc_mean']:.4f} "
          f"± {summary['cpc_std']:.4f}  (n={len(results)})")
    print(f"[ablation] wrote {out_path}")


if __name__ == "__main__":
    main()
