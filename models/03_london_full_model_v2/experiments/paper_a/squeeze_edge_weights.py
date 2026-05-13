"""Squeeze trick #3: edge-weighted GraphSAGE using exp(-t_ij/tau).

Same architecture as Phase B v6 but neighbour aggregation in GraphSAGE
weights edges by ``w_uv = exp(-t_ij[u,v] / tau)`` (with tau auto-set to
the mean travel time over edges). Closer-time neighbours dominate the
mean. Uniform aggregation = baseline (squeeze_ensemble.py).

Tests both constrained + unconstrained × 5 seeds, ensemble.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, cpc, load_data
from experiments.paper_a.spatial_holdout import HELDOUT_BOROUGHS, make_spatial_masks


def compute_edge_weights(edge_index: torch.Tensor, t_ij: torch.Tensor,
                         tau: float = None) -> tuple[torch.Tensor, float]:
    """exp(-t_ij[src,dst] / tau) per edge. Tau auto-set to mean over edges if None."""
    src, dst = edge_index[0], edge_index[1]
    if t_ij.dim() == 3:
        t_ij = t_ij.mean(0)
    t_edges = t_ij[src, dst]                                          # (E,)
    if tau is None:
        tau = float(t_edges.mean().item())
    w = torch.exp(-t_edges / max(tau, 1e-6))
    return w, tau


def train_one(seed: int, enforce_main: bool, edge_weights: torch.Tensor,
              data: dict, edge_index: torch.Tensor,
              train_mask: torch.Tensor, val_mask: torch.Tensor) -> tuple:
    torch.manual_seed(seed); np.random.seed(seed)
    Fdim = data["static"].shape[-1]
    util_net = StructuralGNN(in_features=Fdim, hidden=128, out=1, depth=5)
    # Inject edge weights as default — trainer doesn't need to know
    util_net.set_default_edge_weight(edge_weights)
    trainer = InverseRUMTrainer(
        grid_features=data["static"], edge_index=edge_index,
        observed_OD=data["F_ij"], t_ij_t=data["t_ij"], log_d_ij=data["log_d"],
        K=50, utility_net=util_net,
        train_mask=train_mask, val_mask=val_mask,
        device="cpu", seed=seed, epochs=150, patience=150, residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=True,
        enforce_mainstream_direction=enforce_main,
    )
    t0 = time.time()
    theta, alpha, beta_t, beta_c, gamma, log = trainer.fit()
    dt = time.time() - t0

    pred = trainer.predict_OD().detach().cpu().numpy()
    F_pred = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else (pred.sum(0) if pred.ndim == 3 else pred)
    val_mask_np = val_mask.cpu().numpy().astype(bool)
    val_cpc = cpc(data["F_ij"].cpu().numpy(), F_pred, val_mask_np)
    metrics = {
        "seed": seed, "constrained": enforce_main, "edge_weighted": True,
        "val_cpc_spatial": val_cpc,
        "beta_base": float(beta_t.mean().item()),
        "beta_c": float(beta_c),
        "phi": float(trainer.rum.phi.item()),
        "psi": float(trainer.rum.psi.item()),
        "delta": float(trainer.rum.delta.item()),
        "gamma": float(gamma),
        "train_seconds": dt,
    }
    return val_cpc, F_pred, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tau", type=float, default=None,
                        help="travel-time decay scale; auto = mean over edges")
    args = parser.parse_args()

    print("[squeeze-ew] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)
    F_obs_np = data["F_ij"].cpu().numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)

    edge_weights, tau = compute_edge_weights(edge_index, data["t_ij"], tau=args.tau)
    print(f"[squeeze-ew] edge_weights: tau={tau:.2f} min  "
          f"min={edge_weights.min().item():.4f} median={edge_weights.median().item():.4f} "
          f"max={edge_weights.max().item():.4f}")

    rows = []
    preds_constrained = []
    preds_unconstrained = []
    for cond_label, enforce in [("constrained", True), ("unconstrained", False)]:
        print(f"\n[squeeze-ew] === edge-weighted, condition: {cond_label} ===")
        for seed in range(args.seeds):
            val_cpc, F_pred, m = train_one(seed, enforce, edge_weights, data,
                                            edge_index, train_mask, val_mask)
            print(f"  seed={seed}: val_CPC={val_cpc:.4f}  beta={m['beta_base']:+.4f}  "
                  f"phi={m['phi']:+.4f}  psi={m['psi']:+.4f}  t={m['train_seconds']:.1f}s")
            rows.append(m)
            if enforce:
                preds_constrained.append(F_pred)
            else:
                preds_unconstrained.append(F_pred)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "squeeze_edge_weights.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[squeeze-ew] wrote {out_csv}")

    print(f"\n[squeeze-ew] === edge-weighted ENSEMBLE (mean over {args.seeds} seeds) ===")
    ens_c = np.mean(np.stack(preds_constrained, axis=0), axis=0)
    ens_u = np.mean(np.stack(preds_unconstrained, axis=0), axis=0)
    cpc_ens_c = cpc(F_obs_np, ens_c, val_mask_np)
    cpc_ens_u = cpc(F_obs_np, ens_u, val_mask_np)
    cpc_seed_c = df[df["constrained"] == True]["val_cpc_spatial"].agg(["mean", "std"])
    cpc_seed_u = df[df["constrained"] == False]["val_cpc_spatial"].agg(["mean", "std"])
    print(f"  constrained:   per-seed mean={cpc_seed_c['mean']:.4f} +- {cpc_seed_c['std']:.4f}  "
          f"|  ENSEMBLE={cpc_ens_c:.4f}")
    print(f"  unconstrained: per-seed mean={cpc_seed_u['mean']:.4f} +- {cpc_seed_u['std']:.4f}  "
          f"|  ENSEMBLE={cpc_ens_u:.4f}")

    print(f"\n[squeeze-ew] === comparison vs uniform-edge baseline ===")
    print(f"  uniform constrained ensemble:    0.2899  (squeeze_ensemble.json)")
    print(f"  uniform unconstrained ensemble:  0.3445  (squeeze_ensemble.json)")
    print(f"  edge-weighted constrained:       {cpc_ens_c:.4f}  (Δ {cpc_ens_c - 0.2899:+.4f})")
    print(f"  edge-weighted unconstrained:     {cpc_ens_u:.4f}  (Δ {cpc_ens_u - 0.3445:+.4f})")

    summary = {
        "n_seeds": args.seeds,
        "tau_minutes": float(tau),
        "edge_weight_stats": {
            "min": float(edge_weights.min()),
            "median": float(edge_weights.median()),
            "max": float(edge_weights.max()),
        },
        "constrained": {
            "per_seed_mean": float(cpc_seed_c["mean"]),
            "per_seed_std": float(cpc_seed_c["std"]),
            "ensemble": float(cpc_ens_c),
        },
        "unconstrained": {
            "per_seed_mean": float(cpc_seed_u["mean"]),
            "per_seed_std": float(cpc_seed_u["std"]),
            "ensemble": float(cpc_ens_u),
        },
        "uniform_baseline_constrained": 0.2899,
        "uniform_baseline_unconstrained": 0.3445,
    }
    out_json = ROOT / "evaluation_outputs" / "paper_a" / "squeeze_edge_weights.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[squeeze-ew] wrote {out_json}")


if __name__ == "__main__":
    main()
