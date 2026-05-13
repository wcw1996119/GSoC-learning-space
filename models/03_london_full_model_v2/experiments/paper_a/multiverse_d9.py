"""D9 multiverse: 100 hyperparameter combinations to demonstrate robustness.

Sweeps over a grid of (hidden, depth, K, lr, layer_type, enforce_main, seed)
and reports the distribution of CPC. Goal: show that the headline result
is not cherry-picked.

Output:
  evaluation_outputs/paper_a/multiverse_d9.csv  (one row per config)
  evaluation_outputs/paper_a/multiverse_d9.json (summary stats)
"""
from __future__ import annotations

import argparse
import itertools
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


def train_eval(seed, hidden, depth, K, lr_theta, lr_rum, enforce_main,
               data, edge_index, train_mask, val_mask, epochs=120):
    torch.manual_seed(seed); np.random.seed(seed)
    Fdim = data["static"].shape[-1]
    util_net = StructuralGNN(in_features=Fdim, hidden=hidden, out=1, depth=depth)
    trainer = InverseRUMTrainer(
        grid_features=data["static"], edge_index=edge_index,
        observed_OD=data["F_ij"], t_ij_t=data["t_ij"], log_d_ij=data["log_d"],
        K=K, utility_net=util_net,
        train_mask=train_mask, val_mask=val_mask,
        device="cpu", seed=seed,
        epochs=epochs, patience=epochs, residualise=False,
        lr_theta=lr_theta, lr_rum=lr_rum,
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
    train_mask_np = train_mask.cpu().numpy().astype(bool)
    val_cpc = cpc(data["F_ij"].cpu().numpy(), F_pred, val_mask_np)
    train_cpc = cpc(data["F_ij"].cpu().numpy(), F_pred, train_mask_np)
    return {
        "seed": seed, "hidden": hidden, "depth": depth, "K": K,
        "lr_theta": lr_theta, "lr_rum": lr_rum, "enforce_main": enforce_main,
        "val_cpc": val_cpc, "train_cpc": train_cpc,
        "beta_base": float(beta_t.mean().item()),
        "phi": float(trainer.rum.phi.item()),
        "psi": float(trainer.rum.psi.item()),
        "delta": float(trainer.rum.delta.item()),
        "gamma": float(gamma),
        "train_seconds": dt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_configs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()

    print("[d9] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)

    # Grid
    HIDDENS = [64, 96, 128, 192]
    DEPTHS = [3, 4, 5, 6]
    Ks = [30, 50, 100]
    LR_RUMS = [0.01, 0.03, 0.1]
    ENFORCES = [True, False]
    SEEDS = list(range(5))

    full_grid = list(itertools.product(HIDDENS, DEPTHS, Ks, LR_RUMS, ENFORCES, SEEDS))
    rng = np.random.default_rng(42)
    if len(full_grid) > args.n_configs:
        chosen_idx = rng.choice(len(full_grid), size=args.n_configs, replace=False)
        chosen = [full_grid[i] for i in chosen_idx]
    else:
        chosen = full_grid
    print(f"[d9] sampled {len(chosen)} configs from {len(full_grid)}-cell grid")

    rows = []
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "multiverse_d9.csv"
    out_json = ROOT / "evaluation_outputs" / "paper_a" / "multiverse_d9.json"
    for i, (hid, dep, K, lrr, enf, sd) in enumerate(chosen):
        print(f"\n[d9] config {i+1}/{len(chosen)}: "
              f"hidden={hid} depth={dep} K={K} lr_rum={lrr} enf={enf} seed={sd}")
        try:
            r = train_eval(sd, hid, dep, K, 1e-3, lrr, enf, data, edge_index,
                            train_mask, val_mask, epochs=args.epochs)
            print(f"  val_CPC={r['val_cpc']:.4f}  beta={r['beta_base']:+.4f} "
                  f"t={r['train_seconds']:.1f}s")
            rows.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({"seed": sd, "hidden": hid, "depth": dep, "K": K,
                          "lr_theta": 1e-3, "lr_rum": lrr, "enforce_main": enf,
                          "val_cpc": float("nan"), "train_cpc": float("nan"),
                          "error": str(e)})
        # Save intermediately every 5 configs (in case of crash)
        if (i + 1) % 5 == 0 or i == len(chosen) - 1:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[d9] wrote {out_csv}")

    # Summary by enforce_main
    summary = {}
    for enf in [True, False]:
        sub = df[(df["enforce_main"] == enf) & df["val_cpc"].notna()]
        if len(sub) == 0:
            continue
        summary[f"enforce_main={enf}"] = {
            "n": int(len(sub)),
            "cpc_mean": float(sub["val_cpc"].mean()),
            "cpc_std": float(sub["val_cpc"].std()),
            "cpc_p10": float(sub["val_cpc"].quantile(0.10)),
            "cpc_p50": float(sub["val_cpc"].quantile(0.50)),
            "cpc_p90": float(sub["val_cpc"].quantile(0.90)),
            "cpc_min": float(sub["val_cpc"].min()),
            "cpc_max": float(sub["val_cpc"].max()),
        }
    summary["all_configs_above_gravity_baseline_0.17"] = float(
        (df["val_cpc"].dropna() > 0.17).mean() * 100
    )
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[d9] wrote {out_json}")
    print(f"\n[d9] === ROBUSTNESS SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
