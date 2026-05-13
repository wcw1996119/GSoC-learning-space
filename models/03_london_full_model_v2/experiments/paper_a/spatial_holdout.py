"""D7 (spatial half): 4-borough leave-out holdout for Phase B v6.

Trains the Phase B v6 (sign-constrained mainstream) trainer with the val_mask
set to all grids in 4 held-out boroughs chosen for diversity:
  Westminster (CBD core), Hackney (inner non-CBD),
  Brent (outer high-density), Bromley (outer suburban).

Compares to the random every-5th-origin split CPC reported in
phase_b_v6_seed0.pt (saved baseline, val_CPC=0.3257) — the gap quantifies
how much of the trained CPC was an in-distribution artefact.

The spec also calls for a 3-hour leave-out cell. Phase B v6 uses T=1 (daily-
aggregated OD), so a time holdout requires a separate T=24 retrain (deferred
to a follow-up — flagged in the output JSON).

Run:
    python experiments/paper_a/spatial_holdout.py --seeds 3
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


HELDOUT_BOROUGHS = ["Westminster", "Hackney", "Brent", "Bromley"]


def make_spatial_masks(cache, heldout_names: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    boroughs = cache["boroughs"]
    gbi = cache["grid_borough_idx"]
    name_to_idx = {b: i for i, b in enumerate(boroughs)}
    held_idxs = [name_to_idx[n] for n in heldout_names]
    val_mask_np = np.isin(gbi, held_idxs)
    train_mask_np = ~val_mask_np
    info = {
        "heldout_boroughs": heldout_names,
        "n_train_grids": int(train_mask_np.sum()),
        "n_val_grids": int(val_mask_np.sum()),
        "n_total": int(len(gbi)),
    }
    return torch.tensor(train_mask_np), torch.tensor(val_mask_np), info


def train_one(seed: int, data: dict, edge_index: torch.Tensor,
              train_mask: torch.Tensor, val_mask: torch.Tensor) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    Fdim = data["static"].shape[-1]
    util_net = StructuralGNN(in_features=Fdim, hidden=128, out=1, depth=5)
    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=data["F_ij"],
        t_ij_t=data["t_ij"],
        log_d_ij=data["log_d"],
        K=50, utility_net=util_net,
        train_mask=train_mask, val_mask=val_mask,
        device="cpu", seed=seed,
        epochs=150, patience=150, residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=True,
        enforce_mainstream_direction=True,
    )
    t0 = time.time()
    theta, alpha, beta_t, beta_c, gamma, log = trainer.fit()
    train_seconds = time.time() - t0

    pred = trainer.predict_OD().detach().cpu().numpy()
    pred_F = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else (pred.sum(0) if pred.ndim == 3 else pred)
    F_obs_np = data["F_ij"].cpu().numpy()
    train_cpc = cpc(F_obs_np, pred_F, train_mask.cpu().numpy().astype(bool))
    val_cpc = cpc(F_obs_np, pred_F, val_mask.cpu().numpy().astype(bool))

    return {
        "seed": seed,
        "train_cpc": train_cpc,
        "val_cpc_spatial_holdout": val_cpc,
        "beta_base": float(beta_t.mean().item()),
        "beta_c": float(beta_c),
        "phi": float(trainer.rum.phi.item()),
        "psi": float(trainer.rum.psi.item()),
        "delta": float(trainer.rum.delta.item()),
        "gamma": float(gamma),
        "train_seconds": train_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    print("[D7] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)
    print(f"[D7] mask: train={mask_info['n_train_grids']} val={mask_info['n_val_grids']} "
          f"total={mask_info['n_total']} ({100.*mask_info['n_val_grids']/mask_info['n_total']:.1f}% held out)")
    print(f"[D7] held-out boroughs: {', '.join(HELDOUT_BOROUGHS)}")

    rows = []
    for seed in range(args.seeds):
        print(f"\n[D7] seed={seed} training ...")
        row = train_one(seed, data, edge_index, train_mask, val_mask)
        rows.append(row)
        print(f"  [OK] train_CPC={row['train_cpc']:.4f}  "
              f"val_CPC(spatial)={row['val_cpc_spatial_holdout']:.4f}  "
              f"beta={row['beta_base']:+.4f}  phi={row['phi']:+.4f} psi={row['psi']:+.4f}  "
              f"t={row['train_seconds']:.1f}s")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "evaluation_outputs" / "paper_a" / "spatial_holdout.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[D7] wrote {out_csv}")

    # Compare to random-split CPC saved in ckpt
    ckpt = torch.load(
        ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt",
        map_location="cpu", weights_only=False,
    )
    random_split_cpc = float(ckpt["metrics"]["val_cpc"])
    spatial_mean = float(df["val_cpc_spatial_holdout"].mean())
    spatial_std = float(df["val_cpc_spatial_holdout"].std()) if len(df) > 1 else 0.0
    print(f"\n[D7] === comparison ===")
    print(f"  random every-5th-origin split (saved ckpt) : CPC = {random_split_cpc:.4f}")
    print(f"  4-borough spatial holdout (mean ± std)     : CPC = {spatial_mean:.4f} +- {spatial_std:.4f}")
    print(f"  drop                                        : {(spatial_mean - random_split_cpc):+.4f}  "
          f"({100*(spatial_mean - random_split_cpc)/random_split_cpc:+.1f}% of random)")

    summary = {
        "heldout_boroughs": HELDOUT_BOROUGHS,
        "mask_info": mask_info,
        "n_seeds": args.seeds,
        "rows": rows,
        "random_split_cpc": random_split_cpc,
        "spatial_holdout_cpc_mean": spatial_mean,
        "spatial_holdout_cpc_std": spatial_std,
        "delta_cpc_pct_of_random": 100. * (spatial_mean - random_split_cpc) / random_split_cpc,
        "time_holdout_status": "deferred — Phase B v6 uses T=1 daily aggregation; needs T=24 retrain",
    }
    out_json = ROOT / "evaluation_outputs" / "paper_a" / "spatial_holdout.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D7] wrote {out_json}")


if __name__ == "__main__":
    main()
