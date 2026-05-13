"""Diagnostic for Scenario A signal magnitude.

Questions:
  1. After +65k jobs at grid 908, does V_jt[908] actually go up?
  2. By how much, vs neighbour grids and the global mean shift?
  3. How does ΔV scale with delta_emp (10k, 65k, 200k, 1M)?
  4. Where does the signal dissipate — feature dilution, GNN smoothing, or
     the post-hoc zero-mean/unit-std normalisation in StructuralGNN?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, load_data
from experiments.paper_a.scenario_A_spatial import (
    apply_intervention_raw, build_trainer_from_ckpt,
)


def main() -> None:
    ckpt_path = ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"
    data = load_data()
    trainer, _ = build_trainer_from_ckpt(ckpt_path, data)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    mu = np.asarray(cache["static_mean"])
    sd = np.asarray(cache["static_std"])

    # Compute baseline norm stats (frozen)
    norm_stats = trainer.gnn.compute_norm_stats(
        data["static"].unsqueeze(0), trainer.edge_index
    )
    print(f"frozen baseline norm_stats: mean={float(norm_stats[0]):+.4f} std={float(norm_stats[1]):.4f}")

    # Baseline V_jt under frozen norm
    with torch.no_grad():
        V_base_frozen = trainer.gnn(data["static"], trainer.edge_index, norm_stats=norm_stats).cpu().numpy()
        V_base_self = trainer.gnn(data["static"], trainer.edge_index).cpu().numpy()
        if V_base_frozen.ndim == 2:
            V_base_frozen = V_base_frozen[0]; V_base_self = V_base_self[0]
    print(f"V_base[frozen]: mean={V_base_frozen.mean():+.4f} std={V_base_frozen.std():.4f}")
    print(f"V_base[self]:   mean={V_base_self.mean():+.4f} std={V_base_self.std():.4f}")
    # The two should match exactly (since baseline forward produces the same stats it gets normalised by)
    print(f"|frozen - self|.max() = {np.abs(V_base_frozen - V_base_self).max():.2e}")

    target_grid = 908
    print(f"\n=== Δemp scan at grid {target_grid}: FROZEN baseline norm ===")
    print(f"{'delta_raw':>12} | {'V[908]':>8} | {'ΔV[908]':>10} | {'ΔV[mean]':>10} | {'ΔV[952]':>10}")

    for delta in (0, 10_000, 65_000, 200_000, 1_000_000):
        X_int = apply_intervention_raw(
            data["static"], [target_grid], "total_employment", float(delta), mu, sd,
        )
        with torch.no_grad():
            V_int = trainer.gnn(X_int, trainer.edge_index, norm_stats=norm_stats).cpu().numpy()
            if V_int.ndim == 2:
                V_int = V_int[0]
        dv_target = V_int[target_grid] - V_base_frozen[target_grid]
        dv_mean = (V_int - V_base_frozen).mean()
        dv_n = V_int[952] - V_base_frozen[952]
        print(f"{delta:>12,} | {V_int[target_grid]:>+8.3f} | {dv_target:>+10.4f} | "
              f"{dv_mean:>+10.4f} | {dv_n:>+10.4f}")

    print(f"\n=== Δemp scan at grid {target_grid}: SELF (re-normalised per-call, OLD bug) ===")
    print(f"{'delta_raw':>12} | {'V[908]':>8} | {'ΔV[908]':>10} | {'ΔV[mean]':>10} | {'ΔV[952]':>10}")
    for delta in (0, 10_000, 65_000, 200_000, 1_000_000):
        X_int = apply_intervention_raw(
            data["static"], [target_grid], "total_employment", float(delta), mu, sd,
        )
        with torch.no_grad():
            V_int = trainer.gnn(X_int, trainer.edge_index).cpu().numpy()
            if V_int.ndim == 2:
                V_int = V_int[0]
        dv_target = V_int[target_grid] - V_base_self[target_grid]
        dv_mean = (V_int - V_base_self).mean()
        dv_n = V_int[952] - V_base_self[952]
        print(f"{delta:>12,} | {V_int[target_grid]:>+8.3f} | {dv_target:>+10.4f} | "
              f"{dv_mean:>+10.4f} | {dv_n:>+10.4f}")

    # Show 5 nearest neighbours of 908 in V_jt response space
    print(f"\n=== ΔV map at delta=200000, top-10 grids by |ΔV| ===")
    X_int = apply_intervention_raw(
        data["static"], [target_grid], "total_employment", 200_000.0, mu, sd,
    )
    with torch.no_grad():
        V_int = trainer.gnn(X_int, trainer.edge_index).cpu().numpy()
        if V_int.ndim == 2:
            V_int = V_int[0]
    dv = V_int - V_base
    top_idx = np.argsort(-np.abs(dv))[:10]
    print(f"{'grid_idx':>10} | {'ΔV':>10} | {'is_target':>10}")
    for i in top_idx:
        tag = "*" if i == target_grid else ""
        print(f"{i:>10} | {dv[i]:>+10.4f} | {tag:>10}")

    # Quick sanity: make sure intervention modified the right column
    print(f"\nSanity: X[908, 8] baseline={data['static'][target_grid, 8]:.3f}, "
          f"after +65k={apply_intervention_raw(data['static'], [target_grid], 'total_employment', 65000., mu, sd)[target_grid, 8]:.3f}")


if __name__ == "__main__":
    main()
