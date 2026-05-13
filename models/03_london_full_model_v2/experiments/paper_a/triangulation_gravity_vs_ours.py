"""D11: counterfactual triangulation — gravity vs Phase B v6 direction agreement.

Both models are asked the same question:
  "If we add +Δ jobs at cluster {grids}, how does ΔF distribute?"

We do not expect them to agree on magnitude (Phase B v6 has GNN spatial
spillover that gravity lacks), but they should at minimum agree on the
**sign** of major changes, and on whether the cluster gains net inflow.

Pipeline:
  1. Fit gravity (Wilson production-constrained softmax) on training origins.
  2. Predict F_pred_baseline_grav with baseline D_j.
  3. Bump D_j at cluster grids by Δ (raw), predict F_pred_scenario_grav.
  4. Compute ΔF_grav = scenario - baseline.
  5. Load Phase B v6 ΔF from scenario_A_total_employment_*_coherent_freeflow.json
     (the per-destination delta_inflow vector).
  6. Compute (a) sign agreement on top-K destinations by |ΔF|; (b) Spearman
     correlation across all destinations; (c) cluster ΔInflow comparison.

Output: evaluation_outputs/paper_a/triangulation_gravity_vs_ours.json
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

from experiments.paper_a.train_phase_b_v6_ckpt import load_data, cpc
from experiments.paper_a.spatial_holdout import HELDOUT_BOROUGHS, make_spatial_masks
from experiments.paper_a.baselines_spatial_holdout import fit_gravity_softmax
from experiments.paper_a.scenario_A_spatial import (
    apply_intervention_raw, apply_coherent_employment_intervention, EMP_BLOCK,
    feature_index,
)


def predict_gravity_with_D(D_j: torch.Tensor, t_ij: torch.Tensor, log_d: torch.Tensor,
                           O_i: torch.Tensor,
                           alpha: float, beta: float, gamma: float) -> torch.Tensor:
    """Apply fitted gravity (alpha, beta, gamma) to a (possibly intervened) D_j vector."""
    import torch.nn.functional as F
    log_D = torch.log(D_j.clamp(min=1.0))
    N = D_j.shape[0]
    log_D_b = log_D.unsqueeze(0).expand(N, N)
    V = alpha * log_D_b - beta * t_ij - gamma * log_d
    V = V.clone(); V.fill_diagonal_(-1e9)
    P = F.softmax(V, dim=1)
    return O_i.unsqueeze(1) * P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", required=True)
    parser.add_argument("--feature", type=str, default="total_employment")
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--top_k_compare", type=int, default=20)
    parser.add_argument("--ours_json", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "scenario_A_total_employment_7222_coherent_freeflow.json"))
    args = parser.parse_args()

    print("[D11] loading data ...")
    data = load_data()
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    train_mask, val_mask, mask_info = make_spatial_masks(cache, HELDOUT_BOROUGHS)

    F_ij = data["F_ij"]
    t_ij = data["t_ij"]
    log_d = data["log_d"]
    O_i = F_ij.sum(dim=1)

    df_grid = pd.read_csv(ROOT / "data" / "processed" / "grid_static_features.csv")
    grid_ids = cache["grid_ids"].tolist()
    df_grid = df_grid.set_index("grid_id").reindex(grid_ids)
    D_baseline = torch.tensor(df_grid["total_employment"].fillna(0.0).to_numpy(),
                              dtype=torch.float32)

    # Build scenario D_j (raw) — same coherent rule as D3, but applied to the
    # *raw* employment vector that gravity uses (no log+zscore round-trip needed
    # because gravity takes raw D_j directly).
    D_scenario = D_baseline.clone()
    if args.coherent:
        for g in args.grids:
            D_scenario[g] = float(D_baseline[g] * (1 + args.delta / max(float(D_baseline[g]), 1.0)))
    else:
        for g in args.grids:
            D_scenario[g] = float(D_baseline[g] + args.delta)

    print(f"[D11] cluster baseline emp = {D_baseline[args.grids].sum().item():,.0f}")
    print(f"[D11] cluster scenario emp = {D_scenario[args.grids].sum().item():,.0f} "
          f"(Δ = {(D_scenario[args.grids].sum() - D_baseline[args.grids].sum()).item():+,.0f})")

    # Fit gravity once on training origins
    print("[D11] fitting gravity ...")
    t0 = time.time()
    _, info = fit_gravity_softmax(F_ij, D_baseline, t_ij, log_d, train_mask,
                                  epochs=400, lr=0.05)
    print(f"  alpha={info['alpha']:.3f} beta={info['beta']:.4f} gamma={info['gamma']:.3f}  "
          f"loss={info['loss']:.4f}  t={time.time()-t0:.1f}s")

    # Recover predictions under baseline + scenario
    F_grav_base = predict_gravity_with_D(D_baseline, t_ij, log_d, O_i,
                                         info["alpha"], info["beta"], info["gamma"])
    F_grav_scen = predict_gravity_with_D(D_scenario, t_ij, log_d, O_i,
                                         info["alpha"], info["beta"], info["gamma"])
    delta_F_grav = (F_grav_scen - F_grav_base).cpu().numpy()
    dest_delta_grav = delta_F_grav.sum(axis=0)                          # (N,) per destination
    cluster_inflow_grav = float(delta_F_grav[:, args.grids].sum())

    # Phase B v6 ΔF (per destination) from saved JSON
    ours_path = Path(args.ours_json)
    if not ours_path.exists():
        raise FileNotFoundError(f"Phase B v6 scenario A JSON not found: {ours_path}; "
                                f"run scenario_A_spatial.py with same params first.")
    ours = json.load(open(ours_path))
    cluster_inflow_ours = float(ours["delta_inflow_into_intervened_grids"])

    # Reconstruct dest_delta_ours from top10_dest_gain + top10_dest_loss is sparse.
    # Better: re-run scenario_A for the per-destination vector. Since D3 already
    # writes only the top/bottom 10, we run a minimal scenario forward here too.
    from experiments.paper_a.scenario_A_spatial import (
        build_trainer_from_ckpt, predict_OD_with_X,
    )
    print("\n[D11] re-running Phase B v6 forward to get full per-dest ΔF ...")
    ckpt_path = ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"
    trainer, _ = build_trainer_from_ckpt(ckpt_path, data)
    norm_stats = trainer.gnn.compute_norm_stats(
        data["static"].unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )
    F_ours_base = predict_OD_with_X(trainer, data["static"], norm_stats=norm_stats).sum(0).cpu().numpy()
    if args.coherent:
        X_int = apply_coherent_employment_intervention(
            data["static"], args.grids, args.delta,
            cache["static_mean"], cache["static_std"],
        )
    else:
        X_int = apply_intervention_raw(
            data["static"], args.grids, args.feature, args.delta,
            cache["static_mean"], cache["static_std"],
        )
    F_ours_scen = predict_OD_with_X(trainer, X_int, norm_stats=norm_stats).sum(0).cpu().numpy()
    delta_F_ours = F_ours_scen - F_ours_base
    dest_delta_ours = delta_F_ours.sum(axis=0)

    # ============= comparison metrics =============
    # 1. Cluster inflow (both signs and magnitudes)
    print(f"\n[D11] === cluster ΔInflow comparison ===")
    print(f"  Phase B v6   : {cluster_inflow_ours:+.1f}")
    print(f"  Gravity      : {cluster_inflow_grav:+.1f}")
    sign_agree_cluster = (cluster_inflow_ours * cluster_inflow_grav) > 0

    # 2. Top-K destinations by |ΔF| union, sign agreement
    abs_o = np.abs(dest_delta_ours)
    abs_g = np.abs(dest_delta_grav)
    top_o = np.argsort(-abs_o)[:args.top_k_compare]
    top_g = np.argsort(-abs_g)[:args.top_k_compare]
    union = sorted(set(top_o.tolist()) | set(top_g.tolist()))
    sign_agree = np.sign(dest_delta_ours[union]) == np.sign(dest_delta_grav[union])
    sign_agree_pct = float(sign_agree.mean()) * 100

    # 3. Spearman across ALL grids
    from scipy.stats import spearmanr
    rho, p = spearmanr(dest_delta_ours, dest_delta_grav)

    # 4. Pearson on log-magnitude (robust to scale differences)
    log_o = np.log1p(np.maximum(abs_o, 0.0)) * np.sign(dest_delta_ours)
    log_g = np.log1p(np.maximum(abs_g, 0.0)) * np.sign(dest_delta_grav)
    pear = float(np.corrcoef(log_o, log_g)[0, 1])

    print(f"[D11] sign agreement on cluster inflow : {sign_agree_cluster}")
    print(f"[D11] sign agreement on top-{args.top_k_compare} |ΔF| destinations: "
          f"{int(sign_agree.sum())}/{len(union)}  ({sign_agree_pct:.1f}%)")
    print(f"[D11] Spearman ρ across all dests: {rho:+.4f}  (p = {p:.2e})")
    print(f"[D11] Pearson on signed log-magnitude: {pear:+.4f}")

    summary = {
        "grids": args.grids, "delta": args.delta, "coherent": args.coherent,
        "cluster_inflow_ours": cluster_inflow_ours,
        "cluster_inflow_gravity": cluster_inflow_grav,
        "sign_agree_cluster_inflow": bool(sign_agree_cluster),
        "top_k": args.top_k_compare,
        "top_k_union_size": len(union),
        "top_k_sign_agree_count": int(sign_agree.sum()),
        "top_k_sign_agree_pct": sign_agree_pct,
        "spearman_rho_all_dest": float(rho),
        "spearman_p_all_dest": float(p),
        "pearson_signed_log_mag": pear,
        "gravity_alpha": info["alpha"], "gravity_beta": info["beta"], "gravity_gamma": info["gamma"],
    }
    out_path = ROOT / "evaluation_outputs" / "paper_a" / "triangulation_gravity_vs_ours.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[D11] wrote {out_path}")


if __name__ == "__main__":
    main()
