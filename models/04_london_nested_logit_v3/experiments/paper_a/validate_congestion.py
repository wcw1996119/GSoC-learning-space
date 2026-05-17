"""External validation: model-predicted grid inflow vs observed congestion.

Two-step sanity check using independent data sources:
  - Training OD: GEODS 2019 mobile-phone hourly origin-destination flow
  - Validation: TomTom-derived grid-hourly congestion ratio (from v2 pipeline)

These are NOT the same dataset. If the model captures the spatial-temporal flow
pattern correctly, its predicted grid inflow per hour should correlate with
TomTom-observed congestion at the same (grid, hour).

Two cells per cross-section:
  1. Observed OD inflow vs TomTom congestion (baseline — physics consistency)
  2. Model-predicted inflow vs TomTom congestion (model fit on EXTERNAL signal)

If (2) correlation ≈ (1) correlation → model captures observable signal as well
as ground-truth OD does, validating external generalisation.

Usage:
    python experiments/paper_a/validate_congestion.py \\
        --ckpt evaluation_outputs/paper_a/v3l_matchcut050_s0.pt \\
        --out  evaluation_outputs/paper_a/congestion_validation_s0.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"

if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "v3_train_cervero_shen",
    V3_ROOT / "experiments" / "paper_a" / "train_cervero_shen.py",
)
_trainer = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_trainer)
forward_cs = _trainer.forward_cs

_spec_sa = _ilu.spec_from_file_location(
    "v3l_scenario_A_mod",
    V3_ROOT / "experiments" / "paper_a" / "scenario_v3l_A.py",
)
_sa = _ilu.module_from_spec(_spec_sa)
_spec_sa.loader.exec_module(_sa)
reconstruct_model = _sa.reconstruct_model
load_scenario_inputs = _sa.load_scenario_inputs


def load_grid_congestion() -> np.ndarray:
    """Load grid_hourly_congestion.csv → (1725, 24) congestion_ratio matrix.
    Maps grid_id (e.g., 'L000_020') to integer index using same ordering as v2's
    grid_static_features (which the model uses).
    """
    static_path = V2_ROOT / "data" / "processed" / "grid_static_features.csv"
    grid_id_to_idx = {}
    with open(static_path, "r") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            grid_id_to_idx[row["grid_id"]] = i
    N = len(grid_id_to_idx)

    cong = np.full((N, 24), np.nan, dtype=np.float64)
    cong_path = V2_ROOT / "data" / "processed" / "grid_hourly_congestion.csv"
    with open(cong_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            gid = row["grid_id"]
            if gid not in grid_id_to_idx:
                continue
            idx = grid_id_to_idx[gid]
            hour = int(row["hour"])
            cong[idx, hour] = float(row["congestion_ratio"])
    return cong


def correlation_stats(x: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Pearson + Spearman correlation, dropping NaN cells."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 10:
        return {"name": name, "n": int(n), "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(xv, yv)[0, 1])
    # Spearman via manual rank (numpy only, no scipy dependency)
    rx = np.argsort(np.argsort(xv))
    ry = np.argsort(np.argsort(yv))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"name": name, "n": int(n), "pearson": pearson, "spearman": spearman,
            "x_mean": float(xv.mean()), "y_mean": float(yv.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[congestion-val] device: {device}")
    print(f"[congestion-val] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder, rum, train_args = reconstruct_model(ckpt, device)

    train_args.aux_path = "data/processed/paperA_v3_aux_cervero.npz"
    data = load_scenario_inputs(train_args, device)
    N, T = data["N"], data["T"]
    print(f"[congestion-val] data: N={N} T={T}")

    # Load TomTom congestion (independent dataset)
    print("[congestion-val] loading TomTom grid_hourly_congestion ...")
    cong = load_grid_congestion()                                    # (N, T)
    print(f"        cong: shape={cong.shape}  nan_pct={100*np.isnan(cong).mean():.1f}%  "
          f"range=[{np.nanmin(cong):.3f}, {np.nanmax(cong):.3f}]")

    obs = data["observed_OD"].cpu().numpy()                          # (T, N, N)
    obs_inflow_grid_hour = obs.sum(axis=1).T                         # (N, T) — observed inflow per (grid, hour)
    print(f"        observed inflow: range=[{obs_inflow_grid_hour.min():.0f}, "
          f"{obs_inflow_grid_hour.max():.0f}]")

    # Model forward
    print("[congestion-val] running model forward ...")
    t0 = time.time()
    with torch.no_grad():
        out = forward_cs(
            encoder, rum,
            data["X_static"], data["X_dynamic"], data["edge_index"],
            data["t_per_mode"], data["mode_names"],
            data["log_d_ij"], data["match_prob"],
            data["log_M_z"], data["log_W_z"], data["log_D_z"],
            data["income_score"], data["pct_kids"], data["mean_cars"],
            data["income_tier_props"],
            data["pi_m_pair"], data["grid_borough_idx"],
            data["observed_OD"], data["val_mask"],
        )
        P = out["log_P_D"].exp()                                     # (T, N, N)
        row_total = data["observed_OD"].sum(dim=2, keepdim=True)     # (T, N, 1)
        F_pred = P * row_total                                       # (T, N, N)
        model_inflow_grid_hour = F_pred.sum(dim=1).T.cpu().numpy()   # (N, T)
    print(f"        forward {time.time()-t0:.1f}s")
    print(f"        model inflow: range=[{model_inflow_grid_hour.min():.0f}, "
          f"{model_inflow_grid_hour.max():.0f}]")

    # --- Correlations ---------------------------------------------------------
    # Flatten (N, T) → (N*T,) for overall correlation
    results = {}
    results["overall_obs_vs_cong"] = correlation_stats(
        obs_inflow_grid_hour.ravel(), cong.ravel(),
        "Observed OD inflow vs TomTom congestion (overall)"
    )
    results["overall_model_vs_cong"] = correlation_stats(
        model_inflow_grid_hour.ravel(), cong.ravel(),
        "Model-predicted inflow vs TomTom congestion (overall)"
    )
    results["overall_model_vs_obs"] = correlation_stats(
        model_inflow_grid_hour.ravel(), obs_inflow_grid_hour.ravel(),
        "Model inflow vs Observed OD inflow (sanity)"
    )

    # Per-hour (key for commute peak validation)
    per_hour = []
    for h in range(T):
        r_obs = correlation_stats(obs_inflow_grid_hour[:, h], cong[:, h], f"hour {h} obs")
        r_mod = correlation_stats(model_inflow_grid_hour[:, h], cong[:, h], f"hour {h} model")
        per_hour.append({
            "hour": h,
            "obs_inflow_vs_cong_pearson": r_obs["pearson"],
            "model_inflow_vs_cong_pearson": r_mod["pearson"],
            "obs_n": r_obs["n"], "model_n": r_mod["n"],
        })
    results["per_hour"] = per_hour

    # Peak hours separately
    morning_peak_idx = (obs_inflow_grid_hour.sum(axis=0)).argmax()
    results["morning_peak_hour"] = int(morning_peak_idx)

    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[congestion-val] wrote {out_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Overall correlation (N*T flattened):")
    print(f"  Observed OD vs TomTom cong:  Pearson {results['overall_obs_vs_cong']['pearson']:+.4f}  "
          f"Spearman {results['overall_obs_vs_cong']['spearman']:+.4f}")
    print(f"  Model inflow vs TomTom cong: Pearson {results['overall_model_vs_cong']['pearson']:+.4f}  "
          f"Spearman {results['overall_model_vs_cong']['spearman']:+.4f}")
    print(f"  Model vs Observed (sanity):  Pearson {results['overall_model_vs_obs']['pearson']:+.4f}")

    print(f"\nPer-hour Pearson correlation with TomTom congestion:")
    print(f"  {'hour':>4s}  {'obs':>8s}  {'model':>8s}")
    for ph in per_hour:
        h, o, m = ph["hour"], ph["obs_inflow_vs_cong_pearson"], ph["model_inflow_vs_cong_pearson"]
        marker = "  *PEAK*" if h in (7, 8, 9, 17, 18) else ""
        print(f"  {h:>4d}  {o:>+7.4f}  {m:>+7.4f}{marker}")


if __name__ == "__main__":
    main()
