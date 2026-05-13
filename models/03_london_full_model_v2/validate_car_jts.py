"""External validation of the new OSMnx car travel-time matrix against JTS.

Stage A1 success criterion: this script's MAPE vs JTS Car should drop from
the v2.2 baseline of ~60% to under 25%, AND the mean predicted should be in
the same ballpark as the JTS Car mean (rather than 5x too small).

We compute "median time-to-reach-5000-jobs" per grid cell at peak hour (8am)
using the new free-flow matrix multiplied by destination-borough congestion,
mirroring `validate_jts.py` (which used the deprecated LondonBPRProvider).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))
PROC = V2_ROOT / "data" / "processed"
JTS_PATH = V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "jts0501_msoa_london.csv"
OUT_DIR = V2_ROOT / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)

EMP_TARGET = 5000.0
PEAK_HOUR = 8


def model_time_to_5000(t_row: np.ndarray, employment: np.ndarray, target: float = EMP_TARGET) -> float:
    """Employment-weighted mean t_ij over the 5000-job basket (JTS 5000EmpCart definition).

    Sort destinations by t_ij asc; accumulate employment; pro-rate the last
    destination so total employment in basket equals exactly `target`; return
    weighted mean of t_ij with weights = employment-in-basket / target.
    """
    order = np.argsort(t_row, kind="stable")
    sorted_t = t_row[order]
    sorted_emp = employment[order]
    cum = np.cumsum(sorted_emp)
    hit = int(np.searchsorted(cum, target, side="left"))
    if hit >= len(cum):
        # Total employment < target: weighted mean over all reachable
        if cum[-1] <= 0:
            return float(sorted_t[-1])
        return float(np.average(sorted_t, weights=sorted_emp))
    # Pro-rate destination at `hit`
    weights = sorted_emp[: hit + 1].astype(float).copy()
    overshoot = cum[hit] - target
    weights[-1] -= overshoot
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if weights.sum() <= 0:
        return float(sorted_t[hit])
    return float(np.average(sorted_t[: hit + 1], weights=weights))


def mape(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = (obs > 0) & np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - obs[mask]) / obs[mask]) * 100.0)


def main():
    print("[1/4] Loading inputs ...")
    feats = pd.read_csv(PROC / "grid_static_features.csv")[
        ["grid_id", "centroid_lat", "centroid_lon", "total_employment"]
    ].copy()
    grid_msoa = pd.read_csv(PROC / "grid_msoa_primary.csv")
    jts = pd.read_csv(JTS_PATH)

    car_path = PROC / "car_freeflow_t_ij.npy"
    if not car_path.exists():
        sys.exit(f"Missing {car_path} — run data/scripts/build_car_travel_matrix.py first.")
    t0_car = np.load(car_path).astype(np.float32)  # (N, N) free-flow minutes
    print(f"      car free-flow loaded: shape={t0_car.shape}")

    cong_long = pd.read_csv(PROC / "grid_hourly_congestion.csv")
    grid_id_order = feats["grid_id"].tolist()
    cong_wide = cong_long.pivot(index="grid_id", columns="hour", values="congestion_ratio").reindex(grid_id_order)
    cong_wide = cong_wide.fillna(1.20)
    cong_jt = cong_wide.to_numpy().astype(np.float32)  # (N, 24)

    # Apply peak-hour destination-side congestion
    t_peak = t0_car * cong_jt[None, :, PEAK_HOUR]      # (N, N) minutes at 8am

    employment = feats["total_employment"].to_numpy().astype(float)
    N = len(feats)

    print(f"[2/4] Computing time-to-5000 for {N} grids at hour={PEAK_HOUR} ...")
    pred = np.array([model_time_to_5000(t_peak[i], employment, EMP_TARGET) for i in range(N)])
    feats = feats.copy()
    feats["model_t5000_min"] = pred

    print("[3/4] Merging with JTS ...")
    df = feats.merge(grid_msoa, on="grid_id", how="left")
    df = df.merge(
        jts.rename(columns={"MSOA11CD": "MSOA21CD"})[["MSOA21CD", "5000EmpPTt", "5000EmpCart"]],
        on="MSOA21CD", how="left",
    )
    df_valid = df.dropna(subset=["5000EmpCart"]).copy()
    pred_v = df_valid["model_t5000_min"].to_numpy()
    obs_pt = df_valid["5000EmpPTt"].to_numpy()
    obs_car = df_valid["5000EmpCart"].to_numpy()

    print("[4/4] Computing metrics + plotting ...")
    mape_pt = mape(pred_v, obs_pt)
    mape_car = mape(pred_v, obs_car)
    rho_pt, _ = spearmanr(pred_v, obs_pt)
    rho_car, _ = spearmanr(pred_v, obs_car)
    bias_pt = float(np.mean(pred_v - obs_pt))
    bias_car = float(np.mean(pred_v - obs_car))

    print("\n=== JTS validation (Stage A1: OSMnx car) ===")
    print(f"  Grids compared: {len(df_valid)}")
    print(f"  Model mean t_5000 = {pred_v.mean():.2f} min  (sd {pred_v.std():.2f})")
    print(f"  JTS Car mean      = {obs_car.mean():.2f} min  (sd {obs_car.std():.2f})  ← Car is the apples-to-apples comparison")
    print(f"  JTS PT  mean      = {obs_pt.mean():.2f} min  (sd {obs_pt.std():.2f})")
    print()
    print(f"  vs JTS Car: MAPE = {mape_car:6.1f}% | Spearman ρ = {rho_car:+.3f} | bias = {bias_car:+.2f} min")
    print(f"  vs JTS PT:  MAPE = {mape_pt:6.1f}% | Spearman ρ = {rho_pt:+.3f} | bias = {bias_pt:+.2f} min")
    print()
    print("  Stage A1 success: vs Car MAPE < 25% AND |bias| < 5 min.")
    status = "PASS" if (mape_car < 25 and abs(bias_car) < 5) else "FAIL - diagnose"
    print(f"  Status: {status}")

    # Scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, obs, label, mape_v, rho in [
        (axes[0], obs_car, "JTS Car (min)", mape_car, rho_car),
        (axes[1], obs_pt, "JTS PT  (min)", mape_pt, rho_pt),
    ]:
        ax.scatter(obs, pred_v, s=8, alpha=0.4, edgecolor="none", color="#1f77b4")
        lo = float(min(obs.min(), pred_v.min()))
        hi = float(max(obs.max(), pred_v.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(label)
        ax.set_ylabel("OSMnx car predicted time to 5,000 jobs (min, peak hour)")
        ax.set_title(f"vs {label}\nMAPE={mape_v:.1f}%, ρ={rho:.3f}, n={len(df_valid)}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Stage A1 OSMnx car validation vs DfT JTS0501 (5,000-job basket)", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "v24_jts_car_scatter.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_png}")

    np.savez_compressed(
        OUT_DIR / "v24_jts_car_validation.npz",
        grid_id=df_valid["grid_id"].to_numpy(),
        MSOA21CD=df_valid["MSOA21CD"].to_numpy().astype(str),
        model_t5000=pred_v,
        jts_pt=obs_pt,
        jts_car=obs_car,
        mape_pt=mape_pt, mape_car=mape_car,
        spearman_pt=rho_pt, spearman_car=rho_car,
        bias_pt=bias_pt, bias_car=bias_car,
        n_grids=len(df_valid), peak_hour=PEAK_HOUR, emp_target=EMP_TARGET,
    )


if __name__ == "__main__":
    main()
