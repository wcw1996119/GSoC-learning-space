"""External validation of v2.2 model travel times against DfT JTS benchmark.

For each grid cell, we extract a per-grid 'time to reach 5000 employment
opportunities' metric and compare against the DfT Journey Time Statistics
(JTS0501) which reports real-network PT and car times for the same metric
at MSOA level.

Method:
  1. Load JTS data (5000EmpPTt, 5000EmpCart per MSOA11CD).
  2. Map each grid -> primary MSOA21CD -> JTS time (treat MSOA codes
     as comparable; MSOA11/MSOA21 codes are largely identical for London).
  3. From demo_cache (t_ij_t at hour=8, total_employment per grid), compute
     per-grid model-predicted "median t_ij to reach 5000 jobs":
        sort destinations by t_ij ascending, take cumulative employment,
        find first j where cum_emp >= 5000, take t_ij there as the metric.
  4. Compare model_predicted vs JTS PT and Car: MAPE + Spearman + scatter.

Acknowledged limitation: the v2.2 t_ij is based on Euclidean*1.3 / 20 km/h
free-flow speed with a BPR multiplier; JTS uses observed network PT/car
times. Large MAPE is expected and informative — it quantifies the
model's transport-modeling abstraction.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V2_ROOT = Path(__file__).resolve().parent
PROC = V2_ROOT / "data" / "processed"
JTS_PATH = V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "jts0501_msoa_london.csv"
OUT_DIR = V2_ROOT / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)

EMP_TARGET = 5000.0
PEAK_HOUR = 8


def load_inputs():
    print("Loading JTS data ...")
    jts = pd.read_csv(JTS_PATH)
    print(f"  JTS rows: {len(jts)}; cols: {list(jts.columns)[:8]}")

    print("Loading grid -> MSOA primary mapping ...")
    grid_msoa = pd.read_csv(PROC / "grid_msoa_primary.csv")
    print(f"  grid->MSOA rows: {len(grid_msoa)}")

    print("Loading grid static features (employment) ...")
    feats = pd.read_csv(PROC / "grid_static_features.csv")
    feats = feats[["grid_id", "centroid_lat", "centroid_lon", "total_employment"]].copy()

    print("Loading demo_cache (t_ij_t) ...")
    cache = dict(np.load(PROC / "demo_cache.npz", allow_pickle=True))
    t_ij_t = cache["t_ij_t"]  # (T, N, N)
    print(f"  t_ij_t shape: {t_ij_t.shape}, dtype: {t_ij_t.dtype}")
    return jts, grid_msoa, feats, t_ij_t


def grid_borough_lookup() -> pd.DataFrame:
    p = PROC / "grid_borough_mapping.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["grid_id", "borough"])


def model_time_to_5000(
    t_row: np.ndarray, employment: np.ndarray, target: float = EMP_TARGET
) -> float:
    """Median t_ij over the smallest-time set j whose cumulative employment >= target.

    'Median' is taken over the j's that fall *within* the 5000-job basket.
    If total employment in London is < target (shouldn't happen), returns the
    max t_ij in the row.
    """
    order = np.argsort(t_row, kind="stable")
    sorted_t = t_row[order]
    sorted_emp = employment[order]
    cum_emp = np.cumsum(sorted_emp)
    # find first idx where cum_emp >= target
    hit = np.searchsorted(cum_emp, target, side="left")
    if hit >= len(cum_emp):
        return float(sorted_t[-1])
    # include destinations 0..hit (so cum_emp[hit] >= target)
    basket_t = sorted_t[: hit + 1]
    return float(np.median(basket_t))


def mape(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = (obs > 0) & np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - obs[mask]) / obs[mask]) * 100.0)


def main():
    jts, grid_msoa, feats, t_ij_t = load_inputs()

    # The N grids are ordered by feats — confirm
    n_grids = len(feats)
    print(f"\nN grids: {n_grids}; t_ij dims: {t_ij_t.shape}")
    assert t_ij_t.shape[1] == n_grids and t_ij_t.shape[2] == n_grids, (
        "t_ij_t inner dim must match grid_static_features row count"
    )

    # Use peak hour t_ij
    t_peak = np.asarray(t_ij_t[PEAK_HOUR], dtype=float)  # (N, N)
    employment = feats["total_employment"].to_numpy().astype(float)
    print(f"Total employment (sum): {employment.sum():.0f}; mean per grid: {employment.mean():.1f}")

    # Per-grid model predicted median time-to-5000
    print("\nComputing per-grid model-predicted time to 5000 jobs (peak hour) ...")
    pred_times = np.zeros(n_grids, dtype=float)
    for i in range(n_grids):
        pred_times[i] = model_time_to_5000(t_peak[i], employment, EMP_TARGET)

    feats = feats.copy()
    feats["model_t5000_min"] = pred_times

    # Merge with grid->MSOA, then with JTS
    print("\nMerging grid -> MSOA -> JTS ...")
    df = feats.merge(grid_msoa, on="grid_id", how="left")
    # Treat MSOA21 == MSOA11 (best-effort)
    df = df.merge(
        jts.rename(columns={"MSOA11CD": "MSOA21CD"})[["MSOA21CD", "5000EmpPTt", "5000EmpCart"]],
        on="MSOA21CD",
        how="left",
    )
    n_total = len(df)
    n_match = int(df["5000EmpPTt"].notna().sum())
    print(f"  matched {n_match}/{n_total} grids to a JTS MSOA")

    # Borough info
    bor = grid_borough_lookup()
    if not bor.empty:
        df = df.merge(bor, on="grid_id", how="left")

    df_valid = df.dropna(subset=["5000EmpPTt", "5000EmpCart"]).copy()
    pred = df_valid["model_t5000_min"].to_numpy()
    obs_pt = df_valid["5000EmpPTt"].to_numpy()
    obs_car = df_valid["5000EmpCart"].to_numpy()

    # --- Metrics ---
    mape_pt = mape(pred, obs_pt)
    mape_car = mape(pred, obs_car)
    rho_pt, _ = spearmanr(pred, obs_pt)
    rho_car, _ = spearmanr(pred, obs_car)
    bias_pt = float(np.mean(pred - obs_pt))
    bias_car = float(np.mean(pred - obs_car))

    print("\n=== JTS validation metrics (model peak-hour vs JTS) ===")
    print(f"  Grids compared: {len(df_valid)}")
    print(f"  vs JTS PT:  MAPE = {mape_pt:6.1f}% | Spearman ρ = {rho_pt:+.3f} | bias = {bias_pt:+.2f} min")
    print(f"  vs JTS Car: MAPE = {mape_car:6.1f}% | Spearman ρ = {rho_car:+.3f} | bias = {bias_car:+.2f} min")
    print(f"  Model mean t_5000 = {pred.mean():.1f} min  (sd {pred.std():.1f})")
    print(f"  JTS PT  mean      = {obs_pt.mean():.1f} min  (sd {obs_pt.std():.1f})")
    print(f"  JTS Car mean      = {obs_car.mean():.1f} min  (sd {obs_car.std():.1f})")

    # --- Top discrepancy areas ---
    df_valid["abs_err_pt"] = np.abs(pred - obs_pt)
    df_valid["abs_err_car"] = np.abs(pred - obs_car)
    if "borough" in df_valid.columns:
        bor_err = (
            df_valid.groupby("borough")[["abs_err_pt", "abs_err_car"]]
            .mean()
            .sort_values("abs_err_pt", ascending=False)
        )
        print("\nTop 10 boroughs by mean |error| vs JTS PT:")
        print(bor_err.head(10).round(2))
        top_disc = list(bor_err.head(5).index)
    else:
        top_disc = []
        print("\n(no borough mapping found; skipping per-borough discrepancy ranking)")

    # --- Scatter plot ---
    print("\nGenerating scatter plot ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, obs, label, mape_v, rho in [
        (axes[0], obs_pt, "JTS PT (min)", mape_pt, rho_pt),
        (axes[1], obs_car, "JTS Car (min)", mape_car, rho_car),
    ]:
        ax.scatter(obs, pred, s=8, alpha=0.4, edgecolor="none", color="#1f77b4")
        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(label)
        ax.set_ylabel("Model predicted time to 5,000 jobs (min, peak hour)")
        ax.set_title(f"Model vs {label.split()[1]}\nMAPE={mape_v:.1f}%, Spearman ρ={rho:.3f}, n={len(df_valid)}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("v2.2 model predicted travel time vs DfT JTS0501 (5,000 jobs metric)", fontsize=12)
    fig.tight_layout()
    scatter_path = OUT_DIR / "v22_jts_scatter.png"
    fig.savefig(scatter_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {scatter_path}")

    # --- Save numeric results ---
    out_npz = OUT_DIR / "v22_jts_validation.npz"
    np.savez_compressed(
        out_npz,
        grid_id=df_valid["grid_id"].to_numpy(),
        MSOA21CD=df_valid["MSOA21CD"].to_numpy().astype(str),
        model_t5000=pred,
        jts_pt=obs_pt,
        jts_car=obs_car,
        mape_pt=mape_pt,
        mape_car=mape_car,
        spearman_pt=rho_pt,
        spearman_car=rho_car,
        bias_pt=bias_pt,
        bias_car=bias_car,
        n_grids=len(df_valid),
        peak_hour=PEAK_HOUR,
        emp_target=EMP_TARGET,
        top_discrepancy_boroughs=np.array(top_disc, dtype=object),
    )
    print(f"  saved: {out_npz}")

    return {
        "mape_pt": mape_pt,
        "mape_car": mape_car,
        "rho_pt": rho_pt,
        "rho_car": rho_car,
        "bias_pt": bias_pt,
        "bias_car": bias_car,
        "n": len(df_valid),
        "top_disc": top_disc,
    }


if __name__ == "__main__":
    main()
