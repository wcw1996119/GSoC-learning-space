"""External validation of the heuristic PT travel-time fallback against JTS.

Mirrors `validate_car_jts.py` exactly: at peak hour (8 am) compute the
employment-weighted-mean time-to-reach-5000-jobs per grid cell using
`pt_t_ij_t_fallback[8]`, aggregate to MSOA21CD primary mapping, and compare
against the JTS0501 `5000EmpPTt` column.

Target: MAPE vs JTS PT < 30 % (PT is harder than car; r5py would be more
accurate but the heuristic should be in the right ballpark).
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
JTS_PATH = (
    V2_ROOT.parent
    / "02_london_commuting_model"
    / "data"
    / "processed"
    / "jts0501_msoa_london.csv"
)
OUT_DIR = V2_ROOT / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)

EMP_TARGET = 5000.0
PEAK_HOUR = 8
PT_MATRIX = PROC / "pt_t_ij_t_fallback.npy"


def model_time_to_5000(t_row: np.ndarray, employment: np.ndarray,
                        target: float = EMP_TARGET) -> float:
    """Employment-weighted mean t_ij over the 5000-job basket
    (matches JTS 5000EmpPTt definition; copied from validate_car_jts.py)."""
    order = np.argsort(t_row, kind="stable")
    sorted_t = t_row[order]
    sorted_emp = employment[order]
    cum = np.cumsum(sorted_emp)
    hit = int(np.searchsorted(cum, target, side="left"))
    if hit >= len(cum):
        if cum[-1] <= 0:
            return float(sorted_t[-1])
        return float(np.average(sorted_t, weights=sorted_emp))
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


def main() -> None:
    print("[1/4] Loading inputs ...")
    feats = pd.read_csv(PROC / "grid_static_features.csv")[
        ["grid_id", "centroid_lat", "centroid_lon", "total_employment"]
    ].copy()
    grid_msoa = pd.read_csv(PROC / "grid_msoa_primary.csv")
    jts = pd.read_csv(JTS_PATH)

    if not PT_MATRIX.exists():
        sys.exit(
            f"Missing {PT_MATRIX} — run "
            f"data/scripts/build_pt_travel_matrix_fallback.py first."
        )
    pt_t_ij_t = np.load(PT_MATRIX).astype(np.float32)
    print(f"      PT fallback loaded: shape={pt_t_ij_t.shape}")
    if pt_t_ij_t.ndim != 3 or pt_t_ij_t.shape[0] != 24:
        sys.exit(f"Expected (24, N, N); got {pt_t_ij_t.shape}")

    t_peak = pt_t_ij_t[PEAK_HOUR]                         # (N, N) at 8 am
    employment = feats["total_employment"].to_numpy().astype(float)
    N = len(feats)
    if t_peak.shape != (N, N):
        sys.exit(f"PT slice {t_peak.shape} mismatches grid count N={N}")

    print(f"[2/4] Computing time-to-5000 for {N} grids at hour={PEAK_HOUR} ...")
    pred = np.array(
        [model_time_to_5000(t_peak[i], employment, EMP_TARGET) for i in range(N)]
    )
    feats = feats.copy()
    feats["model_t5000_min"] = pred

    print("[3/4] Merging with JTS ...")
    df = feats.merge(grid_msoa, on="grid_id", how="left")
    df = df.merge(
        jts.rename(columns={"MSOA11CD": "MSOA21CD"})[
            ["MSOA21CD", "5000EmpPTt", "5000EmpCart"]
        ],
        on="MSOA21CD",
        how="left",
    )
    df_valid = df.dropna(subset=["5000EmpPTt"]).copy()
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

    print("\n=== JTS PT validation (heuristic fallback, peak hour) ===")
    print(f"  Grids compared: {len(df_valid)}")
    print(f"  Model mean t_5000 = {pred_v.mean():.2f} min  (sd {pred_v.std():.2f})")
    print(f"  JTS PT  mean      = {obs_pt.mean():.2f} min  (sd {obs_pt.std():.2f})  "
          f"<- apples-to-apples comparison")
    print(f"  JTS Car mean      = {obs_car.mean():.2f} min  (sd {obs_car.std():.2f})")
    print()
    print(f"  vs JTS PT:  MAPE = {mape_pt:6.1f}% | Spearman rho = {rho_pt:+.3f} | "
          f"bias = {bias_pt:+.2f} min")
    print(f"  vs JTS Car: MAPE = {mape_car:6.1f}% | Spearman rho = {rho_car:+.3f} | "
          f"bias = {bias_car:+.2f} min")
    print()
    print("  Target: vs PT MAPE < 30 % (heuristic fallback target).")
    status = "PASS" if mape_pt < 30 else "FAIL - tune speeds/wait or move to r5py"
    print(f"  Status: {status}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, obs, label, mape_v, rho in [
        (axes[0], obs_pt, "JTS PT  (min)", mape_pt, rho_pt),
        (axes[1], obs_car, "JTS Car (min)", mape_car, rho_car),
    ]:
        ax.scatter(obs, pred_v, s=8, alpha=0.4, edgecolor="none", color="#2ca02c")
        lo = float(min(obs.min(), pred_v.min()))
        hi = float(max(obs.max(), pred_v.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(label)
        ax.set_ylabel("Heuristic PT predicted time to 5,000 jobs (min, peak hour)")
        ax.set_title(f"vs {label}\nMAPE={mape_v:.1f}%, rho={rho:.3f}, n={len(df_valid)}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "PT heuristic fallback validation vs DfT JTS0501 (5,000-job basket)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "v24_pt_jts_scatter.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_png}")

    np.savez_compressed(
        OUT_DIR / "v24_pt_jts_validation.npz",
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
