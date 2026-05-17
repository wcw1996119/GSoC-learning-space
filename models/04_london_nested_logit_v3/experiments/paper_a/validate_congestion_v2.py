"""External validation v2: commute-attributable congestion in TomTom-instrumented boroughs.

Methodological refinement over validate_congestion.py:
  1. Filter out boroughs where TomTom data is fallback constant (1.2) — those
     boroughs have zero hourly variance and inflate the noise term.
  2. Attribute total congestion to commute purpose via NTS national hourly
     share: commute_share[hour] = c_h / (c_h × C_total + b_h × B_total)
     where c_h is the commute-purpose hour distribution (sum=1), b_h is the
     other-purpose hour distribution (sum=1), and C_total / B_total are
     annual trip counts by purpose (DfT NTS 2022 aggregate).
  3. Compute model predicted commute load and compare against attributed
     observed congestion, restricted to instrumented boroughs.

This is "Path E*" — the most rigorous achievable under public-data constraints
for London (we lack purpose-decomposed OD and weekend/weekday-split TomTom).

Usage:
    python experiments/paper_a/validate_congestion_v2.py \\
        --ckpt evaluation_outputs/paper_a/v3l_matchcut050_s0.pt \\
        --out evaluation_outputs/paper_a/v3l_congval2_s0.json
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


# ----------------------------------------------------------------------------
# DfT NTS 2022 hourly trip-purpose distributions
# ----------------------------------------------------------------------------
# Source: NTS0501 (Trips per person per year by purpose) +
#         NTS0502 (Hourly start-time distribution, by purpose).
# Commute (HBW) accounts for ~22% of total annual trips; other purposes 78%.
# Hourly distributions sum to 1 within each purpose.

# Commute purpose hourly share (sum=1, from v3l scenario B hardcoded NTS0502)
COMMUTE_HOURLY = np.array([
    0.002, 0.001, 0.001, 0.002, 0.005, 0.020,
    0.060, 0.135, 0.150, 0.060, 0.025, 0.020,
    0.025, 0.030, 0.030, 0.055, 0.105, 0.130,
    0.075, 0.030, 0.015, 0.010, 0.008, 0.006,
])

# Non-commute purpose hourly share (approx UK NTS2022, sum=1).
# Flatter than commute; midday peak (shopping/school-run); evening tail.
NON_COMMUTE_HOURLY = np.array([
    0.005, 0.002, 0.001, 0.001, 0.003, 0.010,
    0.020, 0.030, 0.045, 0.060, 0.070, 0.070,
    0.080, 0.075, 0.075, 0.080, 0.080, 0.080,
    0.070, 0.060, 0.050, 0.040, 0.030, 0.015,
])

COMMUTE_ANNUAL_SHARE = 0.22       # 22% of annual person-trips are commute
NON_COMMUTE_ANNUAL_SHARE = 0.78


def commute_share_per_hour() -> np.ndarray:
    """Compute commute_share[h] = commute_trips(h) / total_trips(h) for h=0..23."""
    c = COMMUTE_ANNUAL_SHARE * COMMUTE_HOURLY
    b = NON_COMMUTE_ANNUAL_SHARE * NON_COMMUTE_HOURLY
    return c / (c + b + 1e-12)


def load_grid_congestion(grid_id_to_idx: dict) -> np.ndarray:
    """Load grid_hourly_congestion.csv → (N, 24) congestion_ratio matrix."""
    N = len(grid_id_to_idx)
    cong = np.full((N, 24), np.nan, dtype=np.float64)
    cong_path = V2_ROOT / "data" / "processed" / "grid_hourly_congestion.csv"
    with open(cong_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            gid = row["grid_id"]
            if gid in grid_id_to_idx:
                cong[grid_id_to_idx[gid], int(row["hour"])] = float(row["congestion_ratio"])
    return cong


def load_grid_meta() -> tuple[dict, np.ndarray, list]:
    """Return (grid_id_to_idx, grid_borough_idx [N], borough_names [n_boroughs]).

    grid_id ordering follows grid_static_features.csv (the canonical one used by
    the model). Borough mapping is joined from grid_borough_mapping.csv.
    """
    static_path = V2_ROOT / "data" / "processed" / "grid_static_features.csv"
    grid_id_to_idx = {}
    with open(static_path, "r") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            grid_id_to_idx[row["grid_id"]] = i

    # Join borough mapping
    borough_map_path = V2_ROOT / "data" / "processed" / "grid_borough_mapping.csv"
    grid_to_borough = {}
    with open(borough_map_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            grid_to_borough[row["grid_id"]] = row["borough"]
    borough_per_grid = [grid_to_borough.get(g, "unknown") for g in grid_id_to_idx]
    unique_b = sorted(set(borough_per_grid))
    b_to_idx = {b: i for i, b in enumerate(unique_b)}
    borough_idx = np.array([b_to_idx[b] for b in borough_per_grid], dtype=np.int64)
    return grid_id_to_idx, borough_idx, unique_b


def identify_real_data_boroughs(cong: np.ndarray, borough_idx: np.ndarray,
                                 borough_names: list, var_threshold: float = 0.001
                                 ) -> tuple[set, dict]:
    """Find boroughs whose 24-hour congestion has variance > threshold.
    Fallback boroughs use a constant (typically 1.2) for all hours.
    """
    real_set = set()
    diag = {}
    for b_id, b_name in enumerate(borough_names):
        grids_in_b = np.where(borough_idx == b_id)[0]
        if len(grids_in_b) == 0:
            continue
        # Variance across hours, averaged across grids in this borough
        per_grid_var = np.nanvar(cong[grids_in_b], axis=1)  # (n_grids_in_b,)
        mean_var = float(np.nanmean(per_grid_var))
        is_real = mean_var > var_threshold
        diag[b_name] = {
            "n_grids": int(len(grids_in_b)),
            "mean_hourly_var": mean_var,
            "is_real": is_real,
        }
        if is_real:
            real_set.add(b_id)
    return real_set, diag


def correlation_stats(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 10:
        return {"label": label, "n": int(n), "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(xv, yv)[0, 1])
    rx = np.argsort(np.argsort(xv))
    ry = np.argsort(np.argsort(yv))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"label": label, "n": int(n), "pearson": pearson, "spearman": spearman}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--borough-var-thresh", type=float, default=0.001,
                    help="Min hourly variance to keep a borough as instrumented")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[congval2] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder, rum, train_args = reconstruct_model(ckpt, device)
    train_args.aux_path = "data/processed/paperA_v3_aux_cervero.npz"
    data = load_scenario_inputs(train_args, device)
    N, T = data["N"], data["T"]

    # --- Load borough mapping + TomTom congestion ---
    grid_id_to_idx, borough_idx, borough_names = load_grid_meta()
    assert len(grid_id_to_idx) == N, f"grid count mismatch {len(grid_id_to_idx)} vs {N}"
    cong = load_grid_congestion(grid_id_to_idx)
    print(f"[congval2] loaded TomTom congestion ({100*np.isfinite(cong).mean():.1f}% non-NaN)")

    # --- Identify TomTom-instrumented boroughs ---
    real_b_set, b_diag = identify_real_data_boroughs(
        cong, borough_idx, borough_names, var_threshold=args.borough_var_thresh
    )
    real_grid_mask = np.array([b in real_b_set for b in borough_idx])  # (N,)
    print(f"[congval2] instrumented boroughs: {len(real_b_set)}/{len(borough_names)}")
    for b_name in borough_names:
        d = b_diag[b_name]
        mark = "[REAL]" if d["is_real"] else "[fallback]"
        print(f"    {mark:11s} {b_name:30s}  hourly_var={d['mean_hourly_var']:.5f}  "
              f"n_grids={d['n_grids']}")
    print(f"    → kept {int(real_grid_mask.sum())} grids out of {N}")

    # --- Compute commute share per hour via NTS ---
    cs = commute_share_per_hour()
    print(f"\n[congval2] NTS commute share per hour (proxy):")
    for h in range(T):
        marker = "  *peak*" if h in (8, 9, 17, 18) else ""
        print(f"    hour {h:2d}: commute={COMMUTE_HOURLY[h]:.3f}  "
              f"other={NON_COMMUTE_HOURLY[h]:.3f}  share={cs[h]:.3f}{marker}")

    # --- Construct commute-attributable congestion ---
    # excess congestion above free-flow (1.0), weighted by commute share
    excess = cong - 1.0
    cong_commute_attr = 1.0 + excess * cs[np.newaxis, :]                # (N, T)

    # --- Run model forward ---
    print("\n[congval2] running model forward ...")
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
        P = out["log_P_D"].exp()
        row_total = data["observed_OD"].sum(dim=2, keepdim=True)
        F_pred = P * row_total
        model_inflow = F_pred.sum(dim=1).T.cpu().numpy()                # (N, T)
    print(f"        forward {time.time()-t0:.1f}s")

    obs = data["observed_OD"].cpu().numpy()                              # (T, N, N)
    obs_inflow = obs.sum(axis=1).T                                       # (N, T)

    # --- Correlations: 4 cells ---
    # (a) raw vs raw (baseline)
    # (b) raw vs commute-attributed (effect of attribution alone)
    # (c) instrumented boroughs vs raw congestion
    # (d) instrumented boroughs vs commute-attributed congestion  ← FINAL TARGET
    results = {
        "config": {
            "ckpt": args.ckpt,
            "ckpt_cpc": float(ckpt["final_cpc"]),
            "n_grids_total": int(N),
            "n_grids_instrumented": int(real_grid_mask.sum()),
            "n_boroughs_total": len(borough_names),
            "n_boroughs_instrumented": len(real_b_set),
            "borough_var_threshold": args.borough_var_thresh,
            "commute_annual_share": COMMUTE_ANNUAL_SHARE,
        },
        "borough_diagnostic": b_diag,
        "instrumented_boroughs": sorted(borough_names[b] for b in real_b_set),
        "commute_share_per_hour": cs.tolist(),
        "correlations": {},
    }

    def add_corr(key: str, x: np.ndarray, y: np.ndarray, label: str):
        results["correlations"][key] = correlation_stats(x, y, label)

    add_corr("a_full_obs_vs_raw",  obs_inflow.ravel(), cong.ravel(),              "(a) all grids · obs OD inflow vs raw congestion")
    add_corr("a_full_mod_vs_raw",  model_inflow.ravel(), cong.ravel(),            "(a) all grids · model inflow vs raw congestion")
    add_corr("b_full_obs_vs_attr", obs_inflow.ravel(), cong_commute_attr.ravel(), "(b) all grids · obs OD inflow vs commute-attr congestion")
    add_corr("b_full_mod_vs_attr", model_inflow.ravel(), cong_commute_attr.ravel(), "(b) all grids · model inflow vs commute-attr congestion")
    add_corr("c_inst_obs_vs_raw",  obs_inflow[real_grid_mask].ravel(),  cong[real_grid_mask].ravel(),
             "(c) instrumented · obs OD inflow vs raw congestion")
    add_corr("c_inst_mod_vs_raw",  model_inflow[real_grid_mask].ravel(), cong[real_grid_mask].ravel(),
             "(c) instrumented · model inflow vs raw congestion")
    add_corr("d_inst_obs_vs_attr", obs_inflow[real_grid_mask].ravel(),
             cong_commute_attr[real_grid_mask].ravel(),
             "(d) instrumented · obs OD inflow vs commute-attr congestion")
    add_corr("d_inst_mod_vs_attr", model_inflow[real_grid_mask].ravel(),
             cong_commute_attr[real_grid_mask].ravel(),
             "(d) instrumented · model inflow vs commute-attr congestion ← MAIN")

    # Per-hour, restricted to instrumented boroughs, commute-attributed
    per_hour = []
    for h in range(T):
        x_o = obs_inflow[real_grid_mask, h]
        x_m = model_inflow[real_grid_mask, h]
        y = cong_commute_attr[real_grid_mask, h]
        r_o = correlation_stats(x_o, y, f"hour {h} obs (inst, attr)")
        r_m = correlation_stats(x_m, y, f"hour {h} mod (inst, attr)")
        per_hour.append({
            "hour": h,
            "obs_inst_attr_pearson": r_o["pearson"],
            "mod_inst_attr_pearson": r_m["pearson"],
            "commute_share": float(cs[h]),
            "n": r_o["n"],
        })
    results["per_hour_instrumented_attributed"] = per_hour

    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[congval2] wrote {out_path}")

    print("\n=== Correlation Cells ===")
    for key, r in results["correlations"].items():
        print(f"  {r['label']:65s}  Pearson {r['pearson']:+.4f}  (n={r['n']})")

    print("\n=== Per-hour (instrumented boroughs, commute-attributed target) ===")
    print(f"  {'hour':>4s}  {'commute share':>13s}  {'obs corr':>8s}  {'model corr':>10s}")
    for ph in per_hour:
        h = ph["hour"]
        peak = "  *PEAK*" if h in (7, 8, 9, 17, 18) else ""
        print(f"  {h:>4d}  {ph['commute_share']:>13.3f}  "
              f"{ph['obs_inst_attr_pearson']:>+7.4f}  {ph['mod_inst_attr_pearson']:>+9.4f}{peak}")


if __name__ == "__main__":
    main()
