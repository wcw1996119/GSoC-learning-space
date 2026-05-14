"""Build mode-level origin features X_{i, m} for v3 DM nested logit.

Each origin i × mode m gets F_mode features describing how well mode m connects
that origin to the job market. These are the X_{i, m} feeding into the upper-nest
inclusive value coupling and the lower-nest V_mode.

Default features (F_mode = 3):
    acc_m(i)         = log Σ_j jobs_j · exp(-β_ref · t_{ij, m})         (Hansen-style)
    mean_t_m(i)      = Σ_j t_{ij, m} · jobs_j / Σ_j jobs_j               (jobs-weighted mean)
    n_jobs_30min_m(i) = log(1 + Σ_{j : t_{ij,m} ≤ 30} jobs_j)            (30-min catchment)

Jobs proxy: total destination inflow Σ_t Σ_i F_{ij, t} from grid_hourly_od_2019.npz.
(Alternative: wage_score_per_dest from paperA_v23_aux.npz — pick via --jobs-source flag.)

Output:
    data/processed/mode_level_features.npz
      X         : (N=1725, M=3, F_mode=3) float32, standardised (mean 0, std 1) per feature
      raw_X     : (N, M, F_mode) before standardisation
      mean      : (M, F_mode) standardisation mean per (mode, feature)
      std       : (M, F_mode) standardisation std per (mode, feature)
      feature_names : list[str], length F_mode
      mode_names    : list[str], length M
      jobs_source   : str, e.g. 'OD_inflow' or 'wage_score'
      beta_ref      : float, Hansen accessibility decay parameter

Run:
    python experiments/paper_a/build_mode_level_features.py
        [--jobs-source OD_inflow|wage_score]
        [--beta-ref 0.1]
        [--out data/processed/mode_level_features.npz]

Reads v2 data via relative path; writes into v3's data/processed/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"


def build_features(
    t_per_mode: dict,           # {mode: (N, N) or (T, N, N)}
    jobs: np.ndarray,           # (N,)
    beta_ref: float = 0.1,
    time_threshold: float = 30.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute (N, M, F_mode) features per (origin, mode).

    For (T, N, N) inputs we average over T before computing features
    (mode-level features are time-invariant by design).
    """
    mode_names = list(t_per_mode.keys())
    M = len(mode_names)
    N = jobs.shape[0]
    assert jobs.ndim == 1 and jobs.shape[0] == N

    # Pre-compute jobs-clipped to avoid log(0)
    jobs_safe = np.maximum(jobs, 1e-6)                               # (N,)

    feature_names = ["acc", "mean_t", "n_jobs_30min"]
    F_mode = len(feature_names)
    X = np.zeros((N, M, F_mode), dtype=np.float32)

    for m_idx, m in enumerate(mode_names):
        t = t_per_mode[m]
        if t.ndim == 3:
            t = t.mean(axis=0)                                       # (N, N) time-averaged
        assert t.shape == (N, N)

        # --- Feature 1: Hansen accessibility ---
        # acc_m(i) = log Σ_j jobs_j · exp(-β_ref · t_{ij, m})
        # Use stable log-sum-exp form:
        #   acc = log Σ exp(log(jobs) - β·t)
        log_jobs = np.log(jobs_safe)                                 # (N,)
        log_term = log_jobs[None, :] - beta_ref * t                  # (N, N) j varies along axis -1
        # log-sum-exp over j: subtract max for stability
        max_j = log_term.max(axis=1, keepdims=True)
        acc = max_j.squeeze(1) + np.log(np.exp(log_term - max_j).sum(axis=1))   # (N,)

        # --- Feature 2: mean travel time, jobs-weighted ---
        # mean_t_m(i) = Σ_j t_{ij,m} · jobs_j / Σ_j jobs_j
        # Use ALL destinations (not just reachable) so feature is well-defined
        weights = jobs_safe / jobs_safe.sum()
        mean_t = (t * weights[None, :]).sum(axis=1)                  # (N,)

        # --- Feature 3: 30-min reachable jobs (log-transformed) ---
        # n_jobs_30min_m(i) = log(1 + Σ_{j : t ≤ 30} jobs_j)
        reachable = (t <= time_threshold).astype(np.float32)         # (N, N)
        n_jobs = (reachable * jobs_safe[None, :]).sum(axis=1)        # (N,)
        n_jobs_log = np.log1p(n_jobs)

        X[:, m_idx, 0] = acc
        X[:, m_idx, 1] = mean_t
        X[:, m_idx, 2] = n_jobs_log

    return X, feature_names


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise X per (mode, feature) to mean 0, std 1.

    Returns standardised X, mean, std (both shape (M, F_mode)).
    """
    mean = X.mean(axis=0)                                            # (M, F_mode)
    std = X.std(axis=0) + 1e-8                                       # (M, F_mode)
    X_std = (X - mean[None, :, :]) / std[None, :, :]
    return X_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jobs-source",
        choices=["OD_inflow", "wage_score"],
        default="OD_inflow",
        help="What to use as destination 'jobs' weight.",
    )
    ap.add_argument("--beta-ref", type=float, default=0.1,
                    help="Hansen decay coefficient (per minute).")
    ap.add_argument("--time-threshold", type=float, default=30.0,
                    help="Threshold (minutes) for 30-min reachable jobs feature.")
    ap.add_argument("--out", default="data/processed/mode_level_features.npz",
                    help="Output .npz path (relative to v3 root).")
    args = ap.parse_args()

    # --- load mode-specific travel times from v2 ---
    mode_path = V2_ROOT / "data" / "processed" / "mode_inputs.npz"
    print(f"Loading t_per_mode from {mode_path} ...")
    mode_data = np.load(mode_path)
    t_per_mode = {
        "car": mode_data["t_car"],
        "transit": mode_data["t_transit"],
        "walk": mode_data["t_walk"],
    }
    for m, t in t_per_mode.items():
        print(f"  t_{m}: shape={t.shape} dtype={t.dtype} mean={t.mean():.2f} min")

    # --- compute jobs proxy ---
    if args.jobs_source == "OD_inflow":
        od_path = V2_ROOT / "data" / "processed" / "grid_hourly_od_2019.npz"
        print(f"Computing jobs from OD inflow at {od_path} ...")
        od = np.load(od_path)
        F = od["F_ij_t"]                                             # (T, N, N)
        # Sum over time and origins: total inflow per destination
        jobs = F.sum(axis=(0, 1)).astype(np.float64)                 # (N,)
        print(f"  jobs (OD inflow): min={jobs.min():.1f} max={jobs.max():.1f} "
              f"mean={jobs.mean():.1f} pct_zero={(jobs==0).mean()*100:.1f}%")
    elif args.jobs_source == "wage_score":
        aux_path = V2_ROOT / "data" / "processed" / "paperA_v23_aux.npz"
        print(f"Computing jobs from wage_score at {aux_path} ...")
        aux = np.load(aux_path)
        jobs = aux["wage_score_per_dest"].astype(np.float64)
        # wage_score may be normalised; rescale to positive
        jobs = np.maximum(jobs - jobs.min() + 1e-3, 1e-3)
        print(f"  jobs (wage_score): min={jobs.min():.2f} max={jobs.max():.2f}")
    else:
        raise ValueError(args.jobs_source)

    # --- compute features ---
    print(f"\nBuilding features (β_ref={args.beta_ref}, t_threshold={args.time_threshold}) ...")
    raw_X, feature_names = build_features(
        t_per_mode, jobs, beta_ref=args.beta_ref, time_threshold=args.time_threshold,
    )
    print(f"  raw_X shape: {raw_X.shape}")

    # --- per (mode, feature) summary ---
    mode_names = list(t_per_mode.keys())
    print("\nRaw feature summary (per mode):")
    for f_idx, fname in enumerate(feature_names):
        for m_idx, m in enumerate(mode_names):
            vals = raw_X[:, m_idx, f_idx]
            print(f"  {fname:14s} mode={m:8s}: "
                  f"mean={vals.mean():+.3f} std={vals.std():.3f} "
                  f"min={vals.min():+.3f} max={vals.max():+.3f}")

    # --- standardise ---
    X_std, mean, std = standardise(raw_X)
    print(f"\nStandardised X shape: {X_std.shape}")

    # --- save ---
    out_path = V3_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_std,
        raw_X=raw_X.astype(np.float32),
        mean=mean,
        std=std,
        feature_names=np.array(feature_names),
        mode_names=np.array(mode_names),
        jobs_source=np.array(args.jobs_source),
        beta_ref=np.array(args.beta_ref),
        time_threshold=np.array(args.time_threshold),
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
