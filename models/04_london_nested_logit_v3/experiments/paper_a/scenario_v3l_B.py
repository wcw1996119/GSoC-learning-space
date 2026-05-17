"""V3l Scenario B: Flex-work departure hour intervention.

Aggregate version (NOT agent-level ABM): for a fraction `flex_frac` of each
origin's commuters, the morning departure-hour distribution is replaced by
a flat 7-11 a.m. spread (keeping the morning-window total unchanged). The
remaining (1 - flex_frac) follow NTS0502 commute-purpose distribution.

Because v3l's P(j|i,t) marginalises out tier mixture, the aggregate result is
mathematically equivalent to running an ABM where each agent independently
samples its hour from the mixed distribution — but it's faster and simpler.

Usage:
    python experiments/paper_a/scenario_v3l_B.py \\
        --ckpt evaluation_outputs/paper_a/v3l_lexico_filter_repro_s0.pt \\
        --flex-frac 0.30 \\
        --out evaluation_outputs/paper_a/v3l_scenario_B_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import csv
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
load_data = _trainer.load_data
forward_cs = _trainer.forward_cs
make_distance_aware_mode_share = _trainer.make_distance_aware_mode_share
CerveroShenHead = _trainer.CerveroShenHead
DualBranchEncoder = _trainer.DualBranchEncoder
DualBranchGATEncoder = _trainer.DualBranchGATEncoder

# Reuse model reconstruction + scenario-input loading from scenario A
_spec_sa = _ilu.spec_from_file_location(
    "v3l_scenario_A_mod",
    V3_ROOT / "experiments" / "paper_a" / "scenario_v3l_A.py",
)
_sa = _ilu.module_from_spec(_spec_sa)
_spec_sa.loader.exec_module(_sa)
reconstruct_model = _sa.reconstruct_model
load_scenario_inputs = _sa.load_scenario_inputs
forward_to_flow = _sa.forward_to_flow
gini_coefficient = _sa.gini_coefficient


# DfT NTS0502 weekday commute-purpose departure-hour distribution (mode='all').
# Source: v2's data/processed/nts_commute_departure_time.csv — hardcoded here so
# this scenario can run on any machine without that data file.
NTS_HOURLY_SHARE = {
    0:  0.002, 1:  0.001, 2:  0.001, 3:  0.002, 4:  0.005, 5:  0.020,
    6:  0.060, 7:  0.135, 8:  0.150, 9:  0.060, 10: 0.025, 11: 0.020,
    12: 0.025, 13: 0.030, 14: 0.030, 15: 0.055, 16: 0.105, 17: 0.130,
    18: 0.075, 19: 0.030, 20: 0.015, 21: 0.010, 22: 0.008, 23: 0.006,
}


def load_nts_distribution() -> dict:
    return dict(NTS_HOURLY_SHARE)


def flex_morning_distribution(nts: dict, morning_hours=range(7, 12)) -> dict:
    """Flex distribution: morning hours (7-11 a.m.) spread uniformly, keeping the
    total morning mass identical to NTS. All other hours (evening peak, off-peak)
    preserved exactly.
    """
    flex = dict(nts)
    mh = list(morning_hours)
    morning_total = sum(nts[h] for h in mh)
    uniform = morning_total / len(mh)
    for h in mh:
        flex[h] = uniform
    total = sum(flex.values())
    return {h: v / total for h, v in flex.items()}


def mixed_dist(nts: dict, flex: dict, frac: float, n_hours: int = 24) -> np.ndarray:
    """frac fraction of agents follow flex; (1-frac) follow NTS. Returns length-T array."""
    out = np.zeros(n_hours, dtype=np.float64)
    for h in range(n_hours):
        nts_h = nts.get(h, 0.0)
        flex_h = flex.get(h, 0.0)
        out[h] = frac * flex_h + (1 - frac) * nts_h
    out = out / out.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--flex-frac", type=float, default=0.30,
                    help="Fraction of commuters switching to flex (uniform 7-11 a.m.) distribution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--aux-path", default="data/processed/paperA_v3_aux_cervero.npz")
    ap.add_argument("--out", required=True, type=str)
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device)
    print(f"[scen-B] device: {device}")

    print(f"[scen-B] loading {args_cli.ckpt}")
    ckpt = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
    print(f"        train CPC: {ckpt['final_cpc']:.4f}  seed={ckpt['args'].get('seed')}")
    encoder, rum, train_args = reconstruct_model(ckpt, device)

    train_args.aux_path = args_cli.aux_path
    data = load_scenario_inputs(train_args, device)
    N, T = data["N"], data["T"]
    print(f"        data: N={N} T={T}")

    # ---- Baseline forward (P(j|i,t) and F[t,i,j] using observed row totals) ----
    print("[scen-B] forward baseline ...")
    t0 = time.time()
    F_base, _ = forward_to_flow(encoder, rum, data, data["observed_OD"], data["val_mask"])
    print(f"        baseline forward {time.time()-t0:.1f}s")

    # Get P(j|i,t) directly so we can rescale by new row totals
    with torch.no_grad():
        out_base = forward_cs(
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
        P_base = out_base["log_P_D"].exp()    # (T, N, N)

    # Per-origin observed row totals per hour (workers leaving origin i at hour t)
    obs = data["observed_OD"]
    obs_row = obs.sum(dim=2)                       # (T, N) — observed per-(t,i) total

    # ---- Flex intervention -------------------------------------------------
    # For each origin i: a fraction `flex_frac` of MORNING (7-11 a.m.) commuters
    # spread their departure uniformly across hours 7-11. The remaining (1-frac)
    # follow observed within-morning distribution. All non-morning hours stay
    # at observed (no intervention outside the morning window).
    morning_hours = [7, 8, 9, 10, 11]
    obs_morning_total = obs_row[morning_hours, :].sum(dim=0)    # (N,) — Σ_{t∈M} obs[t,i]
    flex_per_hour = obs_morning_total / len(morning_hours)       # (N,) — uniform share
    new_row = obs_row.clone()
    for h in morning_hours:
        new_row[h] = (1.0 - args_cli.flex_frac) * obs_row[h] + args_cli.flex_frac * flex_per_hour
    row_scen = new_row                              # (T, N)

    # Sanity reference: NTS distribution (unused in main intervention, just shown
    # in JSON as descriptive context)
    nts = load_nts_distribution()
    flex_dist_only = flex_morning_distribution(nts, morning_hours=range(7, 12))

    # ---- Forward to flow with new row totals -------------------------------
    F_scen = P_base * row_scen.unsqueeze(-1)        # (T, N, N)

    # ---- Per-hour totals & peak analysis -----------------------------------
    per_hour_obs   = obs.sum(dim=(1, 2)).cpu().numpy()
    per_hour_base  = F_base.sum(dim=(1, 2)).cpu().numpy()
    per_hour_scen  = F_scen.sum(dim=(1, 2)).cpu().numpy()

    def _morn_peak(arr, hours=(7, 8, 9)):
        return float(sum(arr[h] for h in hours) / arr.sum())

    def _morn_spread(arr, hours=range(7, 12)):
        return float(sum(arr[h] for h in hours) / arr.sum())

    peak_base = int(np.argmax(per_hour_base))
    peak_scen = int(np.argmax(per_hour_scen))
    mp_base = _morn_peak(per_hour_base)
    mp_scen = _morn_peak(per_hour_scen)
    ms_base = _morn_spread(per_hour_base)
    ms_scen = _morn_spread(per_hour_scen)

    # ---- Per-destination redistribution (sum over t and i) -----------------
    inflow_base = F_base.sum(dim=(0, 1)).cpu().numpy()
    inflow_scen = F_scen.sum(dim=(0, 1)).cpu().numpy()
    # Most-affected destinations (any direction)
    abs_inflow_delta = np.abs(inflow_scen - inflow_base)
    top_changed = np.argsort(-abs_inflow_delta)[:15]

    summary = {
        "config": {
            "ckpt": args_cli.ckpt,
            "ckpt_cpc": float(ckpt["final_cpc"]),
            "flex_frac": args_cli.flex_frac,
            "morning_hours": morning_hours,
            "nts_peak_top3_descriptive": sorted(nts.items(), key=lambda kv: -kv[1])[:3],
            "flex_descriptive_per_hour_share": flex_dist_only[8],
            "intervention": "smooth_morning_only",
        },
        "per_hour": {
            "observed": per_hour_obs.tolist(),
            "model_base": per_hour_base.tolist(),
            "model_scen": per_hour_scen.tolist(),
        },
        "peak_metrics": {
            "peak_hour_base": peak_base,
            "peak_hour_scen": peak_scen,
            "peak_shift_hours": peak_scen - peak_base,
            "morning_peak_share_7_9_base": mp_base,
            "morning_peak_share_7_9_scen": mp_scen,
            "morning_peak_share_change": mp_scen - mp_base,
            "morning_spread_share_7_11_base": ms_base,
            "morning_spread_share_7_11_scen": ms_scen,
            "morning_spread_change": ms_scen - ms_base,
        },
        "destination": {
            "inflow_gini_base": gini_coefficient(inflow_base),
            "inflow_gini_scen": gini_coefficient(inflow_scen),
            "destination_correlation_pearson":
                float(np.corrcoef(inflow_base, inflow_scen)[0, 1]),
            "max_abs_inflow_delta": float(abs_inflow_delta.max()),
            "top_changed": [
                {"grid": int(g), "base": float(inflow_base[g]),
                 "scen": float(inflow_scen[g]),
                 "delta": float(inflow_scen[g] - inflow_base[g])}
                for g in top_changed
            ],
        },
        "totals": {
            "base_total": float(F_base.sum().item()),
            "scen_total": float(F_scen.sum().item()),
            "obs_total": float(obs.sum().item()),
        },
    }

    out_path = V3_ROOT / args_cli.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[scen-B] wrote {out_path}")

    print("\n=== Scenario B Summary ===")
    print(f"  Intervention: smooth_morning_only  flex_frac = {args_cli.flex_frac:.0%}  "
          f"morning_hours = {morning_hours}")
    print(f"  Peak hour: base={peak_base}  scen={peak_scen}  (shift {peak_scen-peak_base:+d}h)")
    print(f"  Morning 7-9 a.m. share: {mp_base:.3f} → {mp_scen:.3f}  "
          f"(Δ {mp_scen-mp_base:+.3f})")
    print(f"  Morning 7-11 a.m. share: {ms_base:.3f} → {ms_scen:.3f}  "
          f"(Δ {ms_scen-ms_base:+.3f})")
    print(f"  Destination corr. (inflow base vs scen): "
          f"{summary['destination']['destination_correlation_pearson']:.4f}")
    print(f"  Max single-grid inflow change: {abs_inflow_delta.max():.1f}")


if __name__ == "__main__":
    main()
