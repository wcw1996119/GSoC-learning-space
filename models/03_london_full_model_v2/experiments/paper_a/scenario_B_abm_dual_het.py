"""ABM-level Scenario B (flex work) using DUAL_HET head.

The original ``scenario_B_dual_het.py`` intervenes on aggregate features
(``congestion_z`` column + per-hour row sums) — this conflates flex-work's
*downstream* effects (congestion relief, demand redistribution) with the
*upstream* behavioural change (workers choosing different departure hours).

This script does the clean version:

  1. Sample each agent's ``departure_hour`` from the NTS commute distribution
     (baseline).
  2. For each agent, compute their personal ``V_ij`` using their known
     ``(home, hour, mode, tier)`` and the trained DUAL_HET head — *no*
     mixture, because mode and tier are known at the agent level.
  3. Sample workplace ``j ~ softmax(V_ij)``.
  4. Aggregate to ``F_ij_t``.
  5. Re-sample ``departure_hour`` for ``flex_frac`` of agents from a flat
     7-11am distribution (the upstream flex-work intervention).
  6. Aggregate to ``F_ij_t_flex``.
  7. Compare per-hour distribution, peak shift, destination concentration.

Output:
  evaluation_outputs/paper_a/scenario_B_abm_dual_het.json
  evaluation_outputs/paper_a/scenario_B_abm_dual_het_flows.npz   (F tensors)

Usage:
  python experiments/paper_a/scenario_B_abm_dual_het.py --flex_frac 0.30
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

from experiments.paper_a.dual_het_scenario_helpers import (   # noqa: E402
    load_dual_het_ckpt, build_scenario_inputs,
)


MODE_TO_IDX = {"car": 0, "pt": 1, "transit": 1, "active": 2, "walk": 2}


# ---------------------------------------------------------------- distributions
def load_nts_distribution() -> dict[int, float]:
    """Population departure-hour distribution from DfT NTS0502 (commute-purpose,
    weekday). Returns {hour: share}. Source: nts_commute_departure_time.csv."""
    df = pd.read_csv(
        ROOT / "data" / "processed" / "nts_commute_departure_time.csv",
        comment="#",
    )
    all_mode = df[df["mode"] == "all"]
    return {int(r["hour"]): float(r["share"]) for _, r in all_mode.iterrows()}


def flat_flex_distribution() -> dict[int, float]:
    """Flex-work hypothetical: morning hours 7-11 spread evenly, keeping the
    *total* morning mass identical to NTS. All other hours (evening peak,
    off-peak) preserved exactly.

    This is the correct semantics for "flex workers redistribute their
    departure across the morning window": same total commute volume, same
    evening pattern, just morning peak smoothed.
    """
    nts = load_nts_distribution()
    flex = dict(nts)
    morning_hours = list(range(7, 12))            # 7,8,9,10,11
    morning_total = sum(nts[h] for h in morning_hours)
    uniform_share = morning_total / len(morning_hours)
    for h in morning_hours:
        flex[h] = uniform_share
    # Sanity check renormalisation
    total = sum(flex.values())
    return {h: v / total for h, v in flex.items()}


def sample_hours(n: int, dist: dict, rng: np.random.Generator) -> np.ndarray:
    hours = np.array(sorted(dist.keys()))
    probs = np.array([dist[h] for h in hours], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(hours, size=n, p=probs)


# ---------------------------------------------------------------- agent V_ij
def compute_V_ij_vector(
    V_jt: torch.Tensor, t_per_mode: dict, log_d_ij: torch.Tensor,
    occ_match: torch.Tensor, head,
    home_idx: int, hour: int, mode_idx: int, tier_idx: int,
) -> torch.Tensor:
    """V_ij[j] for one agent. Closed-form mode×tier-specific (no mixture)
    because the agent's mode and tier are known.
    """
    alpha = head.alpha                              # scalar buffer = 1.0
    gamma = head.gamma                              # scalar
    beta_eff = head.beta_t_per_mode[hour, mode_idx] * head.kappa[tier_idx]
    delta_per_tier = head.delta_per_tier()          # (K,)

    mode_names = ["car", "transit", "walk"]
    t_m = t_per_mode[mode_names[mode_idx]]
    if t_m.dim() == 2:
        t_ij = t_m[home_idx]                        # (N,)
    else:
        t_ij = t_m[hour, home_idx]                  # (N,)

    V_ij = (alpha * V_jt[hour]
            + beta_eff * t_ij
            + gamma * log_d_ij[home_idx])
    if occ_match is not None:
        V_ij = V_ij + delta_per_tier[tier_idx] * occ_match[home_idx]
    return V_ij                                     # (N,)


def run_abm_forward(
    agents: list, V_jt: torch.Tensor, t_per_mode: dict,
    log_d_ij: torch.Tensor, occ_match: torch.Tensor, head,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Iterate agents sequentially, sample workplace, aggregate to (T,N,N) flow."""
    T = V_jt.shape[0]
    N = log_d_ij.shape[0]
    flow = torch.zeros(T, N, N)

    with torch.no_grad():
        for agent in agents:
            V_ij = compute_V_ij_vector(
                V_jt, t_per_mode, log_d_ij, occ_match, head,
                home_idx=agent["home_grid_idx"], hour=agent["departure_hour"],
                mode_idx=agent["mode_idx"], tier_idx=agent["tier_idx"],
            )
            log_p = torch.log_softmax(V_ij, dim=0)
            p = log_p.exp().numpy()
            p = p / p.sum()
            j = int(rng.choice(N, p=p))
            flow[agent["departure_hour"], agent["home_grid_idx"], j] += 1.0
    return flow


# ---------------------------------------------------------------- analysis
def gini_coefficient(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x)[x > 0])
    n = len(x)
    if n == 0:
        return 0.0
    return float((np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (n * x.sum()))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "dual_het_seed0.pt"))
    parser.add_argument("--agents_csv", type=str,
                        default=str(ROOT / "data" / "processed"
                                    / "agent_population.csv"))
    parser.add_argument("--flex_frac", type=float, default=0.30,
                        help="fraction of agents switched to flex distribution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[abm-scenB] loading DUAL_HET ckpt {args.ckpt}")
    enc, head, ckpt = load_dual_het_ckpt(Path(args.ckpt))
    d = build_scenario_inputs(walk_threshold_km=ckpt["config"]["walk_threshold_km"])
    N, T = d["N"], d["T"]
    print(f"[abm-scenB] N={N} T={T} edges={d['edge_index'].shape[1]}")

    # Forward encoder once to get V_jt
    norm_stats = enc.compute_norm_stats(d["X_static"], d["X_dynamic"], d["edge_index"])
    with torch.no_grad():
        V_jt = enc(d["X_static"], d["X_dynamic"], d["edge_index"], norm_stats=norm_stats)
    print(f"[abm-scenB] V_jt shape={tuple(V_jt.shape)} range="
          f"[{V_jt.min().item():.3f}, {V_jt.max().item():.3f}]")

    t_per_mode = {"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]}

    # Load agents
    agents_df = pd.read_csv(args.agents_csv)
    n_agents = len(agents_df)
    print(f"[abm-scenB] {n_agents} agents loaded")

    # ---- BASELINE: sample NTS-distributed departure hours ----------------
    nts_dist = load_nts_distribution()
    print(f"[abm-scenB] NTS peak hours: "
          f"{sorted(nts_dist.items(), key=lambda kv: -kv[1])[:3]}")
    nts_hours = sample_hours(n_agents, nts_dist, rng)

    agents_baseline = []
    for i, row in agents_df.iterrows():
        agents_baseline.append({
            "agent_id": int(row["agent_id"]),
            "home_grid_idx": int(row["home_grid_idx"]),
            "mode_idx": MODE_TO_IDX.get(row["mode_initial"], 0),
            "tier_idx": int(row["income_tier"]) - 1,
            "departure_hour": int(nts_hours[i]),
        })

    print(f"[abm-scenB] running baseline (NTS dist) ...")
    t0 = time.time()
    flow_baseline = run_abm_forward(agents_baseline, V_jt, t_per_mode,
                                     d["log_d"], d["occ_match"], head, rng)
    print(f"[abm-scenB] baseline: {time.time() - t0:.1f}s, "
          f"total trips: {flow_baseline.sum().item():.0f}")

    # ---- INTERVENTION: flex_frac agents get flat flex dist ---------------
    flex_dist = flat_flex_distribution()
    print(f"[abm-scenB] flex peak hours: "
          f"{sorted(flex_dist.items(), key=lambda kv: -kv[1])[:5]}")
    n_flex = int(n_agents * args.flex_frac)
    flex_idx_set = rng.choice(n_agents, size=n_flex, replace=False)
    flex_hours = sample_hours(n_flex, flex_dist, rng)
    agents_flex = [dict(a) for a in agents_baseline]
    for k, idx_agent in enumerate(flex_idx_set):
        agents_flex[int(idx_agent)]["departure_hour"] = int(flex_hours[k])

    print(f"[abm-scenB] running flex intervention ({n_flex} flex agents) ...")
    t0 = time.time()
    flow_flex = run_abm_forward(agents_flex, V_jt, t_per_mode,
                                  d["log_d"], d["occ_match"], head, rng)
    print(f"[abm-scenB] flex: {time.time() - t0:.1f}s, "
          f"total trips: {flow_flex.sum().item():.0f}")

    # ---- Analysis --------------------------------------------------------
    per_hour_base = flow_baseline.sum(dim=(1, 2)).numpy()
    per_hour_flex = flow_flex.sum(dim=(1, 2)).numpy()
    inflow_base = flow_baseline.sum(dim=(0, 1)).numpy()
    inflow_flex = flow_flex.sum(dim=(0, 1)).numpy()

    peak_base = int(per_hour_base.argmax())
    peak_flex = int(per_hour_flex.argmax())
    morning_peak_share_base = float(per_hour_base[7:10].sum() / per_hour_base.sum())
    morning_peak_share_flex = float(per_hour_flex[7:10].sum() / per_hour_flex.sum())
    morning_spread_base = float(per_hour_base[7:12].sum() / per_hour_base.sum())
    morning_spread_flex = float(per_hour_flex[7:12].sum() / per_hour_flex.sum())

    summary = {
        "config": {
            "ckpt": args.ckpt,
            "n_agents": int(n_agents),
            "flex_frac": args.flex_frac,
            "n_flex_agents": int(n_flex),
            "seed": args.seed,
        },
        "distributions": {
            "nts_peak_hours_top3": sorted(nts_dist.items(), key=lambda kv: -kv[1])[:3],
            "flex_peak_hours_top5": sorted(flex_dist.items(), key=lambda kv: -kv[1])[:5],
        },
        "baseline": {
            "total_trips": float(flow_baseline.sum().item()),
            "per_hour": per_hour_base.tolist(),
            "peak_hour": peak_base,
            "morning_peak_share_7_9am": morning_peak_share_base,
            "morning_spread_share_7_11am": morning_spread_base,
            "inflow_gini": gini_coefficient(inflow_base),
        },
        "flex": {
            "total_trips": float(flow_flex.sum().item()),
            "per_hour": per_hour_flex.tolist(),
            "peak_hour": peak_flex,
            "morning_peak_share_7_9am": morning_peak_share_flex,
            "morning_spread_share_7_11am": morning_spread_flex,
            "inflow_gini": gini_coefficient(inflow_flex),
        },
        "delta": {
            "peak_hour_shift": int(peak_flex - peak_base),
            "morning_peak_share_7_9am_change": morning_peak_share_flex - morning_peak_share_base,
            "morning_spread_share_7_11am_change": morning_spread_flex - morning_spread_base,
            "inflow_gini_change": gini_coefficient(inflow_flex) - gini_coefficient(inflow_base),
            "destination_correlation_pearson": float(
                np.corrcoef(inflow_base, inflow_flex)[0, 1]),
        },
    }

    out_path = out_dir / "scenario_B_abm_dual_het.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[abm-scenB] wrote {out_path}")

    flow_path = out_dir / "scenario_B_abm_dual_het_flows.npz"
    np.savez_compressed(flow_path,
                        F_baseline=flow_baseline.numpy().astype(np.float32),
                        F_flex=flow_flex.numpy().astype(np.float32))
    print(f"[abm-scenB] wrote {flow_path}")

    print(f"\n=== Summary ===")
    print(f"  baseline peak={peak_base} (7-9am share {morning_peak_share_base:.3f})")
    print(f"  flex     peak={peak_flex} (7-9am share {morning_peak_share_flex:.3f})")
    print(f"  Δ 7-9am peak share: {morning_peak_share_flex - morning_peak_share_base:+.3f}")
    print(f"  Δ 7-11am spread:    {morning_spread_flex - morning_spread_base:+.3f}")
    print(f"  destination corr:   {summary['delta']['destination_correlation_pearson']:.4f}")


if __name__ == "__main__":
    main()
