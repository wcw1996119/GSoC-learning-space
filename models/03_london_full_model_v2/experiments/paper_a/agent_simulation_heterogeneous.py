"""HET: agent-level heterogeneity simulation + Jensen bias measurement.

Two predictions for the same intervention (Scenario A: 9-grid OOC cluster
+~65k jobs):

  AGGREGATE prediction:
    F_pred(j) = Σ_i O_i · softmax_j(α·V_j + β_eff(i,j)·t_ij + γ·log_d + δ·OM(i,j))
    — single β_eff per (i,j) pair, no within-cell variation.

  AGENT-LEVEL prediction:
    For each agent n at home i_n:
      ε_n ~ N(0, σ²(income_tier_n))
      β_n(j) = β_base · (1 + φ·z_o(i_n) + ψ·z_w(j) + ε_n)
      P_n(j) = softmax_j(α·V_j + β_n(j)·t(i_n,j) + γ·log_d + δ·OM(n,j))
    F_pred(j) = Σ_n P_n(j)
    — true within-cell variation; no Jensen bias.

Difference between the two = empirical Jensen bias for our model.

Scenario A intervention applied to BOTH:
  ΔF_aggregate = F_aggregate(scenario) − F_aggregate(baseline)
  ΔF_agent     = F_agent(scenario)     − F_agent(baseline)
  bias_intervention = ΔF_aggregate − ΔF_agent

Output:
  evaluation_outputs/paper_a/het_agent_vs_aggregate.json
  + per-grid Δ comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import (
    InverseRUMTrainer, StructuralGNN,
    AgentBetaConfig, DEFAULT_COV_BY_TIER, WEBTAG_VOT_BY_TIER_GBP_PER_HOUR,
    assign_per_agent_epsilon, per_agent_beta,
    simulate_agent_choice_softmax, aggregate_per_agent_to_OD,
)
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, load_data
from experiments.paper_a.scenario_A_spatial import (
    apply_coherent_employment_intervention, apply_intervention_raw,
    build_trainer_from_ckpt, predict_OD_with_X, EMP_BLOCK,
)


STRATFORD_TEST_GRIDS = [907, 908, 909, 862, 863, 864, 951, 952, 953]   # 9-cluster OOC proxy
DEFAULT_DELTA = 7222   # per-grid emp delta (×9 ≈ 65k total)


def run_aggregate_prediction(trainer, X_input, norm_stats):
    """Aggregate model prediction over all (i, j) — single β per pair."""
    pred = predict_OD_with_X(trainer, X_input, norm_stats=norm_stats)
    return pred.sum(0).cpu().numpy()                                # (N, N)


def run_agent_prediction(
    trainer, X_input, norm_stats,
    agents_df, occ_match_io, t_ij, log_d_ij,
    income_score_per_origin, wage_score_per_dest,
    config: AgentBetaConfig,
) -> np.ndarray:
    """Per-agent simulation using recovered (β_base, φ, ψ, δ, γ, α) plus
    per-agent ε draw. Returns aggregated (N, N) flow matrix."""
    # Forward GNN to get V_j (under intervention if X_input differs)
    V_j_t = trainer._forward_V(X_input.unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
                                norm_stats=norm_stats)
    V_j = V_j_t.mean(0).detach().cpu().numpy()                       # (N,) day-mean

    # Recovered scalars
    beta_base = float(trainer.rum.beta_t.mean().item())
    phi = float(trainer.rum.phi.item())
    psi = float(trainer.rum.psi.item())
    delta = float(trainer.rum.delta.item())
    gamma = float(trainer.rum.gamma.item())

    # Per-agent inputs
    home_idx = agents_df["home_grid_idx"].to_numpy()
    income_tier = agents_df["income_tier"].to_numpy()
    z_o_per_agent = income_score_per_origin[home_idx]                # (N_agents,)
    eps = assign_per_agent_epsilon(income_tier, config)              # (N_agents,)

    # β_eff(n, j) — (N_agents, N_grids)
    beta_eff = per_agent_beta(beta_base, phi, psi,
                               z_o_per_agent, wage_score_per_dest, eps)

    # Per-agent t_io (origin row of t_ij), log_d_io, occ_match_io (per-agent occ row)
    t_io = t_ij[home_idx]                                            # (N_agents, N_grids)
    log_d_io = log_d_ij[home_idx]
    om_io = occ_match_io[home_idx]                                   # crude proxy: use origin-row OM

    # Per-agent softmax
    P = simulate_agent_choice_softmax(
        V_j=V_j, t_io=t_io, log_d_io=log_d_io, occ_match_io=om_io,
        beta_eff_per_agent=beta_eff, delta=delta, gamma=gamma,
        alpha_V=1.0, home_grid_idx_per_agent=home_idx,
    )                                                                # (N_agents, N_grids)

    # Aggregate to (N_grids, N_grids) OD
    F_agent = aggregate_per_agent_to_OD(P, home_idx, n_grids=V_j.shape[0])
    return F_agent, eps, {
        "beta_base": beta_base, "phi": phi, "psi": psi,
        "delta": delta, "gamma": gamma,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a"
                                    / "phase_b_v6_seed0.pt"))
    parser.add_argument("--grids", type=int, nargs="+", default=STRATFORD_TEST_GRIDS)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--coherent", action="store_true", default=True)
    parser.add_argument("--cov", type=float, default=0.40,
                        help="coefficient of variation on β per income tier (Wardman 2014)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[het] loading data + ckpt + agents ...")
    data = load_data()
    trainer, ckpt = build_trainer_from_ckpt(Path(args.ckpt), data)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    aux = np.load(ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    mu = np.asarray(cache["static_mean"]); sd = np.asarray(cache["static_std"])

    income_score_per_origin = np.asarray(aux["income_score_per_origin"])
    wage_score_per_dest = np.asarray(aux["wage_score_per_dest"])
    occ_match = np.asarray(aux["occ_match"])

    agents_df = pd.read_csv(ROOT / "data" / "processed" / "agent_population_v23.csv")
    print(f"[het] loaded {len(agents_df)} agents; income_tier counts: "
          f"{agents_df['income_tier'].value_counts().to_dict()}")

    cfg = AgentBetaConfig(cov_by_tier={1: args.cov, 2: args.cov, 3: args.cov},
                           seed=args.seed)
    print(f"[het] using CoV per tier: {cfg.cov_by_tier} (Wardman 2014 anchor)")
    print(f"[het] WebTAG VOT (GBP/h, central): {WEBTAG_VOT_BY_TIER_GBP_PER_HOUR}")

    # Frozen baseline norm stats so scenario gets structural ΔV
    X_base = data["static"]
    norm_stats = trainer.gnn.compute_norm_stats(
        X_base.unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )

    # ===== BASELINE (no intervention) =====
    print("\n[het] === BASELINE (no intervention) ===")
    F_agg_base = run_aggregate_prediction(trainer, X_base, norm_stats)
    F_agent_base, eps, scalars = run_agent_prediction(
        trainer, X_base, norm_stats, agents_df, occ_match,
        data["t_ij"].cpu().numpy(), data["log_d"].cpu().numpy(),
        income_score_per_origin, wage_score_per_dest, cfg,
    )
    print(f"[het] recovered scalars: β={scalars['beta_base']:+.4f} "
          f"φ={scalars['phi']:+.4f} ψ={scalars['psi']:+.4f} "
          f"δ={scalars['delta']:+.4f} γ={scalars['gamma']:+.4f}")
    print(f"[het] ε distribution: mean={eps.mean():+.4f} std={eps.std():.4f} "
          f"min={eps.min():+.4f} max={eps.max():+.4f}")

    # Calibrate scales: F_agent uses 10K agents; F_agg has different total mass
    agg_total = F_agg_base.sum(); agent_total = F_agent_base.sum()
    print(f"[het] aggregate baseline total: {agg_total:,.0f}")
    print(f"[het] agent     baseline total: {agent_total:,.0f}")
    scale = agg_total / max(agent_total, 1e-9)
    F_agent_base_scaled = F_agent_base * scale

    # ===== SCENARIO (intervention) =====
    print("\n[het] === SCENARIO (Stratford 9-cluster +Δemp) ===")
    if args.coherent:
        X_int = apply_coherent_employment_intervention(X_base, args.grids, args.delta, mu, sd)
    else:
        X_int = apply_intervention_raw(X_base, args.grids, "total_employment", args.delta, mu, sd)
    F_agg_int = run_aggregate_prediction(trainer, X_int, norm_stats)
    F_agent_int, _, _ = run_agent_prediction(
        trainer, X_int, norm_stats, agents_df, occ_match,
        data["t_ij"].cpu().numpy(), data["log_d"].cpu().numpy(),
        income_score_per_origin, wage_score_per_dest, cfg,
    )
    F_agent_int_scaled = F_agent_int * scale  # use baseline scale for fair Δ

    # ===== JENSEN BIAS COMPARISON =====
    delta_F_agg = F_agg_int - F_agg_base                              # aggregate prediction
    delta_F_agent = F_agent_int_scaled - F_agent_base_scaled          # agent prediction
    diff = delta_F_agg - delta_F_agent                                # Jensen bias

    cluster = args.grids
    print(f"\n[het] === ΔInflow into cluster ({len(cluster)} grids) ===")
    inflow_agg = float(delta_F_agg[:, cluster].sum())
    inflow_agent = float(delta_F_agent[:, cluster].sum())
    bias = inflow_agg - inflow_agent
    print(f"  AGGREGATE prediction:    ΔInflow = {inflow_agg:+.1f}")
    print(f"  AGENT-LEVEL prediction:  ΔInflow = {inflow_agent:+.1f}")
    print(f"  Jensen bias (agg - agent): {bias:+.1f}  "
          f"({100 * bias / max(abs(inflow_agent), 1e-9):+.1f}% of agent prediction)")

    # Per-destination summary
    dest_agg = delta_F_agg.sum(axis=0)
    dest_agent = delta_F_agent.sum(axis=0)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(dest_agg, dest_agent)
    pearson = float(np.corrcoef(dest_agg, dest_agent)[0, 1])
    print(f"\n[het] per-dest Δinflow: aggregate vs agent")
    print(f"  Spearman ρ: {rho:.4f}")
    print(f"  Pearson r:  {pearson:.4f}")
    print(f"  total |ΔF| aggregate: {np.abs(delta_F_agg).sum():.1f}")
    print(f"  total |ΔF| agent:     {np.abs(delta_F_agent).sum():.1f}")

    # Save
    summary = {
        "ckpt": str(args.ckpt),
        "grids_intervened": list(cluster),
        "delta_per_grid": float(args.delta),
        "coherent": bool(args.coherent),
        "cov_by_tier": cfg.cov_by_tier,
        "n_agents": int(len(agents_df)),
        "scalars": scalars,
        "epsilon_stats": {"mean": float(eps.mean()), "std": float(eps.std()),
                            "min": float(eps.min()), "max": float(eps.max())},
        "scale_factor": float(scale),
        "totals": {
            "F_agg_base_total": float(F_agg_base.sum()),
            "F_agent_base_scaled_total": float(F_agent_base_scaled.sum()),
            "F_agg_int_total": float(F_agg_int.sum()),
            "F_agent_int_scaled_total": float(F_agent_int_scaled.sum()),
        },
        "cluster_inflow_aggregate": inflow_agg,
        "cluster_inflow_agent": inflow_agent,
        "jensen_bias_absolute": bias,
        "jensen_bias_pct_of_agent": float(100 * bias / max(abs(inflow_agent), 1e-9)),
        "delta_F_dest_spearman": float(rho),
        "delta_F_dest_pearson": float(pearson),
        "abs_delta_F_aggregate": float(np.abs(delta_F_agg).sum()),
        "abs_delta_F_agent": float(np.abs(delta_F_agent).sum()),
    }
    out = ROOT / "evaluation_outputs" / "paper_a" / "het_agent_vs_aggregate.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[het] wrote {out}")


if __name__ == "__main__":
    main()
