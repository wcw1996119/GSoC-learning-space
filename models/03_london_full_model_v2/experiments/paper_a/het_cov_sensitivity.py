"""HET sensitivity: how does Jensen bias scale with σ (CoV) assumption?

Tests CoV ∈ {0.0, 0.2, 0.4, 0.6, 0.8} on Stratford intervention.
σ=0 = no heterogeneity (should give bias ≈ 0).
σ growing → larger Jensen bias (in either direction).

Output: evaluation_outputs/paper_a/het_cov_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import AgentBetaConfig
from experiments.paper_a.train_phase_b_v6_ckpt import load_data
from experiments.paper_a.scenario_A_spatial import (
    apply_coherent_employment_intervention, build_trainer_from_ckpt,
)
from experiments.paper_a.agent_simulation_heterogeneous import (
    run_aggregate_prediction, run_agent_prediction,
    STRATFORD_TEST_GRIDS, DEFAULT_DELTA,
)


def main():
    print("[het-sens] loading ...")
    data = load_data()
    ckpt_path = ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"
    trainer, ckpt = build_trainer_from_ckpt(ckpt_path, data)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    aux = np.load(ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    mu = np.asarray(cache["static_mean"]); sd = np.asarray(cache["static_std"])
    income_score = np.asarray(aux["income_score_per_origin"])
    wage_score = np.asarray(aux["wage_score_per_dest"])
    occ_match = np.asarray(aux["occ_match"])
    agents_df = pd.read_csv(ROOT / "data" / "processed" / "agent_population_v23.csv")

    X_base = data["static"]
    X_int = apply_coherent_employment_intervention(
        X_base, STRATFORD_TEST_GRIDS, DEFAULT_DELTA, mu, sd
    )
    norm_stats = trainer.gnn.compute_norm_stats(
        X_base.unsqueeze(0).expand(trainer.T, -1, -1).contiguous(),
        trainer.edge_index,
    )
    F_agg_base = run_aggregate_prediction(trainer, X_base, norm_stats)
    F_agg_int = run_aggregate_prediction(trainer, X_int, norm_stats)
    delta_agg = (F_agg_int - F_agg_base)[:, STRATFORD_TEST_GRIDS].sum()

    rows = []
    for cov in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        cfg = AgentBetaConfig(cov_by_tier={1: cov, 2: cov, 3: cov}, seed=42)
        F_agent_base, _, _ = run_agent_prediction(
            trainer, X_base, norm_stats, agents_df, occ_match,
            data["t_ij"].cpu().numpy(), data["log_d"].cpu().numpy(),
            income_score, wage_score, cfg,
        )
        F_agent_int, _, _ = run_agent_prediction(
            trainer, X_int, norm_stats, agents_df, occ_match,
            data["t_ij"].cpu().numpy(), data["log_d"].cpu().numpy(),
            income_score, wage_score, cfg,
        )
        scale = F_agg_base.sum() / max(F_agent_base.sum(), 1e-9)
        delta_agent = ((F_agent_int - F_agent_base) * scale)[:, STRATFORD_TEST_GRIDS].sum()
        bias = float(delta_agg - delta_agent)
        bias_pct = float(100 * bias / max(abs(delta_agent), 1e-9))
        print(f"  CoV={cov:.2f}: aggregate={delta_agg:+.1f}  agent={delta_agent:+.1f}  "
              f"bias={bias:+.1f}  ({bias_pct:+.1f}% of agent)")
        rows.append({"cov": cov, "delta_aggregate": float(delta_agg),
                     "delta_agent": float(delta_agent),
                     "jensen_bias_abs": bias, "jensen_bias_pct": bias_pct})

    out = ROOT / "evaluation_outputs" / "paper_a" / "het_cov_sensitivity.json"
    with open(out, "w") as f:
        json.dump({"intervention": "Stratford 9-cluster +7222 each",
                    "rows": rows}, f, indent=2)
    print(f"\n[het-sens] wrote {out}")


if __name__ == "__main__":
    main()
