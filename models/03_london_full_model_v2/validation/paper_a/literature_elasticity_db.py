"""Literature elasticity database (Paper A, validation #3).

Provides a structured, citeable set of plausibility brackets for the
elasticities the model implies under Scenario A, replacing the
infeasible real-world counterfactual with comparison to published
ranges. Each entry records the central range, a citation string, and
which scenario output it constrains.

References (full list)
----------------------
- Manning, A. (2003). *Monopsony in Motion: Imperfect Competition in
  Labor Markets*, Princeton UP. Chs. 5-6 — wage elasticity of
  commute distance ~0.10–0.30 for FT workers.
- Mulalic, I., van Ommeren, J.N., Pilegaard, N. (2014). "Wages and
  commuting: Quasi-natural experiments' evidence from firms that
  relocate", *J Public Econ*, 117:44-61. Wage elasticity of
  commute distance ≈ 0.11 (Denmark).
- Geurs, K.T., van Wee, B. (2004). "Accessibility evaluation of land-
  use and transport strategies", *J Transport Geography*, 12(2):
  127-140 — employment-accessibility elasticity ~0.05–0.25.
- Department for Transport (2024). *TAG Unit A5.4 — Marginal external
  costs*. Mode-cost elasticity car -0.3..-0.7; PT cost -0.2..-0.5;
  mode-time elasticity -0.3..-0.7.
- Couture, V., Gaubert, C., Handbury, J., Hurst, E. (2022). "Income
  growth and the distributional effects of urban spatial sorting",
  *AER*, 112(11):3618-3654 — within-metro agglomeration-wage
  elasticity ~0.05.
- Glaeser, E.L., Gottlieb, J.D. (2009). "The wealth of cities:
  agglomeration economies and spatial equilibrium in the US",
  *J Econ Lit*, 47(4):983-1028 — housing-price-to-employment
  elasticity 0.15–0.30.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


ELASTICITY_RANGES: Dict[str, Dict[str, object]] = {
    "wage_elasticity_commute_distance": {
        "range": (0.10, 0.30),
        "citation": "Manning 2003 (Monopsony in Motion); "
                    "Mulalic, van Ommeren & Pilegaard 2014 (J Public Econ)",
        "applies_to": "scenario_a_employment_shock",
        "description": (
            "Percent increase in wages a worker requires per 1 % increase "
            "in commute distance. Bounds the OOC compensating-differential."
        ),
    },
    "employment_accessibility_elasticity": {
        "range": (0.05, 0.25),
        "citation": "Geurs & van Wee 2004 (J Transport Geography)",
        "applies_to": "scenario_a_accessibility",
        "description": (
            "Elasticity of accessibility index w.r.t. employment at the "
            "destination. Used to bracket Δaccessibility from +65k OOC."
        ),
    },
    "mode_cost_elasticity_car": {
        "range": (-0.7, -0.3),
        "citation": "DfT TAG A5.4 (2024)",
        "applies_to": "mode_split_response",
        "description": "Own-cost elasticity of car demand (negative).",
    },
    "mode_cost_elasticity_pt": {
        "range": (-0.5, -0.2),
        "citation": "DfT TAG A5.4 (2024)",
        "applies_to": "mode_split_response",
        "description": "Own-cost elasticity of PT demand (negative).",
    },
    "mode_time_elasticity": {
        "range": (-0.7, -0.3),
        "citation": "DfT TAG A5.4 (2024)",
        "applies_to": "mode_split_response",
        "description": "Generalised-time own-elasticity of demand.",
    },
    "agglomeration_wage_elasticity": {
        "range": (0.03, 0.08),
        "citation": "Couture, Gaubert, Handbury & Hurst 2022 (AER)",
        "applies_to": "scenario_a_wage_uplift",
        "description": (
            "Within-metro elasticity of nominal wages to local employment "
            "density. Caps the wage uplift OOC can plausibly produce."
        ),
    },
    "housing_price_employment_elasticity": {
        "range": (0.15, 0.30),
        "citation": "Glaeser & Gottlieb 2009 (J Econ Lit)",
        "applies_to": "scenario_a_housing_demand",
        "description": (
            "Long-run elasticity of local housing prices to employment. "
            "Used as plausibility check on land-use feedback if modelled."
        ),
    },
}


def check_in_range(elasticity_name: str, observed_value: float,
                   tol: float = 0.0) -> bool:
    """Return True iff `observed_value` lies within the literature
    range for `elasticity_name`, optionally widened by ±`tol`
    (additive). Raises KeyError on unknown name."""
    spec = ELASTICITY_RANGES[elasticity_name]
    lo, hi = spec["range"]   # type: ignore[assignment]
    return (lo - tol) <= observed_value <= (hi + tol)


def make_brackets_table(scenario_outputs_dict: Dict[str, float],
                         out_csv: Optional[Path] = None) -> pd.DataFrame:
    """Build a tidy comparison table.

    `scenario_outputs_dict` maps elasticity_name → observed value from
    the model. Names not in the database are recorded with a NaN check.
    """
    rows = []
    for name, obs in scenario_outputs_dict.items():
        spec = ELASTICITY_RANGES.get(name)
        if spec is None:
            rows.append({
                "elasticity": name,
                "observed": obs,
                "range_lo": None,
                "range_hi": None,
                "in_range": None,
                "citation": "<not in database>",
                "applies_to": "",
            })
            continue
        lo, hi = spec["range"]   # type: ignore[assignment]
        rows.append({
            "elasticity": name,
            "observed": float(obs),
            "range_lo": float(lo),
            "range_hi": float(hi),
            "in_range": bool(lo <= obs <= hi),
            "citation": spec["citation"],
            "applies_to": spec["applies_to"],
        })
    df = pd.DataFrame(rows)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def list_entries() -> pd.DataFrame:
    """Pretty dump of the whole DB — useful for the paper appendix."""
    rows = []
    for name, spec in ELASTICITY_RANGES.items():
        lo, hi = spec["range"]   # type: ignore[assignment]
        rows.append({
            "elasticity": name,
            "lo": lo, "hi": hi,
            "citation": spec["citation"],
            "applies_to": spec["applies_to"],
            "description": spec.get("description", ""),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":   # pragma: no cover
    print(list_entries().to_string(index=False))
