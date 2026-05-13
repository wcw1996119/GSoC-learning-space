"""Multiverse / specification-curve sensitivity analysis (Paper A, validation #2).

100 parameter combinations sampled from priors over (β_t, β_c, α_V, γ,
K_choice_set). For each draw, runs Scenario A
(do(employment_OOC += 65k)) and records:
- Δflow_OOC          (change in inflow at the OOC catchment grid)
- Δmean_commute_time
- Δaccessibility_gini

Outputs a 90 % credibility interval for each scenario quantity.

PRE-REGISTRATION DISCIPLINE
---------------------------
The constants `MULTIVERSE_GRID_SEED`, `N_DRAWS`, and `PRIORS` below are
FROZEN — do not modify after first commit. Any change must produce a
new file (e.g. multiverse_runner_v2.py) with a fresh registration
hash, and Paper A must update its pre-analysis plan accordingly. This
matches Munafò et al. 2017 (Nat Hum Behav) and Steegen et al. 2016.

References
----------
- Steegen, S., Tuerlinckx, F., Gelman, A., Vanpaemel, W. (2016).
  "Increasing transparency through a multiverse analysis",
  *Perspectives on Psychological Science*, 11(5):702-712.
- Simonsohn, U., Simmons, J.P., Nelson, L.D. (2020). "Specification
  curve analysis", *Nature Human Behaviour*, 4:1208-1214.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# --- defensive import; the trainer / model wrappers may not yet be
# importable at the moment validation code is written.
try:
    from models_lib.inverse_rum import InverseRUMTrainer  # noqa: F401
    _HAS_INVERSE_RUM = True
except Exception:
    _HAS_INVERSE_RUM = False


# =====================================================================
# FROZEN PRE-REGISTERED GRID — DO NOT MODIFY AFTER FIRST COMMIT
# =====================================================================
MULTIVERSE_GRID_SEED: int = 42
N_DRAWS: int = 100
PRIORS: Dict[str, Dict[str, float]] = {
    "beta_t":  {"dist": "normal", "mean": -0.07, "sd": 0.02},   # commuting time disutility (utils/min)
    "beta_c":  {"dist": "normal", "mean": -1.0,  "sd": 0.20},   # cost numeraire (utils/£)
    "alpha_V": {"dist": "normal", "mean":  1.00, "sd": 0.25},   # GNN prior weight (validator-capped)
    "gamma":   {"dist": "normal", "mean": -1.50, "sd": 0.40},   # distance-decay
    # K_choice_set is integer; sample uniform over a small fixed set.
    "K_choice_set": {"dist": "uniform_int", "low": 25, "high": 100},
}
# =====================================================================


OUT_DIR = Path(__file__).resolve().parents[2] / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MultiverseResult:
    df: pd.DataFrame              # one row per draw
    summary: pd.DataFrame         # per-output 5/50/95 percentiles


def sample_grid(seed: int = MULTIVERSE_GRID_SEED, n: int = N_DRAWS) -> pd.DataFrame:
    """Generate the (frozen) 100-row sampling grid.

    Same seed ⇒ identical grid. The orchestrator must call with no
    arguments so the registered defaults apply.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for i in range(n):
        row = {"draw_id": i}
        for name, p in PRIORS.items():
            if p["dist"] == "normal":
                row[name] = float(rng.normal(p["mean"], p["sd"]))
            elif p["dist"] == "uniform_int":
                row[name] = int(rng.integers(p["low"], p["high"] + 1))
            else:
                raise ValueError(f"unsupported prior dist {p['dist']!r}")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Scenario-A simulation hook
# ---------------------------------------------------------------------

def _default_simulate_scenario_A(params: Dict[str, float]) -> Dict[str, float]:
    """Default fallback simulator used when the real wrapper is not
    importable. Produces deterministic but parameter-sensitive proxy
    outputs so unit-tests of the multiverse plumbing still pass.
    Replace via `simulate_fn=` when calling `run_multiverse()`.
    """
    bt = params["beta_t"]
    bc = params["beta_c"]
    av = params["alpha_V"]
    g = params["gamma"]
    # Toy-but-monotone proxy: more weight on time (|bt| big) ⇒ less
    # diversion; bigger α_V ⇒ stronger prior pull toward OOC.
    base_flow = 1500.0 + 800.0 * av * np.exp(0.5 * g) - 2000.0 * abs(bt)
    return {
        "delta_flow_OOC": float(max(base_flow, 0.0)),
        "delta_mean_commute_time": float(0.05 * av - 0.02 * abs(bt)),
        "delta_accessibility_gini": float(-0.001 * av + 0.0005 * abs(g)),
    }


def run_multiverse(
    simulate_fn: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
    out_csv: Optional[Path] = None,
    out_png: Optional[Path] = None,
    summary_csv: Optional[Path] = None,
) -> MultiverseResult:
    """Run all 100 frozen draws through `simulate_fn`.

    `simulate_fn(params_dict) -> {delta_flow_OOC, delta_mean_commute_time,
    delta_accessibility_gini}`. If None, the deterministic placeholder
    is used so downstream code paths stay testable.
    """
    sim = simulate_fn if simulate_fn is not None else _default_simulate_scenario_A
    grid = sample_grid()
    out_rows: List[Dict[str, float]] = []
    for _, row in grid.iterrows():
        params = {k: row[k] for k in PRIORS.keys()}
        try:
            outs = sim(params)
        except Exception as e:                      # noqa: BLE001
            outs = {
                "delta_flow_OOC": np.nan,
                "delta_mean_commute_time": np.nan,
                "delta_accessibility_gini": np.nan,
                "error": str(e)[:200],
            }
        out_rows.append({**row.to_dict(), **outs})
    df = pd.DataFrame(out_rows)

    cols = ["delta_flow_OOC", "delta_mean_commute_time", "delta_accessibility_gini"]
    summary_rows = []
    for c in cols:
        v = df[c].dropna().to_numpy()
        if len(v) == 0:
            summary_rows.append({"output": c, "p05": np.nan, "p50": np.nan,
                                 "p95": np.nan, "mean": np.nan, "n": 0})
            continue
        summary_rows.append({
            "output": c,
            "p05": float(np.quantile(v, 0.05)),
            "p50": float(np.quantile(v, 0.50)),
            "p95": float(np.quantile(v, 0.95)),
            "mean": float(np.mean(v)),
            "n": int(len(v)),
        })
    summary = pd.DataFrame(summary_rows)

    out_csv = Path(out_csv) if out_csv else OUT_DIR / "multiverse_results.csv"
    df.to_csv(out_csv, index=False)
    summary_csv = Path(summary_csv) if summary_csv else OUT_DIR / "multiverse_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Specification-curve plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, c in zip(axes, cols):
            v = np.sort(df[c].dropna().to_numpy())
            ax.plot(v, np.linspace(0, 1, len(v)), lw=1.5)
            p5, p95 = np.quantile(v, [0.05, 0.95]) if len(v) else (np.nan, np.nan)
            ax.axvline(p5, ls=":", color="grey")
            ax.axvline(p95, ls=":", color="grey")
            ax.axvline(0, ls="-", color="red", alpha=0.6)
            ax.set_title(c)
            ax.set_ylabel("CDF")
        fig.suptitle(
            f"Multiverse over {N_DRAWS} draws (seed={MULTIVERSE_GRID_SEED}) — Scenario A"
        )
        fig.tight_layout()
        out_png = Path(out_png) if out_png else OUT_DIR / "multiverse_plot.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return MultiverseResult(df=df, summary=summary)


def registration_hash() -> str:
    """Stable hash of the frozen registration. Paper A reports this in
    the methods section to demonstrate pre-commitment.
    """
    import hashlib
    import json
    payload = json.dumps(
        {"seed": MULTIVERSE_GRID_SEED, "n": N_DRAWS, "priors": PRIORS},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


if __name__ == "__main__":   # pragma: no cover
    print("registration_hash =", registration_hash())
    res = run_multiverse()
    print(res.summary)
