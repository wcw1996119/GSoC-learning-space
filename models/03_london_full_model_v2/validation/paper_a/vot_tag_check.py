"""VOT vs DfT TAG external-validity check (Paper A, validation #1).

Recovers the implied Value of Travel Time (VOT) from the inverse-RUM
estimator and compares it against the UK Department for Transport's
TAG A1.3 unit reference values (May 2024 release). Following the
validator-round refinement, this is one of the four substitutes for a
real-world counterfactual comparison.

References
----------
- Department for Transport (2024). *TAG Unit A1.3 — User and Provider
  Impacts*. Table A1.3.1 (Values of working / non-working time, 2010
  market prices). Commuting all-modes ~£12.65/h.
- Small, K.A. (2012). "Valuation of travel time", *Economics of
  Transportation*, 1(1-2):2-14 — methodological backing for VOT =
  (β_t / β_c) × 60 conversion.
- Hensher, D.A. (2011). "Valuation of travel time savings",
  *Handbook of Transport Economics*, ch.7.

Implementation notes
--------------------
- VOT_recovered = (β_t / β_c) * 60   # £/hour, with β_t in utils per
  minute and β_c in utils per £.
- Bootstrap CI uses parametric resampling on (β_t, β_c) with the
  asymptotic covariance returned by InverseRUMTrainer.fit().
- Pass criterion: |VOT_recovered - VOT_TAG_central| / VOT_TAG_central
  ≤ 0.30  (validator-round threshold).

This module does NOT call the trainer itself; the caller passes
parameter point estimates + (optional) covariance. A thin wrapper at
the bottom shows how the orchestrator wires it up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Defensive import — inverse_rum may be under construction in a
# parallel branch. We do not need to *call* it for the comparison.
try:
    from models_lib.inverse_rum import InverseRUMTrainer  # noqa: F401
    _HAS_INVERSE_RUM = True
except Exception:
    _HAS_INVERSE_RUM = False

# DfT TAG A1.3 (May 2024 release), 2010 market prices, £/hour.
TAG_VOT_COMMUTE_CENTRAL = 12.65
TAG_VOT_LOW = 8.0
TAG_VOT_MID = 13.0
TAG_VOT_HIGH = 22.0

PASS_THRESHOLD_PCT = 0.30   # |relative error| ≤ 30 % ⇒ pass

OUT_DIR = Path(__file__).resolve().parents[2] / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VOTResult:
    vot_recovered: float
    vot_lo: float
    vot_hi: float
    rel_error_vs_central: float
    passes: bool
    in_band: str        # "low", "mid", "high", or "outside"


def vot_from_betas(beta_t: float, beta_c: float) -> float:
    """Convert (β_t, β_c) to VOT in £/hour.

    β_t units: utils per minute; β_c units: utils per £.
    VOT (£/hour) = (β_t / β_c) × 60. The minus signs in both
    coefficients cancel.
    """
    if abs(beta_c) < 1e-9:
        raise ValueError("β_c too close to zero — VOT undefined")
    return (beta_t / beta_c) * 60.0


def bootstrap_vot_ci(
    beta_t: float,
    beta_c: float,
    cov: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    rng_seed: int = 42,
    quantiles: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[float, float]:
    """Parametric bootstrap of VOT given (β_t, β_c) point estimates.

    If `cov` is None we fall back to ±15 % multiplicative noise — a
    last-resort placeholder; the orchestrator should always pass the
    real asymptotic covariance from the trainer.
    """
    rng = np.random.default_rng(rng_seed)
    if cov is None:
        sd_t = 0.15 * abs(beta_t)
        sd_c = 0.15 * abs(beta_c)
        bt = rng.normal(beta_t, sd_t, n_boot)
        bc = rng.normal(beta_c, sd_c, n_boot)
    else:
        draws = rng.multivariate_normal([beta_t, beta_c], cov, size=n_boot)
        bt, bc = draws[:, 0], draws[:, 1]
    bc = np.where(np.abs(bc) < 1e-9, np.nan, bc)
    vot_draws = (bt / bc) * 60.0
    vot_draws = vot_draws[np.isfinite(vot_draws)]
    return tuple(np.quantile(vot_draws, quantiles).tolist())


def _band_label(v: float) -> str:
    if v < TAG_VOT_LOW * 0.8:
        return "outside"
    if v < (TAG_VOT_LOW + TAG_VOT_MID) / 2:
        return "low"
    if v < (TAG_VOT_MID + TAG_VOT_HIGH) / 2:
        return "mid"
    if v <= TAG_VOT_HIGH * 1.25:
        return "high"
    return "outside"


def vot_tag_check(
    beta_t: float,
    beta_c: float,
    cov: Optional[np.ndarray] = None,
    out_csv: Optional[Path] = None,
    out_png: Optional[Path] = None,
) -> VOTResult:
    """Run the full VOT-vs-TAG comparison and persist artefacts."""
    vot = vot_from_betas(beta_t, beta_c)
    lo, hi = bootstrap_vot_ci(beta_t, beta_c, cov=cov)
    rel_err = abs(vot - TAG_VOT_COMMUTE_CENTRAL) / TAG_VOT_COMMUTE_CENTRAL
    res = VOTResult(
        vot_recovered=vot,
        vot_lo=lo,
        vot_hi=hi,
        rel_error_vs_central=rel_err,
        passes=bool(rel_err <= PASS_THRESHOLD_PCT),
        in_band=_band_label(vot),
    )
    df = pd.DataFrame([{
        "beta_t": beta_t,
        "beta_c": beta_c,
        "vot_recovered_gbp_per_h": vot,
        "vot_ci_lo": lo,
        "vot_ci_hi": hi,
        "tag_central_gbp_per_h": TAG_VOT_COMMUTE_CENTRAL,
        "tag_low": TAG_VOT_LOW,
        "tag_mid": TAG_VOT_MID,
        "tag_high": TAG_VOT_HIGH,
        "rel_error": rel_err,
        "passes_30pct": res.passes,
        "in_tag_band": res.in_band,
    }])
    out_csv = Path(out_csv) if out_csv else OUT_DIR / "vot_validation.csv"
    df.to_csv(out_csv, index=False)

    # Plot — only if matplotlib available; non-fatal on failure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        # TAG bands as horizontal coloured spans.
        ax.axhspan(TAG_VOT_LOW * 0.9, TAG_VOT_LOW * 1.1, alpha=0.18,
                   color="#1f77b4", label="TAG low")
        ax.axhspan(TAG_VOT_MID * 0.9, TAG_VOT_MID * 1.1, alpha=0.18,
                   color="#2ca02c", label="TAG mid")
        ax.axhspan(TAG_VOT_HIGH * 0.9, TAG_VOT_HIGH * 1.1, alpha=0.18,
                   color="#d62728", label="TAG high")
        ax.axhline(TAG_VOT_COMMUTE_CENTRAL, ls="--", color="k",
                   label=f"TAG central £{TAG_VOT_COMMUTE_CENTRAL:.2f}")
        ax.errorbar([0], [vot], yerr=[[vot - lo], [hi - vot]],
                    fmt="o", color="black", capsize=4,
                    label=f"Recovered £{vot:.2f}")
        ax.set_xticks([])
        ax.set_ylabel("VOT (£/hour, 2010 prices)")
        ax.set_title("Recovered VOT vs DfT TAG A1.3")
        ax.legend(loc="best", fontsize=8)
        out_png = Path(out_png) if out_png else OUT_DIR / "vot_validation.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return res


if __name__ == "__main__":   # pragma: no cover — manual smoke check
    # Illustrative β values; the orchestrator overrides them with
    # InverseRUMTrainer.fit() outputs.
    demo = vot_tag_check(beta_t=-0.07, beta_c=-0.35)
    print(demo)
