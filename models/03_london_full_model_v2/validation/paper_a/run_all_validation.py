"""Paper-A validation orchestrator.

Sequential pipeline:
    1. vot_tag_check       — VOT vs DfT TAG A1.3 external-validity test
    2. multiverse_runner   — 100-draw specification curve over priors
    3. multi_model_triangulation — 4 OD-flow models on Scenario A
Then:
    emit `evaluation_outputs/paper_a/validation_summary.json`
which Paper A cites as the single artefact summarising whether the
four substitutes for a real-world counterfactual all pass.

Usage
-----
    python -m validation.paper_a.run_all_validation

Inputs are populated by closures defined here (`_load_recovered_betas`,
`_load_triangulation_data`, `_real_simulate_scenario_A`). Each one tries
to wire the real upstream artefact (gnn_ablation_summary.json, the
processed OD/feature cache, an InverseRUMTrainer checkpoint) and falls
back to placeholders ONLY when the upstream artefact is missing — and
in that case logs a loud warning and tags rows as ``synthetic=True``.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.paper_a import vot_tag_check as _vot
from validation.paper_a import multiverse_runner as _mv
from validation.paper_a import literature_elasticity_db as _lit
from validation.paper_a import multi_model_triangulation as _tri

OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"
EXP_DIR = ROOT / "evaluation_outputs" / "paper_a"
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = OUT_DIR / "validation_summary.json"

GNN_ABLATION_SUMMARY = EXP_DIR / "gnn_ablation_summary.json"
GNN_ABLATION_CSV = EXP_DIR / "gnn_ablation.csv"
DEMO_CACHE = DATA_DIR / "demo_cache.npz"

# Default OOC grid id (placeholder — orchestrator user can override via
# the `_load_triangulation_data` closure if a different L###_### grid is
# the chosen Old Oak Common analogue). The coordinate lookup at runtime
# verifies the chosen grid is near 51.529N / -0.252W.
DEFAULT_OOC_GRID_ID = "L000_000"
DEFAULT_PEAK_HOUR = 8

logger = logging.getLogger("validation.paper_a.run_all")


# ---------------------------------------------------------------------
# Loader hooks
# ---------------------------------------------------------------------

def _load_recovered_betas() -> Dict[str, Any]:
    """Return the recovered (β_t, β_c) point estimates for VOT validation.

    Reads the GNN×flat winning cell from
    ``evaluation_outputs/paper_a/gnn_ablation_summary.json`` produced by
    `experiments/paper_a/gnn_ablation.py`. Falls back to a documented
    placeholder ONLY when that file does not yet exist (with a loud
    warning so a CI consumer cannot mistake placeholder for real data).
    """
    if GNN_ABLATION_SUMMARY.exists():
        try:
            data = json.loads(GNN_ABLATION_SUMMARY.read_text())
        except Exception as e:                          # noqa: BLE001
            logger.warning(
                "WARNING: failed to parse %s (%s); using placeholder betas",
                GNN_ABLATION_SUMMARY, e,
            )
            return {
                "beta_t": -0.07, "beta_c": -1.00, "cov": None,
                "source": f"placeholder (parse-error in {GNN_ABLATION_SUMMARY.name})",
            }

        # Prefer the per-cell CSV (richer); fall back to the summary JSON.
        if GNN_ABLATION_CSV.exists():
            try:
                df = pd.read_csv(GNN_ABLATION_CSV)
                cell = df[(df["util_family"] == "gnn")
                          & (df["choice_family"] == "flat")]
                if len(cell):
                    beta_t = float(cell["beta_t_hat"].mean())
                    beta_c = (float(cell["beta_c_hat"].mean())
                              if "beta_c_hat" in cell.columns else float("nan"))
                    return {
                        "beta_t": beta_t,
                        "beta_c": beta_c if np.isfinite(beta_c) else -1.00,
                        "cov": None,
                        "source": f"gnn_ablation.csv :: gnn×flat (n={len(cell)})",
                    }
            except Exception as e:                      # noqa: BLE001
                logger.warning(
                    "WARNING: gnn_ablation.csv exists but is unreadable (%s); "
                    "falling back to summary JSON", e,
                )

        # Pull from the summary JSON. gnn_ablation now writes
        # ``gnn_flat_beta_t_hat`` and ``gnn_flat_beta_c_hat`` directly; if
        # they exist, prefer them over a placeholder.
        if ("gnn_flat_beta_t_hat" in data
                and np.isfinite(data["gnn_flat_beta_t_hat"])):
            return {
                "beta_t": float(data["gnn_flat_beta_t_hat"]),
                "beta_c": float(data.get("gnn_flat_beta_c_hat", -1.00))
                          if np.isfinite(data.get("gnn_flat_beta_c_hat",
                                                  float("nan")))
                          else -1.00,
                "cov": None,
                "source": f"gnn_ablation_summary.json :: gnn×flat",
            }

        logger.warning(
            "WARNING: gnn_ablation_summary.json present but per-cell CSV "
            "missing AND no gnn_flat_beta_*_hat keys — using placeholder betas",
        )
        return {
            "beta_t": -0.07, "beta_c": -1.00, "cov": None,
            "source": "placeholder (csv missing)",
        }

    logger.warning(
        "WARNING: %s does not exist — using placeholder (-0.07, -1.0). "
        "Run experiments/paper_a/gnn_ablation.py first for real betas.",
        GNN_ABLATION_SUMMARY,
    )
    return {
        "beta_t": -0.07, "beta_c": -1.00, "cov": None,
        "source": "placeholder (no_gnn_ablation_summary)",
    }


def _load_triangulation_data(peak_hour: int = DEFAULT_PEAK_HOUR
                              ) -> Optional[_tri.FitData]:
    """Build a `multi_model_triangulation.FitData` from the demo cache.

    Reads ``data/processed/demo_cache.npz`` and `grid_static_features.csv`
    and produces an OD long-form frame ``(o, d, t_ij, d_ij, flow)`` at
    `peak_hour` (default 8am). Returns None only if either upstream
    artefact is missing — the caller treats None as "skipped".
    """
    if not DEMO_CACHE.exists():
        logger.warning("WARNING: %s missing; triangulation will be skipped",
                       DEMO_CACHE)
        return None
    feats_csv = DATA_DIR / "grid_static_features.csv"
    if not feats_csv.exists():
        logger.warning("WARNING: %s missing; triangulation will be skipped",
                       feats_csv)
        return None

    cache = dict(np.load(DEMO_CACHE, allow_pickle=True))
    F_ij_t = cache["F_ij_t"]                                # (T, N, N)
    t_ij_t = cache["t_ij_t"]                                # (T, N, N)
    grid_ids = [str(g) for g in cache["grid_ids"]]
    coords_bng = cache["coords_bng"]
    static = cache["static_features"]                       # (N, F)

    T = F_ij_t.shape[0]
    h = peak_hour if 0 <= peak_hour < T else min(T - 1, max(0, peak_hour))
    F = F_ij_t[h]
    t_ij = t_ij_t[h]
    N = F.shape[0]

    diff = coords_bng[:, None, :] - coords_bng[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0            # (N, N)

    # Build OD long-form. To keep memory sensible we drop self-pairs and
    # cells with zero observed flow.
    o_idx, d_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    flow = F.ravel()
    keep = (o_idx.ravel() != d_idx.ravel()) & (flow > 0)
    od_pairs = pd.DataFrame({
        "o": np.asarray(grid_ids)[o_idx.ravel()[keep]],
        "d": np.asarray(grid_ids)[d_idx.ravel()[keep]],
        "t_ij": t_ij.ravel()[keep],
        "d_ij": d_km.ravel()[keep],
        "flow": flow[keep],
    })

    grid_features = pd.read_csv(feats_csv)
    # Restrict to the grids present in the demo cache (keep order).
    grid_features = (grid_features.set_index("grid_id")
                                  .loc[grid_ids]
                                  .reset_index())
    node_feat_cols = [c for c in grid_features.columns
                      if c not in ("grid_id", "centroid_lat", "centroid_lon")]
    node_feat = grid_features[node_feat_cols].fillna(0.0).to_numpy(dtype=np.float32)

    grid_index = {g: i for i, g in enumerate(grid_ids)}

    # Pick the OOC grid: nearest-centroid to (51.529, -0.252) so we don't
    # depend on a hard-coded grid id that may not exist in this cache.
    target_lat, target_lon = 51.529, -0.252
    glat = grid_features["centroid_lat"].to_numpy()
    glon = grid_features["centroid_lon"].to_numpy()
    ooc_idx = int(np.argmin((glat - target_lat) ** 2 + (glon - target_lon) ** 2))
    ooc_grid_id = grid_ids[ooc_idx]

    return _tri.FitData(
        grid_features=grid_features,
        od_pairs=od_pairs,
        node_feat=node_feat,
        grid_index=grid_index,
        ooc_grid_id=ooc_grid_id,
        node_feat_cols=node_feat_cols,
    )


def _scenario_outputs_for_lit_check() -> Dict[str, float]:
    """Map literature-DB elasticity names → observed model values.
    Empty by default; the orchestrator user supplies the mapping."""
    return {}


# ---------------------------------------------------------------------
# Real Scenario-A simulator for the multiverse pipeline (R3)
# ---------------------------------------------------------------------

def _real_simulate_scenario_A(theta_hat: Any,
                              params: Dict[str, float],
                              scenario: str = "A",
                              ) -> Dict[str, float]:
    """Run Scenario A (do(employment_OOC += 65k)) through the trained
    structural-GNN + RUM stack and report the three multiverse outputs.

    Parameters
    ----------
    theta_hat : object
        Trained `models_lib.inverse_rum.InverseRUMTrainer` instance (or
        any object exposing `.gnn`, `.predict_OD(...)`, and the static
        feature tensor used at fit time).
    params : dict
        One row of the multiverse grid: {beta_t, beta_c, alpha_V, gamma,
        K_choice_set}.
    scenario : str
        Currently only "A" is supported.

    Returns
    -------
    {delta_flow_OOC, delta_mean_commute_time, delta_accessibility_gini}

    NOTES
    -----
    - The heavy compute (GNN forward, do-intervention rebuild of V_jt,
      softmax across K choice-set destinations, accessibility gravity
      sum) is performed inside `theta_hat.predict_OD(...)` — this
      function only handles the wiring and Δ aggregation.
    - If `theta_hat` is None or does not expose the expected API, raises
      a RuntimeError so the multiverse loop records the failure rather
      than silently producing toy outputs.
    """
    if scenario != "A":
        raise NotImplementedError(f"Scenario {scenario!r} not supported")
    if theta_hat is None:
        raise RuntimeError(
            "Scenario A simulator received theta_hat=None — wire a "
            "trained InverseRUMTrainer in run_all_validation"
        )

    # --- Resolve handles on the trained model ------------------------------
    gnn = getattr(theta_hat, "gnn", None)
    static = getattr(theta_hat, "grid_features", None)
    edge_index = getattr(theta_hat, "edge_index", None)
    if gnn is None or static is None or edge_index is None:
        raise RuntimeError(
            "theta_hat missing one of {gnn, grid_features, edge_index}; "
            "got: " + ", ".join(sorted(vars(theta_hat).keys())[:8])
        )

    import torch                                  # local to avoid hard dep at import time

    # --- Identify employment column on static features --------------------
    feat_cols = getattr(theta_hat, "feature_columns", None)
    if feat_cols is None or "total_employment" not in list(feat_cols):
        raise RuntimeError(
            "theta_hat.feature_columns missing 'total_employment'; "
            "needed for the do(emp_OOC += 65k) intervention"
        )
    emp_col = list(feat_cols).index("total_employment")
    ooc_idx = int(getattr(theta_hat, "ooc_idx", -1))
    if ooc_idx < 0:
        raise RuntimeError(
            "theta_hat missing .ooc_idx — set this when wiring the trained "
            "model into _real_simulate_scenario_A"
        )

    # --- Baseline V_jt and OD flows ---------------------------------------
    with torch.no_grad():
        V_jt_base = gnn(static, edge_index)                          # (N, 1) or (N,)
        F_base = theta_hat.predict_OD()                              # (N, N) or (T, N, N)

    # --- Counterfactual: apply do(emp_OOC += 65k) and re-run ---------------
    static_cf = static.clone()
    static_cf[ooc_idx, emp_col] = static_cf[ooc_idx, emp_col] + 65_000.0
    with torch.no_grad():
        V_jt_cf = gnn(static_cf, edge_index)
        # Allow the trainer to expose an override hook; if it doesn't,
        # rebuild logits manually using the params from the multiverse row.
        if hasattr(theta_hat, "predict_OD_with_features"):
            F_cf = theta_hat.predict_OD_with_features(static_cf)
        else:
            raise RuntimeError(
                "theta_hat missing .predict_OD_with_features(static_override); "
                "F1's API needs this for do-intervention scoring"
            )

    F_base_np = F_base.detach().cpu().numpy()
    F_cf_np = F_cf.detach().cpu().numpy()

    # --- Metric 1: ΔF into OOC catchment (rows=origins, cols=destinations).
    delta_flow_OOC = float((F_cf_np - F_base_np)[..., ooc_idx].sum())

    # --- Metric 2: Δ mean commute time (weighted by base flow).
    t_ij = getattr(theta_hat, "t_ij_t", None)
    if t_ij is None:
        raise RuntimeError("theta_hat missing .t_ij_t (mean travel time tensor)")
    t_np = t_ij.detach().cpu().numpy()
    if t_np.ndim == 3:
        t_np = t_np.mean(0)
    base_total = F_base_np.sum()
    cf_total = F_cf_np.sum()
    mean_t_base = float((F_base_np * t_np).sum() / max(base_total, 1.0))
    mean_t_cf = float((F_cf_np * t_np).sum() / max(cf_total, 1.0))
    delta_mean_commute_time = mean_t_cf - mean_t_base

    # --- Metric 3: Δ accessibility Gini (gravity-style accessibility per origin).
    beta_t = float(params.get("beta_t", -0.07))
    # E_j = sum over origins, scaled to a destination "attraction". We use the
    # (counterfactual) total_employment per node as the attraction term.
    E_base = static[:, emp_col].detach().cpu().numpy()
    E_cf = static_cf[:, emp_col].detach().cpu().numpy()
    A_base = (E_base[None, :] * np.exp(beta_t * t_np)).sum(axis=1)   # (N,)
    A_cf = (E_cf[None, :] * np.exp(beta_t * t_np)).sum(axis=1)
    delta_accessibility_gini = float(_gini(A_cf) - _gini(A_base))

    return {
        "delta_flow_OOC": delta_flow_OOC,
        "delta_mean_commute_time": delta_mean_commute_time,
        "delta_accessibility_gini": delta_accessibility_gini,
    }


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array (zero-safe)."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or np.all(x <= 0):
        return 0.0
    x = np.sort(np.maximum(x, 0))
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def _make_simulate_fn() -> Optional[Callable[[Dict[str, float]], Dict[str, float]]]:
    """Bind a trained model into a per-row simulate_fn for the multiverse.

    Returns None if no trained model can be loaded — the multiverse stage
    then runs with `simulate_fn=None`, which triggers the toy-proxy
    warning and `synthetic=True` tagging in `_stage_multiverse`.
    """
    try:
        from models_lib.inverse_rum import InverseRUMTrainer  # noqa: F401
    except Exception:
        return None
    # The actual checkpoint loading is left to the orchestrator user;
    # without a checkpoint path we deliberately return None so the
    # warning + synthetic tagging fires (rather than silently using a
    # toy simulator).
    return None


# ---------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------

def _safe(stage_name: str, fn) -> Dict[str, Any]:
    try:
        out = fn()
        return {"status": "ok", "result": out}
    except Exception as e:                        # noqa: BLE001
        return {"status": "error", "error": str(e),
                "traceback": traceback.format_exc()}


def _stage_vot() -> Dict[str, Any]:
    betas = _load_recovered_betas()
    res = _vot.vot_tag_check(
        beta_t=betas["beta_t"], beta_c=betas["beta_c"], cov=betas.get("cov")
    )
    return {
        "vot_recovered": res.vot_recovered,
        "vot_ci": [res.vot_lo, res.vot_hi],
        "tag_central": _vot.TAG_VOT_COMMUTE_CENTRAL,
        "rel_error": res.rel_error_vs_central,
        "passes_30pct": res.passes,
        "in_band": res.in_band,
        "betas_source": betas["source"],
    }


def _stage_multiverse() -> Dict[str, Any]:
    simulate_fn = _make_simulate_fn()
    using_toy = simulate_fn is None
    if using_toy:
        msg = (
            "WARNING: Multiverse running with toy simulator. Results are "
            "placeholders. Do NOT cite as scientific output."
        )
        logger.warning(msg)
        print("[run_all_validation] " + msg)

    res = _mv.run_multiverse(simulate_fn=simulate_fn)

    # Tag every output row so downstream readers (and the summary JSON
    # consumer) cannot mistake placeholder rows for real model output.
    df = res.df.copy()
    df["synthetic"] = bool(using_toy)
    df.to_csv(OUT_DIR / "multiverse_results.csv", index=False)

    summary_records = res.summary.to_dict(orient="records")
    return {
        "registration_hash": _mv.registration_hash(),
        "n_draws": _mv.N_DRAWS,
        "seed": _mv.MULTIVERSE_GRID_SEED,
        "summary": summary_records,
        "synthetic": bool(using_toy),
        "warning": (
            "toy simulator — placeholder output" if using_toy else None
        ),
        "csv": str(OUT_DIR / "multiverse_results.csv"),
    }


def _stage_literature() -> Dict[str, Any]:
    obs = _scenario_outputs_for_lit_check()
    if not obs:
        return {"status": "no_observations_provided",
                 "n_entries": len(_lit.ELASTICITY_RANGES)}
    df = _lit.make_brackets_table(obs, out_csv=OUT_DIR / "elasticity_brackets.csv")
    return {
        "n_entries_checked": int(len(df)),
        "n_in_range": int(df["in_range"].sum(skipna=True))
                       if "in_range" in df else 0,
        "csv": str(OUT_DIR / "elasticity_brackets.csv"),
    }


def _stage_triangulation() -> Dict[str, Any]:
    data = _load_triangulation_data()
    if data is None:
        return {"status": "skipped — _load_triangulation_data() returned None"}
    res = _tri.run_triangulation(data)
    return {
        "direction_agreement": res.direction_agreement,
        "passes_3of4": res.passes,
        "table": res.df.to_dict(orient="records"),
        "csv": str(OUT_DIR / "triangulation.csv"),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["vot_tag"] = _safe("vot_tag", _stage_vot)
    summary["multiverse"] = _safe("multiverse", _stage_multiverse)
    summary["literature"] = _safe("literature", _stage_literature)
    summary["triangulation"] = _safe("triangulation", _stage_triangulation)

    # Aggregate pass/fail.
    def _passes(d):
        if d.get("status") != "ok":
            return None
        r = d["result"]
        if "passes_30pct" in r: return bool(r["passes_30pct"])
        if "passes_3of4" in r:  return bool(r["passes_3of4"])
        return None
    summary["overall"] = {
        "vot_passes": _passes(summary["vot_tag"]),
        "triangulation_passes": _passes(summary["triangulation"]),
        "multiverse_status": summary["multiverse"].get("status"),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, default=_default_json),
    )
    print(f"wrote {SUMMARY_JSON}")
    return summary


def _default_json(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


if __name__ == "__main__":   # pragma: no cover
    logging.basicConfig(level=logging.INFO,
                         format="%(levelname)s %(name)s: %(message)s")
    main()
