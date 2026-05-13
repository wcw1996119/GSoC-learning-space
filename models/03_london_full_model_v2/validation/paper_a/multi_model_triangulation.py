"""Multi-model triangulation (Paper A, validation #4).

Runs four independent commuting-flow models on the same Scenario A
intervention (do(employment_OOC += 65k)) and reports whether they
agree in *direction* and *order of magnitude* on Δflow_OOC. This is
the primary substitute for a real-world counterfactual that the
validator round mandated for Paper A.

Models
------
1. Wilson 1971 doubly-constrained gravity (Poisson GLM + IPF) — uses
   `models_lib.baselines.gravity.GravityModel`.
2. Simini 2012 parameter-free radiation model — uses
   `models_lib.baselines.radiation.RadiationModel`.
3. Deep Gravity (Simini et al. 2021, Nature Comms) — uses
   `models_lib.baselines.deep_gravity.DeepGravity`.
   Simplification vs paper: we keep the published 256→128 ReLU MLP
   and softmax-over-destinations training objective unchanged, but
   use only the node features that already exist on the London grid
   (no satellite imagery, no Voronoi tessellation, no neighbouring-
   area hand-crafted features). This matches the "reduced inputs"
   variant Simini et al. report as their baseline ablation.
4. Our model: TSI-GCN encoder + InverseRUMTrainer (`models_lib.
   inverse_rum.InverseRUMTrainer`). For Scenario A we re-run the
   STGNN forward pass with employment_OOC += 65k, then renormalise
   the resulting destination-choice probabilities.

Outputs
-------
`evaluation_outputs/paper_a/triangulation.csv` with columns:
    model | direction (sign) | magnitude (mean ΔF in OOC catchment)
    | overlaps_with_ours (Y/N within 50 % of our magnitude)

Pass criterion: ≥ 3 of 4 models agree on direction. If not, Paper A
must flag Scenario A as exploratory in the abstract.

References
----------
- Wilson, A.G. (1971). "A family of spatial interaction models",
  *Environment & Planning A*, 3(1):1-32.
- Simini, F. et al. (2012). "A universal model for mobility and
  migration patterns", *Nature*, 484:96-100.
- Simini, F., Barlacchi, G., Luca, M., Pappalardo, L. (2021). "A
  Deep Gravity model for mobility flows generation", *Nat Comms*,
  12:6576.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# Defensive imports — the four model classes may live in different
# states of completeness while validation code is being written.
try:
    from models_lib.baselines.gravity import GravityModel
except Exception:                                    # pragma: no cover
    GravityModel = None                              # type: ignore
try:
    from models_lib.baselines.radiation import RadiationModel
except Exception:                                    # pragma: no cover
    RadiationModel = None                            # type: ignore
try:
    from models_lib.baselines.deep_gravity import DeepGravity
except Exception:                                    # pragma: no cover
    DeepGravity = None                               # type: ignore
try:
    from models_lib.inverse_rum import InverseRUMTrainer
    _HAS_INVERSE_RUM = True
except Exception:
    InverseRUMTrainer = None                         # type: ignore
    _HAS_INVERSE_RUM = False


OUT_DIR = Path(__file__).resolve().parents[2] / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOC_LAT, OOC_LON = 51.529, -0.252
OOC_JOBS_DELTA = 65_000

# Catchment radius (km) inside which we sum Δflow.
OOC_CATCHMENT_KM = 5.0


@dataclass
class FitData:
    """Bundle of arrays needed to fit/predict any of the 4 models."""
    grid_features: pd.DataFrame      # cols: grid_id, centroid_lat/lon, population, total_employment, …
    od_pairs: pd.DataFrame           # cols: o, d, t_ij, d_ij, flow
    node_feat: np.ndarray            # (N, F) numeric features for DG / ours
    grid_index: Dict[str, int]       # grid_id → row idx
    ooc_grid_id: str
    # Column names of node_feat (required by Deep Gravity for the Scenario
    # A counterfactual to know which column carries 'total_employment').
    node_feat_cols: Optional[list] = None


@dataclass
class TriangulationResult:
    df: pd.DataFrame
    direction_agreement: int     # number of models agreeing with ours on sign
    passes: bool                 # True if ≥ 3/4 agree on direction


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _ooc_catchment_mask(grid_features: pd.DataFrame, ooc_id: str) -> np.ndarray:
    row = grid_features.loc[grid_features["grid_id"] == ooc_id].iloc[0]
    d = _haversine_km(row["centroid_lat"], row["centroid_lon"],
                      grid_features["centroid_lat"].to_numpy(),
                      grid_features["centroid_lon"].to_numpy())
    return d <= OOC_CATCHMENT_KM


def _delta_flow_ooc(flow_base: np.ndarray, flow_cf: np.ndarray,
                    od: pd.DataFrame, catchment_ids: set) -> float:
    """Sum (cf - base) over OD pairs with destination in catchment."""
    mask = od["d"].isin(catchment_ids).to_numpy()
    return float(np.sum(flow_cf[mask] - flow_base[mask]))


# ---------------------------------------------------------------------
# Per-model wrappers
# ---------------------------------------------------------------------

def _run_gravity(data: FitData) -> Optional[Dict[str, float]]:
    if GravityModel is None:
        return None
    od = data.od_pairs
    O = data.grid_features.set_index("grid_id")["population"].astype(float)
    D = data.grid_features.set_index("grid_id")["total_employment"].astype(float)
    o_arr = od["o"].to_numpy()
    d_arr = od["d"].to_numpy()
    t = od["t_ij"].to_numpy()
    O_i = O.loc[o_arr].to_numpy()
    D_j = D.loc[d_arr].to_numpy()
    m = GravityModel().fit(O_i, D_j, t, od["flow"].to_numpy())

    base_pred = m.predict(O_i, D_j, t, origins=o_arr, dests=d_arr)
    D_cf = D.copy()
    D_cf.loc[data.ooc_grid_id] += OOC_JOBS_DELTA
    D_j_cf = D_cf.loc[d_arr].to_numpy()
    cf_pred = m.predict(O_i, D_j_cf, t, origins=o_arr, dests=d_arr)
    return {"base": base_pred, "cf": cf_pred}


def _run_radiation(data: FitData) -> Optional[Dict[str, np.ndarray]]:
    if RadiationModel is None:
        return None
    od = data.od_pairs
    m = RadiationModel().fit(data.grid_features)
    o_arr = od["o"].to_numpy()
    d_arr = od["d"].to_numpy()
    T_i = od.groupby("o")["flow"].transform("sum").to_numpy()
    base_pred = m.predict(o_arr, d_arr, T_i)

    gf_cf = data.grid_features.copy()
    gf_cf.loc[gf_cf["grid_id"] == data.ooc_grid_id, "total_employment"] += OOC_JOBS_DELTA
    m_cf = RadiationModel().fit(gf_cf)
    cf_pred = m_cf.predict(o_arr, d_arr, T_i)
    return {"base": base_pred, "cf": cf_pred}


def _run_deep_gravity(data: FitData,
                       epochs: int = 30) -> Optional[Dict[str, np.ndarray]]:
    if DeepGravity is None:
        return None
    od = data.od_pairs
    o_idx = od["o"].map(data.grid_index).to_numpy()
    d_idx = od["d"].map(data.grid_index).to_numpy()
    t = od["t_ij"].to_numpy()
    dist = od["d_ij"].to_numpy()
    flow = od["flow"].to_numpy()

    m = DeepGravity(epochs=epochs).fit(o_idx, d_idx, data.node_feat, t, dist, flow)
    T_i = od.groupby("o")["flow"].transform("sum").to_numpy()
    base_pred = m.predict(o_idx, d_idx, data.node_feat, t, dist, T_i)

    # Counterfactual: bump employment at the OOC node. We require an
    # explicit ``total_employment`` column in node_feat — silently doing a
    # 5%-of-everything multiplicative fudge would invalidate the
    # interpretation of the Scenario A intervention (R3).
    feat_cf = data.node_feat.copy()
    ooc_idx = data.grid_index[data.ooc_grid_id]
    node_feat_cols = getattr(data, "node_feat_cols", None)
    if node_feat_cols is None or "total_employment" not in list(node_feat_cols):
        raise KeyError(
            "Deep Gravity counterfactual requires 'total_employment' in "
            "node_feat columns; got: " + str(node_feat_cols)
        )
    emp_col = list(node_feat_cols).index("total_employment")
    feat_cf[ooc_idx, emp_col] += OOC_JOBS_DELTA
    cf_pred = m.predict(o_idx, d_idx, feat_cf, t, dist, T_i)
    return {"base": base_pred, "cf": cf_pred}


def _run_ours(data: FitData,
              ours_predict_fn: Optional[Callable[..., Dict[str, np.ndarray]]] = None,
              ) -> Optional[Dict[str, np.ndarray]]:
    """Run our TSI-GCN + inverse-RUM model on baseline and Scenario A.

    The wiring lives in `models_lib.inverse_rum`; this function only
    expects the orchestrator (`run_all_validation.py`) to pass a
    closure `ours_predict_fn(data, intervention) -> {"base", "cf"}`.
    Falls back to `None` when no closure is supplied AND no trainer
    can be imported, so triangulation degrades gracefully.
    """
    if ours_predict_fn is not None:
        return ours_predict_fn(data, OOC_JOBS_DELTA)
    if not _HAS_INVERSE_RUM:
        return None
    # Final fallback: report None — the orchestrator should always
    # provide an explicit closure when running the real pipeline.
    return None


# ---------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------

def run_triangulation(
    data: FitData,
    ours_predict_fn: Optional[Callable[..., Dict[str, np.ndarray]]] = None,
    out_csv: Optional[Path] = None,
) -> TriangulationResult:
    catchment_mask = _ooc_catchment_mask(data.grid_features, data.ooc_grid_id)
    catchment_ids = set(data.grid_features.loc[catchment_mask, "grid_id"])

    runs = {
        "wilson_gravity": _run_gravity(data),
        "radiation": _run_radiation(data),
        "deep_gravity": _run_deep_gravity(data),
        "ours_tsigcn_irum": _run_ours(data, ours_predict_fn),
    }

    rows = []
    ours_mag = None
    for name, r in runs.items():
        if r is None:
            rows.append({"model": name, "direction": np.nan,
                         "magnitude": np.nan, "overlaps_with_ours": "skip"})
            continue
        mag = _delta_flow_ooc(r["base"], r["cf"], data.od_pairs, catchment_ids)
        rows.append({"model": name, "direction": int(np.sign(mag)),
                     "magnitude": mag, "overlaps_with_ours": ""})
        if name == "ours_tsigcn_irum":
            ours_mag = mag

    df = pd.DataFrame(rows)
    if ours_mag is not None and np.isfinite(ours_mag) and ours_mag != 0:
        df["overlaps_with_ours"] = df["magnitude"].apply(
            lambda m: "Y" if (np.isfinite(m)
                              and abs(m - ours_mag) <= 0.5 * abs(ours_mag))
                       else ("N" if np.isfinite(m) else "skip")
        )

    # Direction agreement (vs ours, or majority-sign if ours is None).
    if ours_mag is not None and np.isfinite(ours_mag):
        target_sign = int(np.sign(ours_mag))
    else:
        signs = [r["direction"] for r in rows
                 if r["model"] != "ours_tsigcn_irum"
                 and np.isfinite(r["direction"])]
        target_sign = int(np.sign(sum(signs))) if signs else 0
    agree = sum(1 for r in rows if np.isfinite(r["direction"])
                and int(r["direction"]) == target_sign)
    passes = agree >= 3

    out_csv = Path(out_csv) if out_csv else OUT_DIR / "triangulation.csv"
    df.to_csv(out_csv, index=False)
    return TriangulationResult(df=df, direction_agreement=agree, passes=passes)


if __name__ == "__main__":   # pragma: no cover
    print("multi_model_triangulation.py — call run_triangulation(...) from "
          "run_all_validation.py orchestrator")
