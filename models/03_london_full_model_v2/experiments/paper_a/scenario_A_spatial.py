"""Scenario A — spatial intervention do(X_j: feature += Δ).

Generic harness: caller specifies (grid_indices, feature_name, delta_raw,
use_bpr, capacity_proxy). Loads a frozen Phase B v6 trainer ckpt, runs
baseline + scenario forward, returns ΔOD and summary metrics.

Intervention is applied in **raw feature units** (e.g. +65 000 jobs), then
re-z-scored using the cached static_mean / static_std so the GNN sees
the same normalisation as during training. Log-transformed columns
(see providers.features.LOG_TRANSFORM_COLS) are unscaled / re-scaled
through log1p / expm1.

Run:
    python experiments/paper_a/scenario_A_spatial.py \
        --ckpt evaluation_outputs/paper_a/phase_b_v6_seed0.pt \
        --feature total_employment --delta 65000 --grids 850

Outputs:
    evaluation_outputs/paper_a/scenario_A_<feature>_<delta>_<bpr>.json
    + console summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import (
    InverseRUMTrainer, StructuralGNN,
    apply_bpr, solve_user_equilibrium,
)
from providers.features import STATIC_COLS, LOG_TRANSFORM_COLS
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, load_data


def feature_index(name: str) -> int:
    if name not in STATIC_COLS:
        raise ValueError(f"Unknown feature {name!r}; valid: {STATIC_COLS}")
    return STATIC_COLS.index(name)


def apply_intervention_raw(
    X_norm: torch.Tensor,
    grid_indices: list[int],
    feature_name: str,
    delta_raw: float,
    static_mean: np.ndarray,
    static_std: np.ndarray,
) -> torch.Tensor:
    """Add ``delta_raw`` to feature[name] at given grids, in *raw* units.

    Round-trip path (per providers.features):
        z = (log1p(raw) - mu) / sd          (for log-transformed cols)
        z = (raw       - mu) / sd          (otherwise)

    Returns a new (N, F) tensor; original is not mutated.
    """
    f_idx = feature_index(feature_name)
    is_log = feature_name in LOG_TRANSFORM_COLS
    X = X_norm.clone().detach()

    z_old = X[:, f_idx].numpy()
    if is_log:
        log_raw_old = z_old * static_std[f_idx] + static_mean[f_idx]
        raw_old = np.expm1(log_raw_old)
        raw_new = np.maximum(raw_old, 0.0)
        for g in grid_indices:
            raw_new[g] = raw_old[g] + delta_raw
        log_raw_new = np.log1p(np.maximum(raw_new, 0.0))
        z_new = (log_raw_new - static_mean[f_idx]) / (static_std[f_idx] + 1e-9)
    else:
        raw_old = z_old * static_std[f_idx] + static_mean[f_idx]
        raw_new = raw_old.copy()
        for g in grid_indices:
            raw_new[g] = raw_old[g] + delta_raw
        z_new = (raw_new - static_mean[f_idx]) / (static_std[f_idx] + 1e-9)

    X[:, f_idx] = torch.tensor(z_new, dtype=X.dtype)
    return X


# Employment-cluster columns: sectoral employment + aggregate + office POI.
# We deliberately leave out non-job features (population, lat/lon, subway,
# retail/F&B POI) — those represent slower-moving infrastructure that does
# not respond instantaneously to a +Δjobs policy.
EMP_BLOCK = ["sec1_primary", "sec2_manufacturing", "sec3_construction",
             "sec4_retail", "sec5_fnb", "sec6_info_finance", "sec7_public",
             "sec8_other", "total_employment", "poi_office"]


def apply_coherent_employment_intervention(
    X_norm: torch.Tensor,
    grid_indices: list[int],
    delta_total_emp: float,
    static_mean: np.ndarray,
    static_std: np.ndarray,
) -> torch.Tensor:
    """Bump the employment cluster (sec1..sec8 + total_emp + poi_office) by a
    constant *fractional* change determined from baseline total_employment.

    Concretely, for each grid g in ``grid_indices``:
        ratio = 1 + delta_total_emp / max(raw_total_emp[g], 1)
        for col in EMP_BLOCK:
            raw_new[g, col] = raw_old[g, col] * ratio

    Then re-z-score each column. This keeps the within-grid sector mix and
    its office-POI tie intact; only the magnitude of the cluster changes.
    Returns a new (N, F) tensor.
    """
    X = X_norm.clone().detach()

    # 1) Recover raw total_employment to compute per-grid ratio
    f_te = feature_index("total_employment")
    z_te = X[:, f_te].numpy()
    log_raw_te = z_te * static_std[f_te] + static_mean[f_te]
    raw_te = np.expm1(log_raw_te)
    ratios = np.ones_like(raw_te)
    for g in grid_indices:
        ratios[g] = 1.0 + delta_total_emp / max(raw_te[g], 1.0)

    # 2) Apply ratio to each EMP_BLOCK column, log+zscore round-trip
    for col in EMP_BLOCK:
        f_idx = feature_index(col)
        is_log = col in LOG_TRANSFORM_COLS
        z_old = X[:, f_idx].numpy()
        if is_log:
            log_raw = z_old * static_std[f_idx] + static_mean[f_idx]
            raw = np.expm1(log_raw)
            raw_new = raw * ratios
            z_new = (np.log1p(np.maximum(raw_new, 0.0)) - static_mean[f_idx]) / (static_std[f_idx] + 1e-9)
        else:
            raw = z_old * static_std[f_idx] + static_mean[f_idx]
            z_new = (raw * ratios - static_mean[f_idx]) / (static_std[f_idx] + 1e-9)
        X[:, f_idx] = torch.tensor(z_new, dtype=X.dtype)
    return X


def build_trainer_from_ckpt(ckpt_path: Path, data: dict) -> tuple[InverseRUMTrainer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    edge_index = build_edge_index(data["t_ij"], k=10)
    util_net = StructuralGNN(
        in_features=cfg["in_features"], hidden=cfg["hidden"],
        out=cfg["out"], depth=cfg["depth"],
    )
    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=data["F_ij"],
        t_ij_t=data["t_ij"],
        log_d_ij=data["log_d"],
        K=cfg["K"],
        utility_net=util_net,
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        device="cpu",
        seed=0,
        epochs=1, patience=1,
        residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=cfg["enable_wage_attraction"],
        enforce_mainstream_direction=cfg["enforce_mainstream_direction"],
    )
    trainer.gnn.load_state_dict(ckpt["gnn_state"])
    trainer.rum.load_state_dict(ckpt["rum_state"])
    trainer.gnn.eval(); trainer.rum.eval()
    return trainer, ckpt


def predict_OD_with_X(trainer: InverseRUMTrainer, X_new: torch.Tensor,
                      t_ij_override: Optional[torch.Tensor] = None,
                      norm_stats: Optional[tuple] = None) -> torch.Tensor:
    """Swap trainer.X temporarily, call predict_OD, restore. Returns (T, N, N) flow.

    trainer.X is shape (T, N, F) (broadcast at construction). If X_new is
    (N, F), we expand it to match before swapping.

    ``norm_stats`` (optional baseline (mean, std) of V_jt) is plumbed through
    to StructuralGNN so counterfactual forwards do not re-centre on the
    intervened distribution.
    """
    X_orig = trainer.X
    if X_new.dim() == 2:
        X_use = X_new.unsqueeze(0).expand_as(X_orig).contiguous()
    else:
        X_use = X_new
    try:
        trainer.X = X_use.to(trainer.device)
        out = trainer.predict_OD(t_ij_override=t_ij_override, norm_stats=norm_stats)
    finally:
        trainer.X = X_orig
    return out


def run_scenario_A(
    ckpt_path: Path,
    grid_indices: list[int],
    feature_name: str,
    delta_raw: float,
    coherent: bool = False,
    use_bpr: bool = False,
    capacity_scale: float = 1.0,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    mult_cap: float = 2.5,
    max_iter: int = 30,
    tol: float = 1e-3,
) -> dict:
    data = load_data()
    trainer, ckpt = build_trainer_from_ckpt(ckpt_path, data)
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    mu = np.asarray(cache["static_mean"])
    sd = np.asarray(cache["static_std"])

    # Freeze V_jt scaling at baseline so the intervention is a structural ΔV
    # rather than re-centred-on-perturbed-distribution.
    X_base_3d = data["static"].unsqueeze(0).expand(trainer.T, -1, -1).contiguous()
    norm_stats = trainer.gnn.compute_norm_stats(X_base_3d, trainer.edge_index)

    # Baseline OD (free-flow t_ij), using the same frozen norm for consistency.
    F_base = predict_OD_with_X(trainer, data["static"], norm_stats=norm_stats).sum(0).cpu().numpy()

    # Build intervened X
    if coherent:
        X_int = apply_coherent_employment_intervention(
            data["static"], grid_indices, delta_raw, mu, sd,
        )
        intervention_mode = f"coherent[{','.join(EMP_BLOCK)}]"
    else:
        X_int = apply_intervention_raw(
            data["static"], grid_indices, feature_name, delta_raw, mu, sd,
        )
        intervention_mode = f"single[{feature_name}]"

    if use_bpr:
        # Capacity proxy: baseline inflow per grid * capacity_scale (so V/C≈1 baseline)
        baseline_inflow = F_base.sum(axis=0) + 1.0           # (N,) avoid 0
        capacity_j = torch.tensor(baseline_inflow * capacity_scale, dtype=torch.float32)
        t0 = data["t_ij"]                                     # free-flow

        def predict_flow(t_ij: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                F_pred = predict_OD_with_X(trainer, X_int, t_ij_override=t_ij,
                                           norm_stats=norm_stats)
            return F_pred.sum(0)                              # (N, N) daily flow

        ue = solve_user_equilibrium(
            t0_ij=t0, capacity_j=capacity_j, predict_flow=predict_flow,
            alpha=bpr_alpha, beta=bpr_beta, mult_cap=mult_cap,
            max_iter=max_iter, tol=tol,
        )
        F_scen = ue.flow_ij.cpu().numpy()
        ue_info = {
            "converged": bool(ue.converged), "n_iter": int(ue.n_iter),
            "final_rel_gap": float(ue.final_gap), "history": [float(g) for g in ue.history],
        }
    else:
        F_scen = predict_OD_with_X(trainer, X_int).sum(0).cpu().numpy()
        ue_info = None

    delta_OD = F_scen - F_base
    inflow_into_intervened = float(delta_OD[:, grid_indices].sum())
    total_abs_change = float(np.abs(delta_OD).sum())
    total_baseline = float(F_base.sum())

    # Top 10 destinations gaining flow
    dest_delta = delta_OD.sum(axis=0)                          # (N,)
    top10_idx = np.argsort(-dest_delta)[:10]
    bot10_idx = np.argsort(dest_delta)[:10]

    summary = {
        "feature": feature_name,
        "intervention_mode": intervention_mode,
        "coherent": coherent,
        "delta_raw": delta_raw,
        "n_grids_intervened": len(grid_indices),
        "grid_indices_first5": list(grid_indices[:5]),
        "use_bpr": use_bpr,
        "ue_info": ue_info,
        "total_baseline_flow": total_baseline,
        "total_abs_delta_flow": total_abs_change,
        "abs_delta_pct_of_baseline": 100.0 * total_abs_change / max(total_baseline, 1.0),
        "delta_inflow_into_intervened_grids": inflow_into_intervened,
        "ckpt_metrics": ckpt["metrics"],
        "top10_dest_gain": [
            {"grid_idx": int(i), "delta_inflow": float(dest_delta[i])} for i in top10_idx
        ],
        "top10_dest_loss": [
            {"grid_idx": int(i), "delta_inflow": float(dest_delta[i])} for i in bot10_idx
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=str(ROOT / "evaluation_outputs" / "paper_a" / "phase_b_v6_seed0.pt"))
    parser.add_argument("--grids", type=int, nargs="+", required=True,
                        help="grid indices to intervene on (0-based; e.g. 850)")
    parser.add_argument("--feature", type=str, default="total_employment")
    parser.add_argument("--delta", type=float, required=True,
                        help="raw additional units (e.g. 65000 for +65k jobs)")
    parser.add_argument("--coherent", action="store_true",
                        help="bump entire employment cluster (sec1..8 + total_emp + poi_office) "
                             "by same fractional change instead of single feature")
    parser.add_argument("--use_bpr", action="store_true")
    parser.add_argument("--capacity_scale", type=float, default=1.0)
    parser.add_argument("--bpr_alpha", type=float, default=0.15)
    parser.add_argument("--bpr_beta", type=float, default=4.0)
    parser.add_argument("--mult_cap", type=float, default=2.5,
                        help="cap on BPR multiplier; matches v2.x model.py default")
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    summary = run_scenario_A(
        ckpt_path=Path(args.ckpt),
        grid_indices=args.grids,
        feature_name=args.feature,
        delta_raw=args.delta,
        coherent=args.coherent,
        use_bpr=args.use_bpr,
        capacity_scale=args.capacity_scale,
        bpr_alpha=args.bpr_alpha,
        bpr_beta=args.bpr_beta,
        mult_cap=args.mult_cap,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    bpr_tag = "bpr" if args.use_bpr else "freeflow"
    coh_tag = "coherent" if args.coherent else "single"
    out_path = (ROOT / "evaluation_outputs" / "paper_a"
                / f"scenario_A_{args.feature}_{int(args.delta)}_{coh_tag}_{bpr_tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[scenario_A] wrote {out_path}")
    print(f"  feature={args.feature}  delta={args.delta}  use_bpr={args.use_bpr}")
    print(f"  n_grids_intervened={summary['n_grids_intervened']}")
    print(f"  Σ|ΔF| = {summary['total_abs_delta_flow']:.0f}  "
          f"({summary['abs_delta_pct_of_baseline']:.2f}% of baseline {summary['total_baseline_flow']:.0f})")
    print(f"  Δinflow into intervened grids = {summary['delta_inflow_into_intervened_grids']:+.0f}")
    if summary['ue_info']:
        print(f"  UE: converged={summary['ue_info']['converged']} "
              f"n_iter={summary['ue_info']['n_iter']} "
              f"rel_gap={summary['ue_info']['final_rel_gap']:.2e}")
    print(f"  Top-3 dest gain: " + ", ".join(
        f"g{d['grid_idx']}(+{d['delta_inflow']:.0f})" for d in summary['top10_dest_gain'][:3]))
    print(f"  Top-3 dest loss: " + ", ".join(
        f"g{d['grid_idx']}({d['delta_inflow']:.0f})" for d in summary['top10_dest_loss'][:3]))


if __name__ == "__main__":
    main()
