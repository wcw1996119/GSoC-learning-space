"""Phase B GNN ablation - mixture (K=3 income tiers) + OccMatch vs Paper A baseline.

Compares the Phase B inverse-RUM trainer (income-tier mixture + occupation-
match auxiliary head) against the Paper A baseline (no mixture, no OccMatch)
on London 2021 real OD. Holds everything else fixed (StructuralGNN utility,
flat softmax, K=50 choice set, 150 epochs, residualise=False).

Per seed we train BOTH conditions and record:
- val CPC (Sorensen-style flow similarity)
- recovered beta_t (scalar for baseline; per-tier for phase_B)
- gamma (log-distance coefficient)
- delta (OccMatch coefficient; phase_B only)
- VOT per tier = (beta_tier / beta_c) * 60 (phase_B only)

Outputs
-------
  evaluation_outputs/paper_a/gnn_ablation_B.csv         (per (seed, condition))
  evaluation_outputs/paper_a/gnn_ablation_B_summary.json (mean + delta)

CLI
---
  python gnn_ablation_B.py --n_seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

import torch
from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN

OUT_DIR = V2_ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "gnn_ablation_B.csv"
SUMMARY_PATH = OUT_DIR / "gnn_ablation_B_summary.json"


def build_edge_index(t_ij: torch.Tensor, k: int = 10) -> torch.Tensor:
    if t_ij.dim() == 3:
        t_ij = t_ij.mean(0)
    N = t_ij.shape[0]
    t_pen = t_ij.clone()
    t_pen.fill_diagonal_(float("inf"))
    nbrs = torch.topk(t_pen, k=k, largest=False).indices
    src = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    return torch.stack([src, dst], dim=0)


def cpc(F_obs_np: np.ndarray, F_pred_np: np.ndarray, mask_np: np.ndarray) -> float:
    """CPC = 2 sum(min(pred, obs)) / (sum(pred) + sum(obs)). Lenormand 2012."""
    obs_masked = F_obs_np[mask_np]
    pred_masked = F_pred_np[mask_np]
    num = 2.0 * np.minimum(pred_masked, obs_masked).sum()
    den = max(pred_masked.sum() + obs_masked.sum(), 1.0)
    return float(num / den)


def load_data():
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz",
                         allow_pickle=True))
    F_ij = torch.tensor(cache["F_ij_t"].sum(0), dtype=torch.float32)   # daily
    # 2026-05-05 BUGFIX: Use car_freeflow_t_ij.npy (real minutes, median 32min,
    # max 68min — matches W4 synthetic_recovery convention) instead of
    # demo_cache.npz t_ij_t (which is congestion-adjusted COST composite —
    # median 117 unit at hour 8, ~3.58x freeflow). Mismatched units caused
    # baseline β = -0.042 (which is ~3-5x off WebTAG) → VOT 2.5 GBP/h. With
    # freeflow minutes the recovered β should be in the WebTAG-aligned range.
    t_ij = torch.tensor(
        np.load(V2_ROOT / "data" / "processed" / "car_freeflow_t_ij.npy").astype(np.float32),
        dtype=torch.float32,
    )  # (1725, 1725) real minutes, free-flow car
    static = torch.tensor(cache["static_features"], dtype=torch.float32)
    coords = cache["coords_bng"]
    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d = torch.tensor(np.log1p(d_km), dtype=torch.float32)
    train_mask = torch.tensor(cache["train_mask"])
    val_mask = torch.tensor(cache["val_mask"])

    aux = np.load(V2_ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    occ_match = torch.tensor(aux["occ_match"], dtype=torch.float32)
    income_score = torch.tensor(aux["income_score_per_origin"], dtype=torch.float32)
    # Phase B-2d: destination-wage interaction. β_eff(i,j) = β·(1 + φ·z_o(i) + ψ·z_w(j)).
    # ψ > 0 expected (rich workers to high-wage CBD destinations show higher VOT).
    wage_score = torch.tensor(aux["wage_score_per_dest"], dtype=torch.float32)

    return {
        "F_ij": F_ij,
        "t_ij": t_ij,
        "log_d": log_d,
        "static": static,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "occ_match": occ_match,
        "income_score": income_score,
        "wage_score": wage_score,
    }


def train_one(condition: str, seed: int, data: dict, edge_index: torch.Tensor) -> dict:
    """Train a single (condition, seed) trainer; return result row."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Fdim = data["static"].shape[-1]
    # Phase B-2f (2026-05-05): bigger GNN to better absorb V_j signal —
    # if V is fully captured, β·t reflects pure time disutility (not residual).
    util_net = StructuralGNN(in_features=Fdim, hidden=128, out=1, depth=5)

    common_kwargs = dict(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=data["F_ij"],
        t_ij_t=data["t_ij"],
        log_d_ij=data["log_d"],
        K=50,
        utility_net=util_net,
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        device="cpu",
        seed=seed,
        epochs=150,
        patience=150,
        residualise=False,
    )
    if condition == "phase_B":
        trainer = InverseRUMTrainer(
            **common_kwargs,
            occ_match=data["occ_match"],
            income_score_per_origin=data["income_score"],
            wage_score_per_dest=data["wage_score"],          # B-2d: dest wage interaction
            enable_wage_attraction=True,                     # B-2e: explicit ξ · log(wage_j)
            enforce_mainstream_direction=True,                # Variant 6: constrain φ≤0, ψ≥0
        )
    elif condition == "baseline":
        trainer = InverseRUMTrainer(**common_kwargs)
    else:
        raise ValueError(condition)

    t0 = time.time()
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    train_seconds = time.time() - t0

    # Predicted OD -> CPC on val mask.
    pred = trainer.predict_OD().detach().cpu().numpy()
    if pred.ndim == 3:
        pred_F = pred[0] if pred.shape[0] == 1 else pred.sum(0)
    else:
        pred_F = pred
    F_obs_np = data["F_ij"].cpu().numpy()
    val_mask_np = data["val_mask"].cpu().numpy().astype(bool)
    val_cpc = cpc(F_obs_np, pred_F, val_mask_np)

    # Pull beta_c for VOT.
    beta_c_val = float(trainer.rum.beta_c.item())

    # beta_t_hat shape: (T, K). For paper A B-2c, K=1 (no mixture); we squeeze.
    bt = beta_t_hat.detach().cpu().numpy() if hasattr(beta_t_hat, "detach") \
        else np.asarray(beta_t_hat)
    bt_scalar = float(bt.mean())  # T=1, K=1 → scalar after squeeze

    if condition == "phase_B":
        # Interaction path: β_eff(i, j) = β_t · (1 + φ·z_o(i) + ψ·z_w(j))
        # We report VOT by DESTINATION WAGE tertile (paper A's primary
        # heterogeneity output): commuters to high-wage destinations are by
        # selection richer workers (Train 2009 §6.4 covariate interactions).
        # Mainstream RP/SP literature: rich → high VOT.
        # So: VOT_low_income (commuters to LOW wage dest) → expect lower VOT
        #     VOT_high_income (commuters to HIGH wage dest) → expect higher VOT
        phi_val = float(trainer.rum.phi.item())
        psi_val = float(trainer.rum.psi.item())
        delta_val = float(trainer.rum.delta.item())
        score_o = trainer.income_score_per_origin.cpu().numpy()  # z-scored IMD
        score_w = trainer.wage_score_per_dest.cpu().numpy()       # z-scored log wage_j
        # Origin-IMD percentiles (for context only; not headline)
        zo_p83 = np.percentile(score_o, 83)
        zo_p50 = np.percentile(score_o, 50)
        zo_p17 = np.percentile(score_o, 17)
        # Destination-wage percentiles (HEADLINE; matches WebTAG income gradient)
        zw_low_inc = np.percentile(score_w, 17)   # low wage j = low-income work
        zw_mid_inc = np.percentile(score_w, 50)
        zw_high_inc = np.percentile(score_w, 83)  # high wage j = high-income work
        # β_eff at "average origin × dest tertile"
        beta_low = bt_scalar * (1.0 + phi_val * 0.0 + psi_val * zw_low_inc)
        beta_mid = bt_scalar * (1.0 + phi_val * 0.0 + psi_val * zw_mid_inc)
        beta_high = bt_scalar * (1.0 + phi_val * 0.0 + psi_val * zw_high_inc)
        if abs(beta_c_val) > 1e-9:
            vot_low = (beta_low / beta_c_val) * 60.0     # commuters to low-wage j (vs WebTAG £8)
            vot_mid = (beta_mid / beta_c_val) * 60.0     # mid-wage j (vs WebTAG £13)
            vot_high = (beta_high / beta_c_val) * 60.0   # high-wage j (vs WebTAG £22)
        else:
            vot_low = vot_mid = vot_high = float("nan")
    else:
        # Baseline: single scalar beta_t.
        phi_val = float("nan")
        psi_val = float("nan")
        beta_low = bt_scalar
        beta_mid = float("nan")
        beta_high = float("nan")
        delta_val = float("nan")
        vot_low = vot_mid = vot_high = float("nan")

    return {
        "seed": seed,
        "condition": condition,
        "val_cpc": val_cpc,
        "beta_base": bt_scalar,
        "phi": phi_val,                 # origin-income interaction coefficient
        "psi": psi_val,                 # destination-wage interaction coefficient
        "beta_low": beta_low,           # β_eff for commuters to LOW-wage dest (vs WebTAG £8)
        "beta_mid": beta_mid,
        "beta_high": beta_high,         # β_eff for commuters to HIGH-wage dest (vs WebTAG £22)
        "gamma": float(gamma_hat) if not hasattr(gamma_hat, "item") else float(gamma_hat.item()),
        "delta": delta_val,
        "vot_low": float(vot_low),      # commuters to low-wage j (low income workers)
        "vot_mid": float(vot_mid),
        "vot_high": float(vot_high),    # commuters to high-wage j (high income workers)
        "training_time_s": train_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    print("[gnn_ablation_B] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    print(f"[gnn_ablation_B] static={tuple(data['static'].shape)} "
          f"edges={edge_index.shape[1]} interaction_path=True")

    rows: list[dict] = []
    for seed in range(args.n_seeds):
        for cond in ("baseline", "phase_B"):
            print(f"\n[seed={seed} | cond={cond}] training ...")
            row = train_one(cond, seed, data, edge_index)
            rows.append(row)
            if cond == "baseline":
                print(f"  [OK] val_CPC={row['val_cpc']:.4f} "
                      f"beta={row['beta_low']:.4f} "
                      f"gamma={row['gamma']:.4f} "
                      f"t={row['training_time_s']:.1f}s")
            else:
                print(f"  [OK] val_CPC={row['val_cpc']:.4f} "
                      f"beta_base={row['beta_base']:.4f} "
                      f"phi={row['phi']:+.4f} psi={row['psi']:+.4f} "
                      f"delta={row['delta']:+.4f}")
                print(f"       beta_eff (by dest wage): "
                      f"low_inc(low_wage_j)={row['beta_low']:.4f} "
                      f"mid={row['beta_mid']:.4f} "
                      f"high_inc(high_wage_j)={row['beta_high']:.4f}")
                print(f"       VOT vs WebTAG: low=GBP{row['vot_low']:.2f}/8  "
                      f"mid=GBP{row['vot_mid']:.2f}/13  "
                      f"high=GBP{row['vot_high']:.2f}/22  "
                      f"t={row['training_time_s']:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[gnn_ablation_B] wrote {CSV_PATH}")

    base = df[df["condition"] == "baseline"]
    pb = df[df["condition"] == "phase_B"]
    base_cpc_mean = float(base["val_cpc"].mean()) if len(base) else float("nan")
    pb_cpc_mean = float(pb["val_cpc"].mean()) if len(pb) else float("nan")
    cpc_delta = pb_cpc_mean - base_cpc_mean

    summary = {
        "n_seeds": int(args.n_seeds),
        "baseline_val_cpc_mean": base_cpc_mean,
        "phase_B_val_cpc_mean": pb_cpc_mean,
        "phase_B_minus_baseline_cpc": cpc_delta,
        "phase_B_beta_base_mean": float(pb["beta_base"].mean()) if len(pb) else float("nan"),
        "phase_B_phi_mean": float(pb["phi"].mean()) if len(pb) else float("nan"),
        "phase_B_phi_std": float(pb["phi"].std()) if len(pb) else float("nan"),
        "phase_B_psi_mean": float(pb["psi"].mean()) if len(pb) else float("nan"),
        "phase_B_psi_std": float(pb["psi"].std()) if len(pb) else float("nan"),
        "phase_B_beta_low_mean": float(pb["beta_low"].mean()) if len(pb) else float("nan"),
        "phase_B_beta_mid_mean": float(pb["beta_mid"].mean()) if len(pb) else float("nan"),
        "phase_B_beta_high_mean": float(pb["beta_high"].mean()) if len(pb) else float("nan"),
        "phase_B_delta_mean": float(pb["delta"].mean()) if len(pb) else float("nan"),
        "phase_B_vot_low_mean": float(pb["vot_low"].mean()) if len(pb) else float("nan"),
        "phase_B_vot_mid_mean": float(pb["vot_mid"].mean()) if len(pb) else float("nan"),
        "phase_B_vot_high_mean": float(pb["vot_high"].mean()) if len(pb) else float("nan"),
        "baseline_beta_mean": float(base["beta_low"].mean()) if len(base) else float("nan"),
        "baseline_gamma_mean": float(base["gamma"].mean()) if len(base) else float("nan"),
        "phase_B_gamma_mean": float(pb["gamma"].mean()) if len(pb) else float("nan"),
        "csv_path": str(CSV_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[gnn_ablation_B] wrote {SUMMARY_PATH}")

    print("\n[gnn_ablation_B summary]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
