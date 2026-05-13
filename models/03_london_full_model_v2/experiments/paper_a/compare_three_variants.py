"""Three-way comparison with multi-metric evaluation.

Variants:
  1. SINGLE          — single-branch StructuralGNN baseline (Phase B+ Variant 6 era)
  2. DUAL_MINIMAL    — dual-branch encoder, NO Q1/Q2 augmentations
  3. DUAL_EXTENDED   — dual-branch encoder + Q2 OccMatch + Q1 income×β interaction

All three run on the same data, masks, optimizer settings, 50 epochs, seed=0.

7 metrics per variant: CPC, RMSE, MAE, Pearson r, Spearman ρ, KL(obs‖pred),
top-10 accuracy. Output saved as JSON.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum.dual_branch_trainer import DualBranchInverseTrainer
from models_lib.inverse_rum.inverse_trainer import InverseRUMTrainer
from models_lib.inverse_rum.structural_gnn import StructuralGNN
from models_lib.inverse_rum.flow_metrics import all_metrics


def load_data():
    cache_path = ROOT / "data" / "processed" / "demo_cache.npz"
    od_path = ROOT / "data" / "processed" / "grid_hourly_od_2019.npz"
    dyn_path = ROOT / "data" / "processed" / "grid_dynamic_features.npz"
    hourly_path = ROOT / "data" / "processed" / "hourly_node_features.npz"
    ff_path = ROOT / "data" / "processed" / "car_freeflow_t_ij.npy"
    aux_path = ROOT / "data" / "processed" / "paperA_v23_aux.npz"

    cache = np.load(cache_path, allow_pickle=True)
    od = np.load(od_path)
    dyn = np.load(dyn_path)
    hourly = np.load(hourly_path)
    ff_t_ij = np.load(ff_path).astype(np.float32)
    aux = np.load(aux_path)

    X_static = torch.from_numpy(cache["static_features"].astype(np.float32))
    X_dynamic = torch.from_numpy(dyn["X_dyn"])
    X_combined_legacy = torch.from_numpy(hourly["X_t"])
    F_ij_t = torch.from_numpy(od["F_ij_t"])
    train_mask = torch.from_numpy(cache["train_mask"])
    val_mask = torch.from_numpy(cache["val_mask"])
    coords = torch.from_numpy(cache["coords_bng"].astype(np.float32))
    occ_match = torch.from_numpy(aux["occ_match"].astype(np.float32))
    income_score = torch.from_numpy(aux["income_score_per_origin"].astype(np.float32))

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist_m = torch.linalg.norm(diff, dim=-1)
    dist_km = (dist_m / 1000.0).clamp(min=0.1)
    log_d = torch.log(dist_km)

    N = X_static.shape[0]
    T = X_dynamic.shape[0]
    t_ij_T = torch.from_numpy(ff_t_ij).unsqueeze(0).expand(T, N, N).contiguous()

    edges = []
    for i in range(N):
        di_km = (dist_m[i] / 1000.0)
        nbrs = torch.where((di_km > 0.0) & (di_km < 1.5))[0]
        for j in nbrs.tolist():
            edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return dict(
        X_static=X_static, X_dynamic=X_dynamic, X_combined_legacy=X_combined_legacy,
        F_ij_t=F_ij_t, t_ij_T=t_ij_T, log_d=log_d, edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, N=N, T=T,
        occ_match=occ_match, income_score=income_score,
    )


def evaluate(pred, obs, mask):
    return all_metrics(pred, obs, mask, K_top=10)


def run_single(d, epochs=50, seed=0):
    print(f"\n[1/3] SINGLE-branch baseline (seed={seed})...")
    t0 = time.time()
    X_legacy = d["X_combined_legacy"]
    F_dim = X_legacy.shape[-1]
    gnn = StructuralGNN(node_dim=F_dim, hidden_dim=32, n_layers=2, gru_hidden=32)
    trainer = InverseRUMTrainer(
        grid_features=X_legacy,
        edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"],
        t_ij_t=d["t_ij_T"],
        log_d_ij=d["log_d"],
        utility_net=gnn,
        train_mask=d["train_mask"],
        val_mask=d["val_mask"],
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    gnn_h, alpha, beta, beta_c, gamma, log = trainer.fit()
    pred = trainer.predict_OD()
    metrics = evaluate(pred, d["F_ij_t"], d["val_mask"])
    metrics["_train_nll"] = log.train_nll[-1]
    metrics["_val_nll"] = log.val_nll[-1]
    metrics["_beta_mean"] = beta.mean().item()
    metrics["_gamma"] = gamma
    metrics["_n_params"] = sum(p.numel() for p in gnn_h.parameters())
    metrics["_n_epochs"] = len(log.train_nll)
    metrics["_time_s"] = time.time() - t0
    print(f"  done ({metrics['_time_s']:.0f}s, {metrics['_n_epochs']} epochs)")
    return metrics


def run_dual(d, extended: bool, epochs=50, seed=0):
    label = "DUAL_EXTENDED" if extended else "DUAL_MINIMAL"
    idx = "[3/3]" if extended else "[2/3]"
    print(f"\n{idx} {label} (seed={seed})...")
    t0 = time.time()
    kwargs = dict(
        X_static=d["X_static"],
        X_dynamic=d["X_dynamic"],
        edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"],
        t_ij_t=d["t_ij_T"],
        log_d_ij=d["log_d"],
        train_mask=d["train_mask"],
        val_mask=d["val_mask"],
        hidden_dim=32, gru_hidden=32, n_sage_layers=2,
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    if extended:
        kwargs["occ_match"] = d["occ_match"]
        kwargs["income_score_per_origin"] = d["income_score"]
    trainer = DualBranchInverseTrainer(**kwargs)
    enc, alpha, beta, beta_c, gamma, log = trainer.fit()
    pred = trainer.predict_OD()
    metrics = evaluate(pred, d["F_ij_t"], d["val_mask"])
    metrics["_train_nll"] = log.train_nll[-1]
    metrics["_val_nll"] = log.val_nll[-1]
    metrics["_beta_mean"] = beta.mean().item()
    metrics["_gamma"] = gamma
    if extended:
        metrics["_delta"] = float(trainer.rum.delta.item())
        metrics["_phi"] = float(trainer.rum.phi.item())
    metrics["_n_params"] = sum(p.numel() for p in enc.parameters())
    metrics["_n_epochs"] = len(log.train_nll)
    metrics["_time_s"] = time.time() - t0
    print(f"  done ({metrics['_time_s']:.0f}s, {metrics['_n_epochs']} epochs)")
    return metrics


def main():
    print("Loading data...")
    d = load_data()
    print(f"  N={d['N']}, T={d['T']}, edges={d['edge_index'].shape[1]}")
    print(f"  F_ij_t total: {d['F_ij_t'].sum():.0f}")
    print(f"  occ_match: range [{d['occ_match'].min():.3f}, {d['occ_match'].max():.3f}]")
    print(f"  income_score (raw): range [{d['income_score'].min():.3f}, {d['income_score'].max():.3f}]")

    EPOCHS = 50
    results = {
        "SINGLE": run_single(d, epochs=EPOCHS),
        "DUAL_MINIMAL": run_dual(d, extended=False, epochs=EPOCHS),
        "DUAL_EXTENDED": run_dual(d, extended=True, epochs=EPOCHS),
    }

    # ---- print comparison table -----------------------------------------------
    print("\n" + "=" * 84)
    print("MULTI-METRIC COMPARISON  (50 epochs, seed=0, val mask = every 5th origin)")
    print("=" * 84)
    metric_keys = [
        ("cpc",                "CPC ↑          "),
        ("rmse",               "RMSE ↓         "),
        ("mae",                "MAE ↓          "),
        ("pearson_r",          "Pearson r ↑    "),
        ("spearman_rho",       "Spearman ρ ↑   "),
        ("kl_obs_to_pred",     "KL(obs|pred) ↓ "),
        ("topK_acc_K10",       "top-10 acc ↑   "),
    ]
    extra_keys = [
        ("_train_nll",         "train NLL ↓    "),
        ("_val_nll",           "val NLL ↓      "),
        ("_beta_mean",         "β mean         "),
        ("_gamma",             "γ              "),
        ("_n_params",          "# params       "),
        ("_n_epochs",          "# epochs       "),
        ("_time_s",            "fit time (s)   "),
    ]
    fmt = "  {:<16} {:>15} {:>15} {:>15}"
    print(fmt.format("metric", "SINGLE", "DUAL_MINIMAL", "DUAL_EXT (Q2+Q1)"))
    print("  " + "-" * 80)
    for k, label in metric_keys + extra_keys:
        row = []
        for variant in ["SINGLE", "DUAL_MINIMAL", "DUAL_EXTENDED"]:
            v = results[variant].get(k, "-")
            if isinstance(v, float):
                if abs(v) >= 100:
                    row.append(f"{v:.1f}")
                else:
                    row.append(f"{v:.4f}")
            else:
                row.append(f"{v}")
        print(fmt.format(label, *row))

    # specifically log Q2/Q1 reverse-engineered coefficients
    if "_delta" in results["DUAL_EXTENDED"]:
        print("  " + "-" * 80)
        print(f"  DUAL_EXTENDED  δ (OccMatch coef) = {results['DUAL_EXTENDED']['_delta']:+.4f}  "
              f"(expected POSITIVE if labour-match attracts)")
        print(f"  DUAL_EXTENDED  φ (income×β coef) = {results['DUAL_EXTENDED']['_phi']:+.4f}  "
              f"(positive = high-income origins more time-sensitive, mainstream)")

    out = ROOT / "evaluation_outputs" / "paper_a" / "three_variant_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "run": {"date": "2026-05-08", "epochs": EPOCHS, "seed": 0,
                    "n_grids": d["N"], "n_hours": d["T"], "edges": int(d["edge_index"].shape[1])},
            "results": results,
        }, f, indent=2)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
