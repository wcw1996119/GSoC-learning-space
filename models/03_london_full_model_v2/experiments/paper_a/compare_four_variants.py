"""Four-way comparison with multi-metric evaluation.

Variants:
  1. SINGLE         — single-branch StructuralGNN baseline
  2. DUAL_MIN       — dual-branch encoder, no augmentations
  3. DUAL_EXT       — dual-branch + Q2 OccMatch + Q1 income×β scalar
  4. DUAL_HET       — dual-branch + 3-mode mixture × 3-tier mixture + OccMatch
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
from models_lib.inverse_rum.dual_branch_mixture_trainer import DualBranchMixtureTrainer
from models_lib.inverse_rum.inverse_trainer import InverseRUMTrainer
from models_lib.inverse_rum.structural_gnn import StructuralGNN
from models_lib.inverse_rum.flow_metrics import all_metrics


def load_data(cache_name: str = "demo_cache.npz"):
    """Load shared inputs. ``cache_name`` lets callers swap in extended-feature
    caches (e.g., ``demo_cache_ext27.npz``) without modifying every scenario
    script. Defaults to the original 22-feature cache for backward compat."""
    cache_path = ROOT / "data" / "processed" / cache_name
    od_path = ROOT / "data" / "processed" / "grid_hourly_od_2019.npz"
    dyn_path = ROOT / "data" / "processed" / "grid_dynamic_features.npz"
    hourly_path = ROOT / "data" / "processed" / "hourly_node_features.npz"
    ff_path = ROOT / "data" / "processed" / "car_freeflow_t_ij.npy"
    aux_path = ROOT / "data" / "processed" / "paperA_v23_aux.npz"
    mode_path = ROOT / "data" / "processed" / "mode_inputs.npz"

    cache = np.load(cache_path, allow_pickle=True)
    od = np.load(od_path)
    dyn = np.load(dyn_path)
    hourly = np.load(hourly_path)
    ff_t_ij = np.load(ff_path).astype(np.float32)
    aux = np.load(aux_path)
    mode = np.load(mode_path)

    X_static = torch.from_numpy(cache["static_features"].astype(np.float32))
    X_dynamic = torch.from_numpy(dyn["X_dyn"])
    X_combined_legacy = torch.from_numpy(hourly["X_t"])
    F_ij_t = torch.from_numpy(od["F_ij_t"])
    train_mask = torch.from_numpy(cache["train_mask"])
    val_mask = torch.from_numpy(cache["val_mask"])
    coords = torch.from_numpy(cache["coords_bng"].astype(np.float32))
    # 33 London borough idx per grid (used as IRM environment partition)
    grid_borough_idx = torch.from_numpy(cache["grid_borough_idx"].astype(np.int64))
    occ_match = torch.from_numpy(aux["occ_match"].astype(np.float32))
    income_score = torch.from_numpy(aux["income_score_per_origin"].astype(np.float32))
    income_tier_props = torch.from_numpy(aux["income_tier_props"].astype(np.float32))
    t_car = torch.from_numpy(mode["t_car"])
    t_transit = torch.from_numpy(mode["t_transit"])
    t_walk = torch.from_numpy(mode["t_walk"])
    mode_share = torch.from_numpy(mode["mode_share"])

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
        income_tier_props=income_tier_props,
        t_car=t_car, t_transit=t_transit, t_walk=t_walk, mode_share=mode_share,
        grid_borough_idx=grid_borough_idx,
    )


def evaluate(pred, obs, mask):
    return all_metrics(pred, obs, mask, K_top=10)


def run_single(d, epochs=50, seed=0):
    print(f"\n[1/4] SINGLE-branch baseline (seed={seed})...")
    t0 = time.time()
    X_legacy = d["X_combined_legacy"]
    F_dim = X_legacy.shape[-1]
    gnn = StructuralGNN(node_dim=F_dim, hidden_dim=32, n_layers=2, gru_hidden=32)
    trainer = InverseRUMTrainer(
        grid_features=X_legacy, edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"], t_ij_t=d["t_ij_T"], log_d_ij=d["log_d"],
        utility_net=gnn, train_mask=d["train_mask"], val_mask=d["val_mask"],
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


def run_dual_min(d, epochs=50, seed=0):
    print(f"\n[2/4] DUAL_MIN (seed={seed})...")
    t0 = time.time()
    trainer = DualBranchInverseTrainer(
        X_static=d["X_static"], X_dynamic=d["X_dynamic"], edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"], t_ij_t=d["t_ij_T"], log_d_ij=d["log_d"],
        train_mask=d["train_mask"], val_mask=d["val_mask"],
        hidden_dim=32, gru_hidden=32,
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    enc, alpha, beta, beta_c, gamma, log = trainer.fit()
    pred = trainer.predict_OD()
    metrics = evaluate(pred, d["F_ij_t"], d["val_mask"])
    metrics["_train_nll"] = log.train_nll[-1]
    metrics["_val_nll"] = log.val_nll[-1]
    metrics["_beta_mean"] = beta.mean().item()
    metrics["_gamma"] = gamma
    metrics["_n_params"] = sum(p.numel() for p in enc.parameters())
    metrics["_n_epochs"] = len(log.train_nll)
    metrics["_time_s"] = time.time() - t0
    print(f"  done ({metrics['_time_s']:.0f}s, {metrics['_n_epochs']} epochs)")
    return metrics


def run_dual_ext(d, epochs=50, seed=0):
    print(f"\n[3/4] DUAL_EXT (Q2+Q1 scalar) (seed={seed})...")
    t0 = time.time()
    trainer = DualBranchInverseTrainer(
        X_static=d["X_static"], X_dynamic=d["X_dynamic"], edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"], t_ij_t=d["t_ij_T"], log_d_ij=d["log_d"],
        train_mask=d["train_mask"], val_mask=d["val_mask"],
        occ_match=d["occ_match"], income_score_per_origin=d["income_score"],
        hidden_dim=32, gru_hidden=32,
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    enc, alpha, beta, beta_c, gamma, log = trainer.fit()
    pred = trainer.predict_OD()
    metrics = evaluate(pred, d["F_ij_t"], d["val_mask"])
    metrics["_train_nll"] = log.train_nll[-1]
    metrics["_val_nll"] = log.val_nll[-1]
    metrics["_beta_mean"] = beta.mean().item()
    metrics["_gamma"] = gamma
    metrics["_delta"] = float(trainer.rum.delta.item())
    metrics["_phi"] = float(trainer.rum.phi.item())
    metrics["_n_params"] = sum(p.numel() for p in enc.parameters())
    metrics["_n_epochs"] = len(log.train_nll)
    metrics["_time_s"] = time.time() - t0
    print(f"  done ({metrics['_time_s']:.0f}s, {metrics['_n_epochs']} epochs)")
    return metrics


def run_dual_het(d, epochs=50, seed=0):
    print(f"\n[4/4] DUAL_HET (3-mode × 3-tier mixture + OccMatch) (seed={seed})...")
    t0 = time.time()
    trainer = DualBranchMixtureTrainer(
        X_static=d["X_static"], X_dynamic=d["X_dynamic"], edge_index=d["edge_index"],
        observed_OD=d["F_ij_t"],
        t_ij_per_mode={"car": d["t_car"], "transit": d["t_transit"], "walk": d["t_walk"]},
        mode_share_per_origin=d["mode_share"],
        income_tier_props=d["income_tier_props"],
        log_d_ij=d["log_d"],
        occ_match=d["occ_match"],
        train_mask=d["train_mask"], val_mask=d["val_mask"],
        hidden_dim=32, gru_hidden=32,
        epochs=epochs, patience=15, seed=seed, verbose=False,
    )
    enc, head, log = trainer.fit()
    pred = trainer.predict_OD()
    metrics = evaluate(pred, d["F_ij_t"], d["val_mask"])
    metrics["_train_nll"] = log.train_nll[-1]
    metrics["_val_nll"] = log.val_nll[-1]
    metrics["_beta_mean"] = float(head.beta_t_per_mode.mean().item())
    metrics["_gamma"] = float(head.gamma.item())
    metrics["_delta"] = float(head.delta.item())
    # Mode-specific beta means (averaged across hours)
    bm = head.beta_t_per_mode.mean(dim=0).tolist()                    # (M,)
    metrics["_beta_car"] = bm[0]
    metrics["_beta_transit"] = bm[1]
    metrics["_beta_walk"] = bm[2]
    # Tier scaling
    kap = head.kappa.tolist()                                         # (K,)
    metrics["_kappa"] = kap
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
    print(f"  mode_share row mean: car={d['mode_share'][:, 0].mean():.3f} "
          f"transit={d['mode_share'][:, 1].mean():.3f} walk={d['mode_share'][:, 2].mean():.3f}")
    print(f"  income_tier_props row mean: tier1={d['income_tier_props'][:, 0].mean():.3f} "
          f"tier2={d['income_tier_props'][:, 1].mean():.3f} tier3={d['income_tier_props'][:, 2].mean():.3f}")
    print(f"  t_car median: {d['t_car'].median():.1f}, t_transit median: {d['t_transit'].median():.1f}, "
          f"t_walk median: {d['t_walk'].median():.1f}")

    EPOCHS = 50
    results = {
        "SINGLE": run_single(d, epochs=EPOCHS),
        "DUAL_MIN": run_dual_min(d, epochs=EPOCHS),
        "DUAL_EXT": run_dual_ext(d, epochs=EPOCHS),
        "DUAL_HET": run_dual_het(d, epochs=EPOCHS),
    }

    # ---- print comparison table -----------------------------------------------
    print("\n" + "=" * 100)
    print("MULTI-METRIC 4-WAY COMPARISON  (50 epochs, seed=0, val=every 5th origin)")
    print("=" * 100)
    metric_keys = [
        ("cpc",                "CPC ↑"),
        ("rmse",               "RMSE ↓"),
        ("mae",                "MAE ↓"),
        ("pearson_r",          "Pearson r ↑"),
        ("spearman_rho",       "Spearman ρ ↑"),
        ("kl_obs_to_pred",     "KL(obs|pred) ↓"),
        ("topK_acc_K10",       "top-10 acc ↑"),
    ]
    extra_keys = [
        ("_train_nll",         "train NLL ↓"),
        ("_val_nll",           "val NLL ↓"),
        ("_beta_mean",         "β mean (overall)"),
        ("_gamma",             "γ"),
        ("_n_params",          "# params"),
        ("_time_s",            "fit time (s)"),
    ]
    fmt = "  {:<18} {:>12} {:>12} {:>12} {:>12}"
    print(fmt.format("metric", "SINGLE", "DUAL_MIN", "DUAL_EXT", "DUAL_HET"))
    print("  " + "-" * 70)
    for k, label in metric_keys + extra_keys:
        row = []
        for v in ["SINGLE", "DUAL_MIN", "DUAL_EXT", "DUAL_HET"]:
            x = results[v].get(k, "-")
            if isinstance(x, float):
                row.append(f"{x:.1f}" if abs(x) >= 100 else f"{x:.4f}")
            else:
                row.append(f"{x}")
        print(fmt.format(label, *row))

    # DUAL_HET reverse-engineered details
    het = results["DUAL_HET"]
    print("\n  --- DUAL_HET reverse-engineered parameters ---")
    print(f"  β_car (mean over hours):     {het['_beta_car']:+.4f}")
    print(f"  β_transit (mean over hours): {het['_beta_transit']:+.4f}")
    print(f"  β_walk (mean over hours):    {het['_beta_walk']:+.4f}")
    print(f"  κ tier scaling [1.0, κ_2, κ_3]: {[round(x, 3) for x in het['_kappa']]}")
    print(f"  δ (OccMatch coef):           {het['_delta']:+.4f}")
    print(f"  γ:                           {het['_gamma']:+.4f}")

    # VOT estimates per mode (assuming β_c = -1.0 per WebTAG cost numeraire)
    # VOT in £/h = β_t / β_c × 60 ; here β_c not in this trainer so we use a
    # rough reference: assume £6.5/h Wardman central, scale by β ratio
    # This is illustrative — full VOT computation needs β_c from cost numeraire
    print(f"\n  Implied relative VOT (β_walk : β_car : β_transit ratio):")
    bc, bt, bw = het['_beta_car'], het['_beta_transit'], het['_beta_walk']
    base = abs(bc)
    print(f"    car=1.00  transit={abs(bt)/base:.2f}  walk={abs(bw)/base:.2f}")

    # DUAL_EXT
    if "_delta" in results["DUAL_EXT"]:
        print(f"\n  --- DUAL_EXT reverse-engineered ---")
        print(f"  δ (OccMatch): {results['DUAL_EXT']['_delta']:+.4f}")
        print(f"  φ (income×β): {results['DUAL_EXT']['_phi']:+.4f}")

    out = ROOT / "evaluation_outputs" / "paper_a" / "four_variant_compare.json"
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
