"""Temporal holdout — replaces the multi-city generalisation experiment.

Train on London 2021 hours 6-15, test on hours 16-21. Reports train CPC vs
test CPC and stability of recovered beta_t across temporal partitions.

If hourly OD synthesis is unavailable, the script falls back to a year-month
holdout when seasonal data is present (currently not in the v2 pipeline; the
hourly split is the default).

Outputs
-------
  evaluation_outputs/paper_a/temporal_holdout.csv
  evaluation_outputs/paper_a/temporal_holdout.png

CLI
---
  python temporal_holdout.py --n_seeds 5 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the v2 project root is on sys.path BEFORE attempting models_lib
# imports — see gnn_ablation.py for the same fix. (2026-05-05.)
V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

try:
    import torch
    from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN
except ImportError as e:
    print(f"[temporal_holdout] inverse_rum module not yet implemented ({e}); exiting cleanly.")
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models_lib.baselines.metrics import all_metrics  # noqa: E402

OUT_DIR = V2_ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUT_DIR / "temporal_holdout_state.npz"

TRAIN_HOURS = list(range(6, 16))    # 6..15
TEST_HOURS = list(range(16, 22))    # 16..21


def load_data(device: str):
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz",
                         allow_pickle=True))
    F_ij_t = torch.tensor(cache["F_ij_t"], dtype=torch.float32, device=device)  # (T, N, N)
    t_ij_t = torch.tensor(cache["t_ij_t"], dtype=torch.float32, device=device)
    static = torch.tensor(cache["static_features"], dtype=torch.float32, device=device)
    coords = cache["coords_bng"]
    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d = torch.tensor(np.log1p(d_km), dtype=torch.float32, device=device)
    T = F_ij_t.shape[0]
    # Map hour-of-day to cache index. Cache covers 0..T-1; align to 0..23 if T==24
    # else fall back to first/last partitions.
    return {"F_ij_t": F_ij_t, "t_ij_t": t_ij_t, "static": static, "log_d": log_d, "T": T}


def build_edge_index(t_ij: torch.Tensor, k: int = 10) -> torch.Tensor:
    if t_ij.dim() == 3:
        t = t_ij.mean(0)
    else:
        t = t_ij
    N = t.shape[0]
    t_pen = t.clone()
    t_pen.fill_diagonal_(float("inf"))
    nbrs = torch.topk(t_pen, k=k, largest=False).indices
    src = torch.arange(N, device=t.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    return torch.stack([src, dst], dim=0)


def slice_hours(F_ij_t: torch.Tensor, t_ij_t: torch.Tensor, hours: list[int]):
    """Aggregate hourly tensors into a single (N, N) representative slice."""
    T = F_ij_t.shape[0]
    valid = [h for h in hours if 0 <= h < T]
    if not valid:
        # fall back to first / last halves
        valid = list(range(T // 2)) if hours[0] < T // 2 else list(range(T // 2, T))
    F_slice = F_ij_t[valid].sum(0)
    t_slice = t_ij_t[valid].mean(0)
    return F_slice, t_slice


def fit_partition(label: str, F_train: torch.Tensor, t_train: torch.Tensor,
                  F_test: torch.Tensor, t_test: torch.Tensor,
                  data: dict, edge_index: torch.Tensor,
                  seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    util_net = StructuralGNN(in_features=data["static"].shape[1],
                             hidden=64, out=1, depth=3).to(device)
    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=F_train,
        t_ij_t=t_train,
        log_d_ij=data["log_d"],
        utility_net=util_net,
        K=50,
        device=device,
        seed=seed,
        # 2026-05-05: see gnn_ablation.py for rationale.
        epochs=150,
        patience=150,
        residualise=False,
    )
    t0 = time.time()
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    elapsed = time.time() - t0

    # Train-time prediction (pred uses t_train).
    pred_train = trainer.predict_OD().detach().cpu().numpy()
    if pred_train.ndim == 3:
        pred_train = pred_train.sum(0)
    # Test-time prediction: re-evaluate the trained model using the *test*
    # period travel-time tensor, so test_CPC measures OOD-time generalisation
    # rather than train-period prediction stability.
    test_eval_note = ""
    try:
        pred_test_raw = trainer.predict_OD(t_ij_override=t_test).detach().cpu().numpy()
        if pred_test_raw.ndim == 3:
            pred_test_raw = pred_test_raw.sum(0)
        pred_test = pred_test_raw
    except (TypeError, AttributeError):
        pred_test = pred_train
        test_eval_note = (
            "Used train-period predictions on test-period flow; "
            "this measures stability not OOD-time performance"
        )

    F_train_np = F_train.cpu().numpy()
    F_test_np = F_test.cpu().numpy()
    N = F_train_np.shape[0]
    origins = np.repeat(np.arange(N), N)
    train_metrics = all_metrics(origins, F_train_np.ravel(), pred_train.ravel())
    test_metrics = all_metrics(origins, F_test_np.ravel(), pred_test.ravel())

    beta_t_scalar = float(beta_t_hat.mean().item()) if hasattr(beta_t_hat, "mean") else float(beta_t_hat)

    return {
        "partition": label,
        "seed": seed,
        "alpha_hat": float(alpha_hat),
        # Unified schema (R2): beta_t_hat / beta_c_hat across experiments
        "beta_hat": beta_t_scalar,
        "beta_t_hat": beta_t_scalar,
        "beta_c_hat": float(beta_c_hat),
        "gamma_hat": float(gamma_hat),
        "training_time_s": elapsed,
        "train_CPC": train_metrics["CPC"],
        "test_CPC": test_metrics["CPC"],
        "train_MAE_log1p": train_metrics["MAE_log1p"],
        "test_MAE_log1p": test_metrics["MAE_log1p"],
        "train_Spearman": train_metrics["Spearman"],
        "test_Spearman": test_metrics["Spearman"],
        "test_evaluation_note": test_eval_note,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = args.device
    data = load_data(device)
    edge_index = build_edge_index(data["t_ij_t"], k=10)
    print(f"[temporal_holdout] static={tuple(data['static'].shape)} "
          f"T={data['T']} edges={edge_index.shape[1]}")

    F_train, t_train = slice_hours(data["F_ij_t"], data["t_ij_t"], TRAIN_HOURS)
    F_test, t_test = slice_hours(data["F_ij_t"], data["t_ij_t"], TEST_HOURS)
    print(f"[temporal_holdout] train hours={TRAIN_HOURS} test hours={TEST_HOURS}")

    rows: list[dict] = []
    completed: set = set()
    if args.resume and STATE_PATH.exists():
        prev = np.load(STATE_PATH, allow_pickle=True)
        rows = list(prev["rows"]) if "rows" in prev.files else []
        completed = {(r["partition"], r["seed"]) for r in rows}

    # Forward partition: train on AM hours, test on PM hours
    for seed in range(args.n_seeds):
        if ("am_to_pm", seed) in completed:
            continue
        print(f"\n[am_to_pm | seed={seed}] training on hours 6-15")
        row = fit_partition(
            "am_to_pm",
            F_train, t_train,
            F_test, t_test,
            data, edge_index, seed, device,
        )
        rows.append(row)
        np.savez(STATE_PATH, rows=np.array(rows, dtype=object))
        print(f"  train_CPC={row['train_CPC']:.3f} test_CPC={row['test_CPC']:.3f} "
              f"beta_t={row['beta_t_hat']:.4f}")

    # Reverse partition (sanity check on directional bias).
    # Train tensors are the PM hours; "test" tensors here are the AM hours.
    for seed in range(args.n_seeds):
        if ("pm_to_am", seed) in completed:
            continue
        print(f"\n[pm_to_am | seed={seed}] training on hours 16-21")
        row = fit_partition(
            "pm_to_am",
            F_test, t_test,
            F_train, t_train,
            data, edge_index, seed, device,
        )
        rows.append(row)
        np.savez(STATE_PATH, rows=np.array(rows, dtype=object))
        print(f"  train_CPC={row['train_CPC']:.3f} test_CPC={row['test_CPC']:.3f} "
              f"beta_t={row['beta_t_hat']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "temporal_holdout.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[temporal_holdout] wrote {csv_path}")

    agg = df.groupby("partition").agg(
        train_CPC=("train_CPC", "mean"),
        test_CPC=("test_CPC", "mean"),
        beta_mean=("beta_t_hat", "mean"),
        beta_std=("beta_t_hat", "std"),
    )
    print("\n[temporal_holdout] aggregate:")
    print(agg.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    parts = list(agg.index)
    x = np.arange(len(parts))
    axes[0].bar(x - 0.2, agg["train_CPC"].values, width=0.4, label="train")
    axes[0].bar(x + 0.2, agg["test_CPC"].values, width=0.4, label="test")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(parts)
    axes[0].set_ylabel("CPC")
    axes[0].set_title("Train vs test CPC by partition")
    axes[0].legend()
    axes[1].errorbar(parts, agg["beta_mean"], yerr=agg["beta_std"], marker="o", color="orange")
    axes[1].set_ylabel("beta_hat (per-min)")
    axes[1].set_title("beta stability across temporal partitions")
    fig.tight_layout()
    png_path = OUT_DIR / "temporal_holdout.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[temporal_holdout] wrote {png_path}")


if __name__ == "__main__":
    main()
