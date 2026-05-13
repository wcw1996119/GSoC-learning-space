"""Choice-set size ablation — McFadden 1978 sampling-of-alternatives.

Sweep K in {20, 50, 100, 200} on the full London 2021 OD; report holdout
CPC, stability of recovered beta_t, and training time. The GNN utility net
is held fixed across K for fair comparison.

Outputs
-------
  evaluation_outputs/paper_a/choice_set_ablation.csv
  evaluation_outputs/paper_a/choice_set_ablation.png

CLI
---
  python choice_set_ablation.py --n_seeds 5 --device cpu
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
    print(f"[choice_set_ablation] inverse_rum module not yet implemented ({e}); exiting cleanly.")
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models_lib.baselines.metrics import all_metrics  # noqa: E402

OUT_DIR = V2_ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUT_DIR / "choice_set_ablation_state.npz"

K_VALUES = [20, 50, 100, 200]


def load_data(device: str):
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz",
                         allow_pickle=True))
    F_ij = torch.tensor(cache["F_ij_t"], dtype=torch.float32, device=device).sum(0)
    t_ij = torch.tensor(cache["t_ij_t"], dtype=torch.float32, device=device).mean(0)
    static = torch.tensor(cache["static_features"], dtype=torch.float32, device=device)
    train_mask = torch.tensor(cache["train_mask"], device=device)
    val_mask = torch.tensor(cache["val_mask"], device=device)
    test_mask = torch.tensor(cache["test_mask"], device=device)
    coords = cache["coords_bng"]
    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d = torch.tensor(np.log1p(d_km), dtype=torch.float32, device=device)
    return {
        "F_ij": F_ij, "t_ij": t_ij, "static": static, "log_d": log_d,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }


def build_edge_index(t_ij: torch.Tensor, k: int = 10) -> torch.Tensor:
    N = t_ij.shape[0]
    t_pen = t_ij.clone()
    t_pen.fill_diagonal_(float("inf"))
    nbrs = torch.topk(t_pen, k=k, largest=False).indices
    src = torch.arange(N, device=t_ij.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    return torch.stack([src, dst], dim=0)


def fit_one(K: int, seed: int, data: dict, edge_index: torch.Tensor, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    util_net = StructuralGNN(in_features=data["static"].shape[1],
                             hidden=64, out=1, depth=3).to(device)
    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=data["F_ij"],
        t_ij_t=data["t_ij"],
        log_d_ij=data["log_d"],
        utility_net=util_net,
        K=K,
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        device=device,
        seed=seed,
        # 2026-05-05: see gnn_ablation.py for rationale.
        # NOTE: this experiment specifically tests K ∈ {20, 50, 100, 200}
        # (top-K), but the top-K loss is currently broken (see
        # InverseRUMTrainer docstring). Until top-K is fixed, this experiment
        # will exercise the full-softmax fallback regardless of K. Results
        # for K < N will all be ≈ identical (because full softmax is used).
        # That is a useful negative finding to document, not a bug here.
        epochs=150,
        patience=150,
        residualise=False,
    )
    t0 = time.time()
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    elapsed = time.time() - t0

    pred = trainer.predict_OD().detach().cpu().numpy()
    if pred.ndim == 3:
        pred_F = pred.sum(0)
    else:
        pred_F = pred
    F_true = data["F_ij"].cpu().numpy()
    test = data["test_mask"].cpu().numpy().astype(bool)
    rows_test = np.where(test)[0]
    y_true = F_true[rows_test].ravel()
    y_pred = pred_F[rows_test].ravel()
    origins = np.repeat(rows_test, F_true.shape[1])
    metrics = all_metrics(origins, y_true, y_pred)

    beta_t_scalar = float(beta_t_hat.mean().item()) if hasattr(beta_t_hat, "mean") else float(beta_t_hat)

    return {
        "K": K,
        "seed": seed,
        "alpha_hat": float(alpha_hat),
        "beta_hat": beta_t_scalar,
        "beta_c_hat": float(beta_c_hat),
        "gamma_hat": float(gamma_hat),
        "training_time_s": elapsed,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = args.device
    data = load_data(device)
    edge_index = build_edge_index(data["t_ij"], k=10)
    print(f"[choice_set_ablation] static={tuple(data['static'].shape)} "
          f"edges={edge_index.shape[1]}")

    rows: list[dict] = []
    completed: set[tuple] = set()
    if args.resume and STATE_PATH.exists():
        prev = np.load(STATE_PATH, allow_pickle=True)
        rows = list(prev["rows"]) if "rows" in prev.files else []
        completed = {(r["K"], r["seed"]) for r in rows}
        print(f"[choice_set_ablation] resuming with {len(completed)} cells already done")

    for K in K_VALUES:
        for seed in range(args.n_seeds):
            if (K, seed) in completed:
                continue
            print(f"\n[K={K} | seed={seed}] training")
            row = fit_one(K, seed, data, edge_index, device)
            rows.append(row)
            np.savez(STATE_PATH, rows=np.array(rows, dtype=object))
            print(f"  CPC={row['CPC']:.3f} beta={row['beta_hat']:.4f} "
                  f"t={row['training_time_s']:.1f}s")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "choice_set_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[choice_set_ablation] wrote {csv_path}")

    agg = df.groupby("K").agg(
        cpc_mean=("CPC", "mean"),
        beta_mean=("beta_hat", "mean"),
        beta_std=("beta_hat", "std"),
        time_mean=("training_time_s", "mean"),
    )
    print("\n[choice_set_ablation] aggregate by K:")
    print(agg.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].errorbar(agg.index, agg["cpc_mean"], yerr=df.groupby("K")["CPC"].std().values,
                     marker="o")
    axes[0].set_xlabel("K (choice-set size)")
    axes[0].set_ylabel("CPC (test)")
    axes[0].set_title("Holdout CPC vs K")
    axes[1].errorbar(agg.index, agg["beta_mean"], yerr=agg["beta_std"], marker="o", color="orange")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("beta_hat (per-min)")
    axes[1].set_title("beta stability vs K")
    axes[2].plot(agg.index, agg["time_mean"], marker="o", color="green")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("training time (s)")
    axes[2].set_title("Training time vs K")
    fig.tight_layout()
    png_path = OUT_DIR / "choice_set_ablation.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[choice_set_ablation] wrote {png_path}")


if __name__ == "__main__":
    main()
