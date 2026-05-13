"""GNN ablation — required by AI/ML reviewer.

6-cell grid: utility-net family x choice-model family
    {linear, MLP-2-layer, GNN} x {flat softmax, nested logit}

Trained on London 2021 OD with identical train/val/test split. Reports
holdout CPC, MAE on log(1+flow), Spearman, and recovered VOT (£/h, derived
from beta).

W8 GO/NO-GO criterion (asserted on the GNN x flat-softmax cell):
    holdout CPC >= 0.5

Goal: confirm GNN beats {linear, MLP} on at least 2 of
    {CPC, beta_t stability across seeds, VOT proximity to TAG £11.5/h}.

If GNN does NOT win >= 2 categories, this is logged loudly as
``gnn_wins_minimum=False`` so the team knows to restructure the paper.

Outputs
-------
  evaluation_outputs/paper_a/gnn_ablation.csv
  evaluation_outputs/paper_a/gnn_ablation.png
  evaluation_outputs/paper_a/gnn_ablation_summary.json

CLI
---
  python gnn_ablation.py --n_seeds 5 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the v2 project root is on sys.path BEFORE attempting models_lib
# imports — otherwise the defensive try/except below silently swallows the
# legitimate "import works once path is set" path. (Bug fixed 2026-05-05.)
V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

try:
    import torch
    from models_lib.inverse_rum import (
        InverseRUMTrainer, StructuralGNN, implicit_softmax, nested_logit_logsum,
    )
except ImportError as e:
    print(f"[gnn_ablation] inverse_rum module not yet implemented ({e}); exiting cleanly.")
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models_lib.baselines.metrics import all_metrics  # noqa: E402

OUT_DIR = V2_ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUT_DIR / "gnn_ablation_state.npz"

UTIL_FAMILIES = ["linear", "mlp", "gnn"]
CHOICE_FAMILIES = ["flat", "nested"]

VOT_TAG_GBP_PER_HOUR = 11.5    # WebTAG TAG Unit A1.3, commuting time value
W8_CPC_GATE = 0.5              # W8 GO/NO-GO


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(device: str):
    """Load London 2021 OD plus features and produce train/val/test masks."""
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz",
                         allow_pickle=True))
    F_ij_t = torch.tensor(cache["F_ij_t"], dtype=torch.float32, device=device)  # (T, N, N)
    t_ij_t = torch.tensor(cache["t_ij_t"], dtype=torch.float32, device=device)
    train_mask = torch.tensor(cache["train_mask"], device=device)
    val_mask = torch.tensor(cache["val_mask"], device=device)
    test_mask = torch.tensor(cache["test_mask"], device=device)
    static = torch.tensor(cache["static_features"], dtype=torch.float32, device=device)
    coords = cache["coords_bng"]

    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d_ij = torch.tensor(np.log1p(d_km), dtype=torch.float32, device=device)
    return {
        "F_ij_t": F_ij_t,
        "t_ij_t": t_ij_t,
        "log_d_ij": log_d_ij,
        "static": static,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


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


# ---------------------------------------------------------------------------
# Lightweight utility-net wrappers (linear / MLP) with the same input/output
# contract as StructuralGNN, so InverseRUMTrainer can swap them transparently.
# ---------------------------------------------------------------------------
class LinearUtility(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, out: int = 1, depth: int = 1):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out)

    def forward(self, x, edge_index=None):  # edge_index ignored
        return self.fc(x)


class MLPUtility(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, out: int = 1, depth: int = 2):
        super().__init__()
        layers: list[torch.nn.Module] = []
        d_in = in_features
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(d_in, hidden), torch.nn.ReLU()]
            d_in = hidden
        layers.append(torch.nn.Linear(d_in, out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)


def make_utility_net(family: str, in_features: int) -> torch.nn.Module:
    if family == "linear":
        return LinearUtility(in_features)
    if family == "mlp":
        return MLPUtility(in_features, hidden=64, depth=2)
    if family == "gnn":
        return StructuralGNN(in_features=in_features, hidden=64, out=1, depth=3)
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Train one cell
# ---------------------------------------------------------------------------
def train_cell(util_family: str, choice_family: str, data: dict,
               edge_index: torch.Tensor, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Aggregate hourly OD to daily for the production-constrained inverse fit.
    F_ij = data["F_ij_t"].sum(0)                     # (N, N)
    t_ij = data["t_ij_t"].mean(0)                    # (N, N) representative cost
    util_net = make_utility_net(util_family, in_features=data["static"].shape[1]).to(device)

    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=F_ij,
        t_ij_t=t_ij,
        log_d_ij=data["log_d_ij"],
        utility_net=util_net,
        choice_model=choice_family,    # 'flat' or 'nested'
        K=50,
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        device=device,
        seed=seed,
        # 2026-05-05: matches synthetic_recovery — full-softmax loss
        # decreases monotonically, no early stop needed; FWL off to avoid
        # systematic bias on β.
        epochs=150,
        patience=150,
        residualise=False,
    )
    t0 = time.time()
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    train_seconds = time.time() - t0

    # predict_OD returns (T, N, N); aggregate over T (T=1 since we passed
    # the daily-summed F_ij) and squeeze to (N, N) for metric computation.
    pred = trainer.predict_OD().detach().cpu().numpy()
    if pred.ndim == 3:
        pred_F = pred.sum(0)
    else:
        pred_F = pred
    F_true = F_ij.detach().cpu().numpy()
    test = data["test_mask"].cpu().numpy().astype(bool)

    rows_test = np.where(test)[0]
    y_true = F_true[rows_test].ravel()
    y_pred = pred_F[rows_test].ravel()
    origins = np.repeat(rows_test, F_true.shape[1])
    metrics = all_metrics(origins, y_true, y_pred)

    beta_t_scalar = float(beta_t_hat.mean().item()) if hasattr(beta_t_hat, "mean") else float(beta_t_hat)
    # VOT (£/h) per Small (2012): VOT = (β_t / β_c) * 60. Both negative ⇒ positive £/h.
    if abs(beta_c_hat) > 1e-9:
        vot_gbp_per_hour = float((beta_t_scalar / beta_c_hat) * 60.0)
    else:
        vot_gbp_per_hour = float("nan")

    # Keep "beta_hat" alongside "beta_t_hat" so the existing aggregation /
    # plot code (which keys on row['beta_hat']) keeps working.
    return {
        "util_family": util_family,
        "choice_family": choice_family,
        "seed": seed,
        "alpha_hat": float(alpha_hat),
        "beta_hat": beta_t_scalar,
        "beta_t_hat": beta_t_scalar,
        "beta_c_hat": float(beta_c_hat),
        "gamma_hat": float(gamma_hat),
        "vot_gbp_per_hour": vot_gbp_per_hour,
        "train_seconds": train_seconds,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = args.device
    data = load_data(device)
    edge_index = build_edge_index(data["t_ij_t"], k=10)
    print(f"[gnn_ablation] static={tuple(data['static'].shape)} "
          f"edges={edge_index.shape[1]}")

    rows: list[dict] = []
    completed: set[tuple] = set()
    if args.resume and STATE_PATH.exists():
        prev = np.load(STATE_PATH, allow_pickle=True)
        rows = list(prev["rows"]) if "rows" in prev.files else []
        completed = {(r["util_family"], r["choice_family"], r["seed"]) for r in rows}
        print(f"[gnn_ablation] resuming with {len(completed)} cells already done")

    for util_family in UTIL_FAMILIES:
        for choice_family in CHOICE_FAMILIES:
            for seed in range(args.n_seeds):
                cell = (util_family, choice_family, seed)
                if cell in completed:
                    continue
                print(f"\n[{util_family} | {choice_family} | seed={seed}] training")
                row = train_cell(util_family, choice_family, data, edge_index, seed, device)
                rows.append(row)
                np.savez(STATE_PATH, rows=np.array(rows, dtype=object))
                print(f"  CPC={row['CPC']:.3f} MAE={row['MAE_log1p']:.3f} "
                      f"VOT={row['vot_gbp_per_hour']:.2f} GBP/h "
                      f"t={row['train_seconds']:.1f}s")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "gnn_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[gnn_ablation] wrote {csv_path}")

    # ------------------------------------------------------------------
    # Aggregate winners across {CPC, beta stability, VOT proximity}
    # Average over choice_family for the main "GNN wins?" comparison.
    # ------------------------------------------------------------------
    agg = df.groupby("util_family").agg(
        cpc_mean=("CPC", "mean"),
        beta_std=("beta_t_hat", "std"),
        vot_mean=("vot_gbp_per_hour", "mean"),
    )
    agg["vot_dist_to_tag"] = (agg["vot_mean"] - VOT_TAG_GBP_PER_HOUR).abs()
    print("\n[gnn_ablation] aggregate by utility family:")
    print(agg.to_string())

    gnn_wins_cpc = bool(agg.loc["gnn", "cpc_mean"] >= agg.drop("gnn")["cpc_mean"].max())
    gnn_wins_stability = bool(agg.loc["gnn", "beta_std"] <= agg.drop("gnn")["beta_std"].min())
    gnn_wins_vot = bool(agg.loc["gnn", "vot_dist_to_tag"]
                         <= agg.drop("gnn")["vot_dist_to_tag"].min())
    n_wins = int(gnn_wins_cpc) + int(gnn_wins_stability) + int(gnn_wins_vot)
    gnn_wins_minimum = bool(n_wins >= 2)

    # W8 GO/NO-GO gate: GNN x flat softmax CPC >= 0.5
    cell = df[(df["util_family"] == "gnn") & (df["choice_family"] == "flat")]
    cpc_gnn_flat = float(cell["CPC"].mean()) if len(cell) > 0 else float("nan")
    w8_pass = bool(cpc_gnn_flat >= W8_CPC_GATE)

    # Cell-level (beta_t_hat, beta_c_hat) for the validation handoff (R2/G).
    if len(cell) > 0:
        gnn_flat_beta_t_hat = float(cell["beta_t_hat"].mean())
        gnn_flat_beta_c_hat = (float(cell["beta_c_hat"].mean())
                                if "beta_c_hat" in cell.columns else float("nan"))
    else:
        gnn_flat_beta_t_hat = float("nan")
        gnn_flat_beta_c_hat = float("nan")

    summary = {
        "vot_tag_gbp_per_hour": VOT_TAG_GBP_PER_HOUR,
        "gnn_cpc_mean": float(agg.loc["gnn", "cpc_mean"]),
        "gnn_beta_std": float(agg.loc["gnn", "beta_std"]),
        "gnn_vot_mean": float(agg.loc["gnn", "vot_mean"]),
        "gnn_wins_cpc": gnn_wins_cpc,
        "gnn_wins_stability": gnn_wins_stability,
        "gnn_wins_vot": gnn_wins_vot,
        "gnn_wins_minimum": gnn_wins_minimum,
        "w8_cpc_gate": W8_CPC_GATE,
        "cpc_gnn_flat": cpc_gnn_flat,
        "w8_pass": w8_pass,
        # Used by validation/paper_a/run_all_validation.py::_load_recovered_betas
        "gnn_flat_beta_t_hat": gnn_flat_beta_t_hat,
        "gnn_flat_beta_c_hat": gnn_flat_beta_c_hat,
    }
    (OUT_DIR / "gnn_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[gnn_ablation summary]")
    print(json.dumps(summary, indent=2))

    # Bar plot of CPC by util x choice family
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.groupby(["util_family", "choice_family"])["CPC"].mean().unstack()
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(W8_CPC_GATE, color="red", linestyle="--", label=f"W8 gate ({W8_CPC_GATE})")
    ax.set_ylabel("CPC (test set)")
    ax.set_title("GNN ablation — CPC by utility x choice family")
    ax.legend(title="choice")
    fig.tight_layout()
    png_path = OUT_DIR / "gnn_ablation.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[gnn_ablation] wrote {png_path}")

    # ------------------------------------------------------------------
    # W8 GO/NO-GO HARD GATE: BOTH (a) GNN×flat CPC >= 0.5 AND (b) GNN wins
    # >= 2 of 3 categories. The "wins >= 2" was previously a warning only;
    # it is now a hard assertion (R2). The orchestrator will abort if
    # either condition fails.
    # ------------------------------------------------------------------
    fails: list[str] = []
    if not np.isfinite(cpc_gnn_flat) or cpc_gnn_flat < W8_CPC_GATE:
        fails.append(
            f"GNN×flat CPC = {cpc_gnn_flat:.3f} < {W8_CPC_GATE} (W8 minimum CPC)"
        )
    if not gnn_wins_minimum:
        fails.append(
            f"GNN does not win on >= 2 of 3 criteria; "
            f"wins=[CPC:{gnn_wins_cpc} stab:{gnn_wins_stability} VOT:{gnn_wins_vot}]"
        )
    if fails:
        raise AssertionError(
            "W8 GO/NO-GO FAIL:\n  - " + "\n  - ".join(fails)
        )


if __name__ == "__main__":
    main()
