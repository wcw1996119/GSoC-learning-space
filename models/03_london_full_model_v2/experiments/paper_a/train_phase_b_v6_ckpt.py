"""Train Phase B v1 Variant 6 (sign-constrained mainstream-direction trainer)
and save a clean checkpoint to disk for downstream scenario use (D3+).

Variant 6 = the only Phase B v1 trainer that recovers the mainstream
direction (φ ≤ 0, ψ ≥ 0 via softplus reparametrisation), at a CPC cost
of ~10% vs the unconstrained best. Recorded in STATUS_FOR_USER.md.

Run:
    python experiments/paper_a/train_phase_b_v6_ckpt.py [--seed 0]

Outputs:
    evaluation_outputs/paper_a/phase_b_v6_seed{seed}.pt
        - gnn_state    : StructuralGNN state_dict
        - rum_state    : _RUMHead state_dict
        - config       : reconstruction kwargs for StructuralGNN
        - data_keys    : list of data tensors needed (for D3 sanity)
        - metrics      : val_cpc, beta_c, beta_base, phi, psi, delta, gamma
        - meta         : seed, timestamp, hyperparams
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN

OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def cpc(F_obs: np.ndarray, F_pred: np.ndarray, mask: np.ndarray) -> float:
    obs = F_obs[mask]; pr = F_pred[mask]
    return float(2.0 * np.minimum(pr, obs).sum() / max(pr.sum() + obs.sum(), 1.0))


def load_data():
    cache = dict(np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    F_ij = torch.tensor(cache["F_ij_t"].sum(0), dtype=torch.float32)
    t_ij = torch.tensor(
        np.load(ROOT / "data" / "processed" / "car_freeflow_t_ij.npy").astype(np.float32),
        dtype=torch.float32,
    )
    static = torch.tensor(cache["static_features"], dtype=torch.float32)
    coords = cache["coords_bng"]
    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d = torch.tensor(np.log1p(d_km), dtype=torch.float32)
    train_mask = torch.tensor(cache["train_mask"])
    val_mask = torch.tensor(cache["val_mask"])

    aux = np.load(ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    occ_match = torch.tensor(aux["occ_match"], dtype=torch.float32)
    income_score = torch.tensor(aux["income_score_per_origin"], dtype=torch.float32)
    wage_score = torch.tensor(aux["wage_score_per_dest"], dtype=torch.float32)

    return {
        "F_ij": F_ij, "t_ij": t_ij, "log_d": log_d, "static": static,
        "train_mask": train_mask, "val_mask": val_mask,
        "occ_match": occ_match, "income_score": income_score, "wage_score": wage_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--K", type=int, default=50)
    args = parser.parse_args()

    print(f"[ckpt] loading data ...")
    data = load_data()
    edge_index = build_edge_index(data["t_ij"], k=10)
    N = data["static"].shape[0]
    Fdim = data["static"].shape[-1]
    print(f"[ckpt] N={N} F={Fdim} edges={edge_index.shape[1]} seed={args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    util_net = StructuralGNN(in_features=Fdim, hidden=args.hidden, out=1, depth=args.depth)
    trainer = InverseRUMTrainer(
        grid_features=data["static"],
        edge_index=edge_index,
        observed_OD=data["F_ij"],
        t_ij_t=data["t_ij"],
        log_d_ij=data["log_d"],
        K=args.K,
        utility_net=util_net,
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        device="cpu",
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=True,
        enforce_mainstream_direction=True,        # Variant 6
    )

    t0 = time.time()
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    train_seconds = time.time() - t0

    # Compute final val CPC
    pred = trainer.predict_OD().detach().cpu().numpy()
    pred_F = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else (pred.sum(0) if pred.ndim == 3 else pred)
    val_mask_np = data["val_mask"].cpu().numpy().astype(bool)
    val_cpc = cpc(data["F_ij"].cpu().numpy(), pred_F, val_mask_np)

    phi = float(trainer.rum.phi.item())
    psi = float(trainer.rum.psi.item())
    delta = float(trainer.rum.delta.item())
    beta_base = float(beta_t_hat.mean().item())
    beta_c = float(beta_c_hat)
    gamma = float(gamma_hat)
    print(f"[ckpt] trained in {train_seconds:.1f}s | val_CPC={val_cpc:.4f}")
    print(f"[ckpt] beta_base={beta_base:+.4f} beta_c={beta_c:+.4f} "
          f"phi={phi:+.4f} psi={psi:+.4f} delta={delta:+.4f} gamma={gamma:+.4f}")

    ckpt_path = OUT_DIR / f"phase_b_v6_seed{args.seed}.pt"
    torch.save(
        {
            "gnn_state": trainer.gnn.state_dict(),
            "rum_state": trainer.rum.state_dict(),
            "config": {
                "in_features": Fdim,
                "hidden": args.hidden,
                "depth": args.depth,
                "out": 1,
                "K": args.K,
                "enforce_mainstream_direction": True,
                "enable_wage_attraction": True,
            },
            "data_keys": [
                "static", "t_ij", "log_d", "occ_match",
                "income_score", "wage_score", "F_ij",
            ],
            "metrics": {
                "val_cpc": val_cpc,
                "beta_base": beta_base,
                "beta_c": beta_c,
                "phi": phi,
                "psi": psi,
                "delta": delta,
                "gamma": gamma,
            },
            "meta": {
                "seed": args.seed,
                "epochs": args.epochs,
                "patience": args.patience,
                "train_seconds": train_seconds,
                "variant": 6,
                "framework": "phase_B_v1_softplus_sign_constrained",
            },
        },
        ckpt_path,
    )
    print(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
