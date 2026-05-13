"""Train Phase B v6 on 2011 OD data → ckpt for retrodiction.

Mirrors train_phase_b_v6_ckpt.py but:
  * F_ij from 2011 (NOMIS WU03EW disaggregated to grid)
  * static features = 2011 z-scored (employment scaled per-grid by
    BRES-substitute ratio; see methodology/retrodiction_scope.md §6)
  * t_ij still free-flow car (network ~time-invariant; documented caveat)
  * occ_match / income_score / wage_score from paperA_v23_aux.npz
    (2024 inputs — coarse but available; documented caveat)
  * everything else identical

Output: evaluation_outputs/paper_a/phase_b_v6_2011_seed{seed}.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN
from experiments.paper_a.train_phase_b_v6_ckpt import build_edge_index, cpc

OUT_DIR = ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_2011():
    """Identical schema to train_phase_b_v6_ckpt.load_data() but 2011 inputs."""
    cache_2011 = dict(np.load(ROOT / "data" / "processed" / "demo_cache_2011.npz",
                               allow_pickle=True))
    F_ij = torch.tensor(cache_2011["F_ij"], dtype=torch.float32)
    static = torch.tensor(cache_2011["static_features"], dtype=torch.float32)
    coords = cache_2011["coords_bng"]
    t_ij = torch.tensor(
        np.load(ROOT / "data" / "processed" / "car_freeflow_t_ij.npy").astype(np.float32),
        dtype=torch.float32,
    )
    diff = coords[:, None, :] - coords[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(-1)) / 1000.0
    log_d = torch.tensor(np.log1p(d_km), dtype=torch.float32)

    # Default mask: every 5th origin held out (matches Phase B v6 random split)
    N = len(cache_2011["grid_ids"])
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[::5] = True
    train_mask = ~val_mask

    aux = np.load(ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    occ_match = torch.tensor(aux["occ_match"], dtype=torch.float32)
    income_score = torch.tensor(aux["income_score_per_origin"], dtype=torch.float32)
    wage_score = torch.tensor(aux["wage_score_per_dest"], dtype=torch.float32)

    return {
        "F_ij": F_ij, "t_ij": t_ij, "log_d": log_d, "static": static,
        "train_mask": train_mask, "val_mask": val_mask,
        "occ_match": occ_match, "income_score": income_score,
        "wage_score": wage_score,
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

    print(f"[2011-ckpt] loading 2011 data ...")
    data = load_data_2011()
    edge_index = build_edge_index(data["t_ij"], k=10)
    N = data["static"].shape[0]; Fdim = data["static"].shape[-1]
    print(f"[2011-ckpt] N={N} F={Fdim} edges={edge_index.shape[1]} seed={args.seed}")
    print(f"[2011-ckpt] F_ij sum 2011: {data['F_ij'].sum().item():,.0f}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

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
        device="cpu", seed=args.seed,
        epochs=args.epochs, patience=args.patience, residualise=False,
        occ_match=data["occ_match"],
        income_score_per_origin=data["income_score"],
        wage_score_per_dest=data["wage_score"],
        enable_wage_attraction=True,
        enforce_mainstream_direction=True,
    )

    t0 = time.time()
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    train_seconds = time.time() - t0

    pred = trainer.predict_OD().detach().cpu().numpy()
    pred_F = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else (pred.sum(0) if pred.ndim == 3 else pred)
    val_mask_np = data["val_mask"].cpu().numpy().astype(bool)
    val_cpc = cpc(data["F_ij"].cpu().numpy(), pred_F, val_mask_np)
    train_cpc = cpc(data["F_ij"].cpu().numpy(), pred_F, data["train_mask"].cpu().numpy().astype(bool))

    phi = float(trainer.rum.phi.item())
    psi = float(trainer.rum.psi.item())
    delta = float(trainer.rum.delta.item())
    beta_base = float(beta_t_hat.mean().item())
    beta_c = float(beta_c_hat)
    gamma = float(gamma_hat)
    print(f"[2011-ckpt] trained in {train_seconds:.1f}s | train_CPC={train_cpc:.4f} val_CPC={val_cpc:.4f}")
    print(f"[2011-ckpt] beta_base={beta_base:+.4f} beta_c={beta_c:+.4f} "
          f"phi={phi:+.4f} psi={psi:+.4f} delta={delta:+.4f} gamma={gamma:+.4f}")

    ckpt_path = OUT_DIR / f"phase_b_v6_2011_seed{args.seed}.pt"
    torch.save(
        {
            "gnn_state": trainer.gnn.state_dict(),
            "rum_state": trainer.rum.state_dict(),
            "config": {
                "in_features": Fdim, "hidden": args.hidden, "depth": args.depth,
                "out": 1, "K": args.K,
                "enforce_mainstream_direction": True,
                "enable_wage_attraction": True,
            },
            "metrics": {
                "train_cpc": train_cpc, "val_cpc": val_cpc,
                "beta_base": beta_base, "beta_c": beta_c,
                "phi": phi, "psi": psi, "delta": delta, "gamma": gamma,
            },
            "meta": {
                "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
                "train_seconds": train_seconds, "data_year": 2011,
                "framework": "phase_B_v1_softplus_sign_constrained_2011",
            },
        },
        ckpt_path,
    )
    print(f"[2011-ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()
