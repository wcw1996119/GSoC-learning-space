"""W4 root-cause diagnostic — run InverseRUMTrainer with verbose tracing
across {sigma=0, sigma=0.15, sigma=0.5} on a single synthetic seed.

Goal: distinguish three failure modes for the W4 bias=36% result:
  (a) trainer DOES update beta but converges slowly → epochs/patience problem
  (b) trainer does NOT update beta → grad-flow problem (top-K stochasticity,
      grad clipping, FWL-residual zeroing the V channel, etc)
  (c) sigma=0.5 is simply too aggressive → trainer can update but cannot
      escape the perturbation basin in 50 epochs

Outputs traces to stdout; no CSV / PNG / state files. ~3-5 min per sigma.

Usage:
  python experiments/paper_a/diag_trainer.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

import torch
from models_lib.inverse_rum import InverseRUMTrainer, StructuralGNN

from experiments.paper_a.synthetic_recovery import (
    BETA_STAR, load_grid_features, build_edge_index, synthesize_od,
)


def diagnose_one(grid_features, edge_index, F_obs, t_ij, log_d_ij,
                 sigma: float, seed: int, epochs: int, device: str):
    """Single fit with full epoch trace."""
    print(f"\n{'='*72}\n  σ = {sigma}    seed = {seed}    epochs = {epochs}\n{'='*72}")
    torch.manual_seed(seed)
    Fdim = grid_features.shape[-1]
    student_net = StructuralGNN(in_features=Fdim, hidden=64, out=1, depth=3).to(device)

    trainer = InverseRUMTrainer(
        grid_features=grid_features,
        edge_index=edge_index,
        observed_OD=F_obs,
        t_ij_t=t_ij,
        log_d_ij=log_d_ij,
        K=50,
        utility_net=student_net,
        device=device,
        seed=seed,
        epochs=epochs,
        patience=epochs,            # disable early stop
        verbose=False,              # we print our own
    )

    # snapshot pre-perturbation init
    beta_init_raw = float(trainer.rum.raw_beta_t.mean().item())
    beta_init = float(trainer.rum.beta_t.mean().item())

    if sigma > 0:
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
            for p in (trainer.rum.raw_beta_t, trainer.rum.raw_gamma, trainer.rum.raw_beta_c):
                p.data.add_(torch.randn(p.shape, generator=gen).to(p.device) * sigma)

    beta_after_perturb = float(trainer.rum.beta_t.mean().item())
    gamma_after_perturb = float(trainer.rum.gamma.item())
    print(f"β_true = {BETA_STAR:+.6f}")
    print(f"β init (no perturb) = {beta_init:+.6f}     raw = {beta_init_raw:+.4f}")
    print(f"β after perturb     = {beta_after_perturb:+.6f}     "
          f"γ = {gamma_after_perturb:+.4f}")
    print(f"perturb bias        = {(beta_after_perturb - BETA_STAR) / BETA_STAR * 100:+.2f}%\n")

    print(f"{'ep':>4}  {'train_nll':>10}  {'val_nll':>10}  {'cpc':>6}  "
          f"{'β':>10}  {'γ':>8}  {'Δβ_step':>10}")
    print("-" * 72)
    t0 = time.time()
    prev_beta = beta_after_perturb

    # We re-implement the training loop here to print every epoch.
    log = []
    best_val = float("inf")
    best_ep = -1
    for ep in range(epochs):
        trainer.gnn.train(); trainer.rum.train()
        trainer.optimizer.zero_grad()
        V_jt = trainer._forward_V(trainer.X)
        if trainer.residualise:
            from models_lib.inverse_rum.inverse_trainer import _fwl_inspired_ridge_penalty
            V_jt = _fwl_inspired_ridge_penalty(V_jt, trainer.t_ij_t, trainer.log_d_ij)
        cs = trainer._build_choice_sets(V_jt) if trainer.K < trainer.N else None
        if cs is not None:
            loss = trainer._topk_log_likelihood(
                V_jt, trainer.t_ij_t, trainer.log_d_ij, trainer.observed_OD, cs,
                mask=trainer.train_mask,
            )
        else:
            log_p = trainer._per_hour_log_p_full(V_jt, trainer.t_ij_t, trainer.log_d_ij)
            loss = trainer._nll(log_p, trainer.observed_OD, mask=trainer.train_mask)
        loss.backward()
        # capture grad on raw_beta_t before clip
        grad_beta_raw = float(trainer.rum.raw_beta_t.grad.norm().item()) if trainer.rum.raw_beta_t.grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(
            list(trainer.gnn.parameters()) + list(trainer.rum.parameters()), 1.0
        )
        trainer.optimizer.step()

        trainer.gnn.eval(); trainer.rum.eval()
        with torch.no_grad():
            V_jt_e = trainer._forward_V(trainer.X)
            if trainer.residualise:
                V_jt_e = _fwl_inspired_ridge_penalty(V_jt_e, trainer.t_ij_t, trainer.log_d_ij)
            log_p_e = trainer._per_hour_log_p_full(V_jt_e, trainer.t_ij_t, trainer.log_d_ij)
            val_nll = trainer._nll(log_p_e, trainer.observed_OD, mask=trainer.val_mask).item()
            cpc_val = trainer._cpc(log_p_e, trainer.observed_OD, trainer.val_mask)

        cur_beta = float(trainer.rum.beta_t.mean().item())
        cur_gamma = float(trainer.rum.gamma.item())
        d_beta = cur_beta - prev_beta
        prev_beta = cur_beta
        if val_nll < best_val:
            best_val = val_nll
            best_ep = ep
        log.append({"ep": ep, "train_nll": float(loss.item()), "val_nll": val_nll,
                    "cpc": cpc_val, "beta": cur_beta, "gamma": cur_gamma,
                    "grad_beta_raw": grad_beta_raw})

        if ep < 10 or ep % 10 == 0 or ep == epochs - 1:
            print(f"{ep:4d}  {float(loss.item()):10.4f}  {val_nll:10.4f}  "
                  f"{cpc_val:6.3f}  {cur_beta:+10.6f}  {cur_gamma:+8.4f}  "
                  f"{d_beta:+10.6f}  (|grad_β_raw|={grad_beta_raw:.2e})")

    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")
    print(f"\nFINAL:")
    print(f"  β_final = {log[-1]['beta']:+.6f}    bias = {(log[-1]['beta'] - BETA_STAR) / BETA_STAR * 100:+.2f}%")
    print(f"  γ_final = {log[-1]['gamma']:+.4f}")
    print(f"  best val_nll @ epoch {best_ep} = {best_val:.4f}")
    betas = [r["beta"] for r in log]
    print(f"  β trajectory: min={min(betas):+.6f} max={max(betas):+.6f}  "
          f"|range|={max(betas) - min(betas):.6f}")
    grads = [r["grad_beta_raw"] for r in log]
    print(f"  |grad_raw_β| trajectory: mean={np.mean(grads):.3e} max={max(grads):.3e}")
    return log


def main():
    device = "cpu"
    print("Loading grid features and synthesizing OD (seed=0, n_trips=1M) ...")
    grid_features, t_ij, log_d_ij = load_grid_features(device)
    edge_index = build_edge_index(t_ij, k=10)
    F_obs, P_true, info = synthesize_od(
        grid_features, t_ij, log_d_ij, edge_index,
        n_total_trips=1_000_000, seed=0, device=device,
    )
    N = grid_features.shape[0]
    print(f"  N grids = {N}    edges = {edge_index.shape[1]}    F_obs.sum = {float(F_obs.sum()):.0f}")
    print(f"  ground truth: β={BETA_STAR}  α={0.30}  γ=-0.50")

    # Three diagnostic conditions, single seed each, 100 epochs no early stop.
    EPOCHS = 100
    SEED = 0
    for sigma in [0.0, 0.15, 0.5]:
        diagnose_one(grid_features, edge_index, F_obs, t_ij, log_d_ij,
                     sigma=sigma, seed=SEED, epochs=EPOCHS, device=device)


if __name__ == "__main__":
    main()
