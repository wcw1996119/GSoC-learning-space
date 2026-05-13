"""W4 root-cause diag #2 — verify top-K loss is the broken path.

Hypothesis: _topk_log_likelihood has a structural bug — chosen destinations
are not forced into C_i (include_chosen=False, chosen_idx=None), so once V
trains to peak, the K sampled destinations miss observed-flow j's, F_at_idx
→ 0, loss collapses to a constant (~322.14), grad vanishes.

Test: re-run σ=0 and σ=0.5 with K=2000 > N=1725 to force the trainer's
full-softmax fallback path (`if N <= K: return None`, then _per_hour_log_p_full
+ _nll). If hypothesis is correct:
  - train_nll decreases monotonically (no 322.14 plateau)
  - β converges back to truth (-0.07) regardless of σ
  - W4 bias < 5%

Outputs trace to stdout. ~2 min per σ.
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


def run(grid_features, edge_index, F_obs, t_ij, log_d_ij,
        sigma: float, K: int, seed: int, epochs: int, device: str):
    print(f"\n{'='*72}")
    print(f"  σ={sigma}   K={K} (N={grid_features.shape[0]})   seed={seed}   epochs={epochs}")
    print(f"  full-softmax path: {'YES' if K >= grid_features.shape[0] else 'NO (top-K)'}")
    print(f"{'='*72}")
    torch.manual_seed(seed)
    Fdim = grid_features.shape[-1]
    student_net = StructuralGNN(in_features=Fdim, hidden=64, out=1, depth=3).to(device)
    trainer = InverseRUMTrainer(
        grid_features=grid_features, edge_index=edge_index,
        observed_OD=F_obs, t_ij_t=t_ij, log_d_ij=log_d_ij,
        K=K, utility_net=student_net, device=device, seed=seed,
        epochs=epochs, patience=epochs, verbose=False,
    )
    if sigma > 0:
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
            for p in (trainer.rum.raw_beta_t, trainer.rum.raw_gamma, trainer.rum.raw_beta_c):
                p.data.add_(torch.randn(p.shape, generator=gen).to(p.device) * sigma)
    beta_after = float(trainer.rum.beta_t.mean().item())
    print(f"β_true = {BETA_STAR:+.6f}   β_init_after_perturb = {beta_after:+.6f} "
          f"({(beta_after - BETA_STAR) / BETA_STAR * 100:+.2f}%)\n")

    print(f"{'ep':>4}  {'train_nll':>10}  {'val_nll':>10}  {'cpc':>6}  "
          f"{'β':>10}  {'γ':>8}  {'|grad_β|':>10}")
    print("-" * 72)
    t0 = time.time()
    log = []
    from models_lib.inverse_rum.inverse_trainer import _fwl_inspired_ridge_penalty
    for ep in range(epochs):
        trainer.gnn.train(); trainer.rum.train()
        trainer.optimizer.zero_grad()
        V_jt = trainer._forward_V(trainer.X)
        if trainer.residualise:
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
        gb = float(trainer.rum.raw_beta_t.grad.norm().item()) if trainer.rum.raw_beta_t.grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(
            list(trainer.gnn.parameters()) + list(trainer.rum.parameters()), 1.0
        )
        trainer.optimizer.step()
        trainer.gnn.eval(); trainer.rum.eval()
        with torch.no_grad():
            V_e = trainer._forward_V(trainer.X)
            if trainer.residualise:
                V_e = _fwl_inspired_ridge_penalty(V_e, trainer.t_ij_t, trainer.log_d_ij)
            log_p_e = trainer._per_hour_log_p_full(V_e, trainer.t_ij_t, trainer.log_d_ij)
            val_nll = trainer._nll(log_p_e, trainer.observed_OD, mask=trainer.val_mask).item()
            cpc_val = trainer._cpc(log_p_e, trainer.observed_OD, trainer.val_mask)
        b = float(trainer.rum.beta_t.mean().item())
        g = float(trainer.rum.gamma.item())
        log.append({"ep": ep, "loss": float(loss.item()), "val_nll": val_nll,
                    "cpc": cpc_val, "beta": b, "gamma": g, "grad_b": gb})
        if ep < 10 or ep % 10 == 0 or ep == epochs - 1:
            print(f"{ep:4d}  {float(loss.item()):10.4f}  {val_nll:10.4f}  "
                  f"{cpc_val:6.3f}  {b:+10.6f}  {g:+8.4f}  {gb:10.2e}")
    elapsed = time.time() - t0
    bias = (log[-1]["beta"] - BETA_STAR) / BETA_STAR * 100
    print(f"\nelapsed: {elapsed:.1f}s")
    print(f"FINAL: β={log[-1]['beta']:+.6f}  bias={bias:+.2f}%  γ={log[-1]['gamma']:+.4f}")
    losses = [r['loss'] for r in log]
    print(f"  train_nll: ep0={losses[0]:.2f}  ep5={losses[5]:.2f}  "
          f"ep{epochs-1}={losses[-1]:.2f}  monotone_decr={all(losses[i] >= losses[i+1] - 1.0 for i in range(0, min(20, len(losses)-1)))}")
    return log


def main():
    device = "cpu"
    print("Loading + synthesizing OD ...")
    grid_features, t_ij, log_d_ij = load_grid_features(device)
    edge_index = build_edge_index(t_ij, k=10)
    F_obs, _, _ = synthesize_od(
        grid_features, t_ij, log_d_ij, edge_index,
        n_total_trips=1_000_000, seed=0, device=device,
    )
    N = grid_features.shape[0]
    print(f"  N={N}  edges={edge_index.shape[1]}  F.sum={float(F_obs.sum()):.0f}")
    EPOCHS = 60
    K_FULL = 2000  # > N=1725 → forces full softmax fallback
    for sigma in [0.0, 0.5]:
        run(grid_features, edge_index, F_obs, t_ij, log_d_ij,
            sigma=sigma, K=K_FULL, seed=0, epochs=EPOCHS, device=device)


if __name__ == "__main__":
    main()
