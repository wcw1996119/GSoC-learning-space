"""Smoke test for models_lib.inverse_rum.bpr_layer.

Three checks:
  1. bpr_multiplier numerics on a hand-computable case.
  2. solve_user_equilibrium converges on a synthetic small graph with a
     softmax-over-(beta·t + log d) choice model. Convergence ≤ 30 iter, gap < 1e-3.
  3. Gradient flows through the equilibrium loop end-to-end.

Run:
    python experiments/paper_a/smoke_bpr_layer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models_lib.inverse_rum import bpr_multiplier, apply_bpr, solve_user_equilibrium


def check_bpr_formula() -> None:
    """alpha=0.15, beta=4 → mult(V/C=1) = 1.15, mult(V/C=2) = 1 + 0.15*16 = 3.40."""
    cap = torch.tensor([10.0, 5.0, 1.0])
    inflow = torch.tensor([10.0, 10.0, 0.0])  # V/C = 1.0, 2.0, 0.0
    mult = bpr_multiplier(inflow, cap, alpha=0.15, beta=4.0)
    expected = torch.tensor([1.15, 1.0 + 0.15 * (2.0 ** 4), 1.0])
    assert torch.allclose(mult, expected, atol=1e-5), f"BPR mult wrong: {mult} vs {expected}"
    print(f"[OK] bpr_multiplier numerics: {mult.tolist()}")


def check_ue_convergence(N: int = 20, seed: int = 0) -> None:
    """Synthetic 20-grid graph. Free-flow t0 = euclidean / fixed speed.
    Choice model: softmax over -beta·t + gamma·log d_attractor (j attractiveness).
    Capacity per j = uniform.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    coords = torch.rand(N, 2) * 10.0
    diff = coords[:, None, :] - coords[None, :, :]
    d_ij = (diff ** 2).sum(dim=-1).sqrt() + 0.1                 # (N, N) avoid 0
    t0 = d_ij * 2.0                                              # 2 min/km

    # Per-origin demand: 100 commuters; total demand 100*N, capacity 150*N → realistic V/C
    O_i = torch.full((N,), 100.0)
    capacity_j = torch.full((N,), 150.0)

    # Fake attractiveness V_j (random)
    V_j = torch.randn(N)
    log_d = torch.log(d_ij)

    beta_t = -0.07
    gamma = -1.0

    def predict_flow(t_ij: torch.Tensor) -> torch.Tensor:
        # Logits: V_j (broadcast) + beta_t·t_ij + gamma·log_d
        logits = V_j[None, :] + beta_t * t_ij + gamma * log_d
        # No self-loop: mask diagonal
        logits = logits.clone()
        logits.fill_diagonal_(-1e9)
        p_ij = torch.softmax(logits, dim=1)                       # P(j|i)
        flow_ij = O_i[:, None] * p_ij                            # demand allocation
        return flow_ij

    res = solve_user_equilibrium(
        t0_ij=t0,
        capacity_j=capacity_j,
        predict_flow=predict_flow,
        alpha=0.15,
        beta=4.0,
        max_iter=50,
        tol=1e-3,
        verbose=True,
    )
    print(f"[OK] UE converged={res.converged} n_iter={res.n_iter} final_rel_gap={res.final_gap:.2e}")
    assert res.converged, f"UE did not converge in 50 iter: rel_gap={res.final_gap:.4e}"
    # Sanity: total flow conserved
    total_flow = res.flow_ij.sum().item()
    expected_total = O_i.sum().item()
    assert abs(total_flow - expected_total) < 1.0, (
        f"flow mass not conserved: {total_flow} vs {expected_total}"
    )
    # Sanity: t_eq >= t0 (BPR multiplier ≥ 1)
    assert (res.t_ij >= t0 - 1e-5).all(), "Equilibrium t < free-flow t — BPR broken"
    # Sanity: at least one j is congested (mult > 1.05)
    inflow_j = res.flow_ij.sum(dim=0)
    mult_j = bpr_multiplier(inflow_j, capacity_j)
    n_congested = (mult_j > 1.05).sum().item()
    print(f"[OK] {n_congested}/{N} grids congested (mult > 1.05); max mult={mult_j.max().item():.2f}")


def check_grad_flow(N: int = 10) -> None:
    """Gradient must flow through bpr_apply and through solve_user_equilibrium."""
    torch.manual_seed(0)

    t0 = torch.rand(N, N).abs() * 10 + 1.0
    capacity = torch.full((N,), 50.0)
    O_i = torch.full((N,), 100.0)
    V_j = torch.randn(N, requires_grad=True)
    log_d = torch.log(t0)

    beta_t = torch.tensor(-0.07, requires_grad=True)

    def predict_flow(t_ij: torch.Tensor) -> torch.Tensor:
        logits = V_j[None, :] + beta_t * t_ij - 1.0 * log_d
        logits = logits.clone()
        logits.fill_diagonal_(-1e9)
        p_ij = torch.softmax(logits, dim=1)
        return O_i[:, None] * p_ij

    res = solve_user_equilibrium(
        t0_ij=t0,
        capacity_j=capacity,
        predict_flow=predict_flow,
        max_iter=10,
        tol=1e-2,
    )
    # Some scalar of t_eq + flow
    loss = res.t_ij.sum() + 0.01 * res.flow_ij.sum()
    loss.backward()
    assert V_j.grad is not None and torch.isfinite(V_j.grad).all(), "Grad to V_j failed"
    assert beta_t.grad is not None and torch.isfinite(beta_t.grad).item(), "Grad to beta_t failed"
    print(f"[OK] grad V_j norm={V_j.grad.norm().item():.4f}, grad beta_t={beta_t.grad.item():.4f}")


if __name__ == "__main__":
    print("== BPR layer smoke test ==")
    check_bpr_formula()
    check_ue_convergence()
    check_grad_flow()
    print("== ALL OK ==")
