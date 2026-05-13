"""BPR congestion forward layer for counterfactual scenarios.

Pure-PyTorch, autograd-friendly. Used **only** in scenario simulation
(D3 spatial / D4 temporal), not in inverse training — the inverse loop
keeps free-flow t_ij to dodge the simultaneity between flow and time.

Conventions
-----------
Congestion is destination-side: each grid j has capacity C_j (jobs / vehicles
the grid can absorb per hour). Inflow_j = sum_i flow_ij. The BPR multiplier
is applied per destination, broadcast over all origins:

    t_ij = t0_ij * (1 + alpha * (inflow_j / C_j)^beta)

This matches the per-grid TomTom proxy used in `OSMnxCarProvider` and the
mesa `_update_congestion` in model.py — keeps the v2.x data plumbing reusable.

A per-OD or per-link version (e.g. routing on the OS Open Roads graph) is
out of scope; left for D7+ if the spatial-holdout work motivates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Pure BPR formula
# ---------------------------------------------------------------------------
def bpr_multiplier(
    inflow_j: torch.Tensor,
    capacity_j: torch.Tensor,
    alpha: float = 0.15,
    beta: float = 4.0,
    eps: float = 1e-6,
    mult_cap: Optional[float] = None,
) -> torch.Tensor:
    """Standard BPR multiplier per destination.

    Parameters
    ----------
    inflow_j   : (..., N) total inflow at each destination per period.
    capacity_j : (N,) destination-side capacity (same units as inflow).
    alpha, beta: BPR coefficients (defaults 0.15, 4.0 — Sheffi 1985).
    mult_cap   : optional upper bound on the multiplier (e.g. 3.0). Stops the
                 quartic from blowing up under heavy oversaturation. v2.x
                 model.py used 2.5 for the same reason.

    Returns
    -------
    mult : (..., N) multiplier in [1, mult_cap or ∞). Multiply free-flow t0_ij
           by ``mult[..., None, j]`` to get congested travel time.
    """
    cap = capacity_j.clamp(min=eps)
    ratio = inflow_j / cap
    mult = 1.0 + alpha * ratio.clamp(min=0.0) ** beta
    if mult_cap is not None:
        mult = mult.clamp(max=mult_cap)
    return mult


def apply_bpr(
    t0_ij: torch.Tensor,
    flow_ij: torch.Tensor,
    capacity_j: torch.Tensor,
    alpha: float = 0.15,
    beta: float = 4.0,
    mult_cap: Optional[float] = None,
) -> torch.Tensor:
    """Apply BPR multiplier to free-flow travel time.

    Parameters
    ----------
    t0_ij   : (..., N, N) free-flow travel time (minutes).
    flow_ij : (..., N, N) flow on each OD pair this period.
    capacity_j : (N,) destination capacity.
    mult_cap   : optional upper bound on the BPR multiplier; recommended for
                 fixed-point loops where high-elasticity choice models can push
                 a few links into the quartic blow-up regime.

    Returns
    -------
    t_ij : (..., N, N) congested travel time.

    Shape rule: leading batch dims (e.g. T hours) are broadcast.
    """
    inflow_j = flow_ij.sum(dim=-2)                              # (..., N)
    mult_j = bpr_multiplier(inflow_j, capacity_j, alpha, beta, mult_cap=mult_cap)
    return t0_ij * mult_j.unsqueeze(-2)                         # broadcast over origin axis


# ---------------------------------------------------------------------------
# Fixed-point user equilibrium via MSA
# ---------------------------------------------------------------------------
@dataclass
class UEResult:
    t_ij: torch.Tensor          # (..., N, N) equilibrium travel time
    flow_ij: torch.Tensor       # (..., N, N) MSA-averaged flow
    n_iter: int                 # iterations used
    final_gap: float            # max |t_new - t_prev| at stop
    converged: bool             # gap < tol within max_iter
    history: list               # per-iter gap (for diagnostics)


def solve_user_equilibrium(
    t0_ij: torch.Tensor,
    capacity_j: torch.Tensor,
    predict_flow: Callable[[torch.Tensor], torch.Tensor],
    *,
    alpha: float = 0.15,
    beta: float = 4.0,
    mult_cap: Optional[float] = None,
    max_iter: int = 30,
    tol: float = 1e-3,
    init_flow: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> UEResult:
    """Method-of-Successive-Averages (MSA) for BPR user equilibrium.

    The choice model is given as a black box ``predict_flow(t_ij) -> flow_ij``,
    e.g. a softmax over (alpha·V + beta·t + gamma·log d) with frozen parameters.
    We iterate

        flow_new = predict_flow(t_k)
        flow_avg = (1 - 1/k) * flow_avg + (1/k) * flow_new            # MSA
        t_{k+1} = apply_bpr(t0, flow_avg, capacity)

    Convergence is measured by the **relative** gap
    ``max|t_{k+1} - t_k| / mean(t_k) < tol`` (unit-agnostic, default 1e-3
    means "max link time changes < 0.1% of mean").

    The MSA step weight ``1/k`` is the standard Sheffi 1985 schedule and gives
    asymptotic convergence under mild monotonicity. No proof of convergence on
    learned (non-monotone) utilities — treat the iteration as best-effort and
    inspect ``history`` for oscillation.

    Notes
    -----
    * Differentiable: every op is autograd-tracked, so backprop through the
      whole loop is possible (gradients via unrolled iteration). For long loops
      consider implicit differentiation (jaxopt / torchopt) — not implemented.
    * The first iterate uses ``init_flow`` if given (e.g. the free-flow
      prediction from a prior call), otherwise a zero flow → uncongested first
      step → forces ``predict_flow`` to be evaluated at t0.
    """
    device = t0_ij.device
    if init_flow is None:
        flow_avg = torch.zeros_like(t0_ij)
    else:
        flow_avg = init_flow

    t_prev = t0_ij
    history = []
    converged = False
    final_gap = float("inf")
    n_iter = 0

    for k in range(1, max_iter + 1):
        t_k = apply_bpr(t0_ij, flow_avg, capacity_j, alpha=alpha, beta=beta, mult_cap=mult_cap)
        flow_new = predict_flow(t_k)
        # MSA step weight 1/k
        weight = 1.0 / k
        flow_avg = (1.0 - weight) * flow_avg + weight * flow_new

        # Relative gap: max abs change divided by mean t. Unit-agnostic.
        denom = t_k.abs().mean().clamp(min=1e-6).item()
        gap = (t_k - t_prev).abs().max().item() / denom
        history.append(gap)
        n_iter = k
        final_gap = gap
        if verbose:
            print(f"[UE] iter={k:02d} rel_gap={gap:.4e}")
        if gap < tol and k > 1:
            converged = True
            t_prev = t_k
            break
        t_prev = t_k

    # one final BPR call so the returned t_ij matches the returned flow_avg
    t_final = apply_bpr(t0_ij, flow_avg, capacity_j, alpha=alpha, beta=beta)
    return UEResult(
        t_ij=t_final,
        flow_ij=flow_avg,
        n_iter=n_iter,
        final_gap=final_gap,
        converged=converged,
        history=history,
    )


__all__ = [
    "bpr_multiplier",
    "apply_bpr",
    "solve_user_equilibrium",
    "UEResult",
]
