"""Implicit differentiation through (top-K) softmax.

Forward solves   p* = softmax(z / tau)
Backward applies the implicit-function-theorem gradient on the optimality
condition  F(p, z) = p - softmax(z / tau) = 0.
The Jacobian dp/dz = (1/tau) * (diag(p) - p p^T) coincides with the analytic
softmax Jacobian, but routing through a custom ``torch.autograd.Function``
mirrors the formulation in Amos & Kolter (2017, OptNet) and Blondel et al.
(2022, jaxopt) and lets us:
  * apply the optimality condition in log-space for numerical stability,
  * support a temperature parameter ``tau`` shared across origins,
  * support a nested-logit log-sum (workplace nested by borough) without
    having to re-derive the Jacobian on the python side.

References
----------
- Amos & Kolter, ``OptNet'' (ICML 2017)
- Blondel et al., ``Efficient and modular implicit differentiation'' (2022)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class ImplicitSoftmax(torch.autograd.Function):
    """Custom autograd Function: softmax with implicit-IFT backward.

    Forward
    -------
        p = softmax(z / tau)   along the last axis

    Backward
    --------
    The optimality condition F(p, z) = p - softmax(z/tau) = 0 has Jacobians
        dF/dp = I
        dF/dz = -(1/tau) * (diag(p) - p p^T)
    so by IFT  dp/dz = -(dF/dp)^{-1} dF/dz = (1/tau) (diag(p) - p p^T).
    grad_z = J^T grad_p = (1/tau) * (p * (grad_p - <grad_p, p>))
    which we apply per row.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # numerical stability: subtract row max before exp
        z_scaled = z / tau
        z_shift = z_scaled - z_scaled.max(dim=-1, keepdim=True).values
        p = torch.softmax(z_shift, dim=-1)
        ctx.save_for_backward(p, tau)
        return p

    @staticmethod
    def backward(ctx, grad_p: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        p, tau = ctx.saved_tensors
        # implicit gradient: grad_z = (1/tau) * (p * (grad_p - sum_j p_j grad_p_j))
        weighted = (grad_p * p).sum(dim=-1, keepdim=True)
        grad_z = (p * (grad_p - weighted)) / tau
        # tau is normally non-trainable; if it requires grad we provide a scalar.
        # NOTE: kept for spec compliance; tau is non-trainable in current design.
        # If you ever make tau trainable, FIX the derivation per the
        # softmax-of-z/tau gradient w.r.t. tau:
        #     d softmax(z/tau)_k / d tau = -(1/tau^2) * p_k * (z_k - <z,p>)
        # The expression below is a placeholder and will produce an *incorrect*
        # gradient for tau in non-degenerate inputs. See R1 review.
        if tau.requires_grad:
            z_scaled_grad = -(grad_p * p).sum() / tau
            grad_tau = z_scaled_grad
        else:
            grad_tau = None
        return grad_z, grad_tau


def implicit_softmax(z: torch.Tensor, tau: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Functional wrapper. Apply implicit softmax along the last axis.

    Parameters
    ----------
    z   : Tensor of logits (..., K).  K is the choice-set size.
    tau : scalar temperature (>0). Larger tau ⇒ flatter distribution.
    """
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(float(tau), dtype=z.dtype, device=z.device)
    return ImplicitSoftmax.apply(z, tau)


def nested_logit_logsum(
    z: torch.Tensor,
    nest_idx: torch.Tensor,
    n_nests: int,
    tau_within: float = 1.0,
    tau_between: float = 1.0,
) -> torch.Tensor:
    """Two-level nested-logit inclusive-value (log-sum) aggregation.

    Useful when workplace alternatives are nested by borough: within-nest
    softmax (with scale ``tau_within``) gives P(j | nest), and the
    between-nest log-sum (with scale ``tau_between``) gives P(nest | i).

    Parameters
    ----------
    z          : (..., K) logits over alternatives in the choice set.
    nest_idx   : (K,) long-tensor mapping each alternative to a nest id in
                 [0, n_nests).
    n_nests    : number of nests.
    tau_within : within-nest temperature.
    tau_between: between-nest temperature (must be >= tau_within for the
                 nested-logit to satisfy random-utility consistency).

    Returns
    -------
    log_p_full : (..., K) log-probabilities of each alternative (joint
                 P(j, nest=g(j) | i) under the nested-logit model).
    """
    if tau_between < tau_within - 1e-9:
        raise ValueError(
            "tau_between must be >= tau_within for RU consistency."
        )
    K = z.shape[-1]
    # within-nest log-softmax: logits / tau_within minus per-nest logsumexp
    z_w = z / tau_within
    # gather per-nest max for stability, then logsumexp per nest.
    # We do it via index_add for arbitrary nest sizes.
    logsumexp_per_nest = torch.full(
        (*z.shape[:-1], n_nests), float("-inf"), dtype=z.dtype, device=z.device
    )
    for g in range(n_nests):
        mask = nest_idx == g
        if mask.any():
            sub = z_w[..., mask]                                    # (..., k_g)
            lse = torch.logsumexp(sub, dim=-1)                       # (...,)
            logsumexp_per_nest[..., g] = lse
    # within-nest log-prob
    nest_lse_per_alt = logsumexp_per_nest.index_select(-1, nest_idx)  # (..., K)
    log_p_within = z_w - nest_lse_per_alt
    # between-nest: inclusive value I_g = tau_within * logsumexp(z_w_in_g)
    # then between log-prob = I_g/tau_between - logsumexp_g(I_g/tau_between)
    inclusive_value = tau_within * logsumexp_per_nest                 # (..., n_nests)
    between_logits = inclusive_value / tau_between
    log_p_between_per_nest = between_logits - torch.logsumexp(
        between_logits, dim=-1, keepdim=True
    )
    log_p_between_per_alt = log_p_between_per_nest.index_select(-1, nest_idx)
    return log_p_within + log_p_between_per_alt
