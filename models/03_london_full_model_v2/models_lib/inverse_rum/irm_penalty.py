"""IRM penalties for OOD generalization (Paper A Module 1).

Two variants supported via ``variant`` flag:
  * "rex"   — V-REx (Krueger et al. 2021, arxiv 2003.00688): variance of
              per-environment risks. Simple, stable, recommended.
  * "irmv1" — IRM-v1 (Arjovsky et al. 2019, arxiv 1907.02893): gradient
              penalty on a dummy classifier scale. Theoretical original,
              optimization-fragile.

Environment partition: 33 London boroughs (origin grid_borough_idx).
Each origin grid cell i belongs to one env e = grid_borough_idx[i].

Inputs are per-origin NLLs (shape (N,)) — produced by the trainer with
``_per_origin_nll`` (sum over t, j; before averaging).
"""
from __future__ import annotations

import torch


def compute_rex_penalty(
    per_origin_nll: torch.Tensor,
    origin_env_idx: torch.Tensor,
    origin_active_mask: torch.Tensor,
    n_envs: int,
) -> torch.Tensor:
    """V-REx variance penalty.

    Parameters
    ----------
    per_origin_nll : (N,) tensor
        NLL summed over (t, j) for each origin i.
    origin_env_idx : (N,) long tensor
        Environment index for each origin (0..n_envs-1).
    origin_active_mask : (N,) bool tensor
        Which origins are in train set (used to skip masked-out origins
        whose NLL is 0 by construction).
    n_envs : int
        Total number of environments.

    Returns
    -------
    penalty : 0-d tensor — `Var_e [ mean_{i in env e} NLL_i ]`.
    """
    env_means = []
    for e in range(n_envs):
        sel = (origin_env_idx == e) & origin_active_mask
        if sel.sum() == 0:
            continue
        env_means.append(per_origin_nll[sel].mean())
    if len(env_means) < 2:
        return per_origin_nll.new_zeros(())
    stacked = torch.stack(env_means)                # (n_envs_active,)
    return stacked.var(unbiased=False)


def compute_irmv1_penalty(
    per_origin_nll: torch.Tensor,
    origin_env_idx: torch.Tensor,
    origin_active_mask: torch.Tensor,
    dummy_scale: torch.Tensor,
    n_envs: int,
) -> torch.Tensor:
    """IRM-v1 gradient penalty (Arjovsky 2019).

    For each env e, computes (d R_e / d w)^2 at w=1, where ``per_origin_nll``
    was produced with logits scaled by ``dummy_scale``. Sums across envs and
    averages.

    Caller responsibility: ``per_origin_nll`` must depend on ``dummy_scale``
    (multiply utility / logits by dummy_scale before softmax in trainer).

    Parameters
    ----------
    per_origin_nll : (N,) tensor — depending on dummy_scale via autograd graph.
    origin_env_idx : (N,) long tensor.
    origin_active_mask : (N,) bool tensor.
    dummy_scale : 0-d tensor with requires_grad=True (held at 1.0).
    n_envs : int.

    Returns
    -------
    penalty : 0-d tensor — `mean_e (d R_e / d w)^2`.
    """
    if not dummy_scale.requires_grad:
        raise ValueError(
            "dummy_scale must have requires_grad=True for IRM-v1 gradient penalty"
        )
    grads_sq = []
    for e in range(n_envs):
        sel = (origin_env_idx == e) & origin_active_mask
        if sel.sum() == 0:
            continue
        env_risk = per_origin_nll[sel].mean()
        grad = torch.autograd.grad(
            env_risk, dummy_scale, create_graph=True, retain_graph=True,
        )[0]
        grads_sq.append(grad ** 2)
    if len(grads_sq) < 2:
        return per_origin_nll.new_zeros(())
    return torch.stack(grads_sq).mean()


def warmup_lambda(epoch: int, warmup_epochs: int, target_lambda: float) -> float:
    """Step warmup: λ=0 for first ``warmup_epochs`` epochs, then ``target_lambda``.

    Per Krueger 2021: lets the model first learn a sensible ERM representation
    before kicking in the invariance penalty. Otherwise the penalty dominates
    early training and the model never escapes a degenerate basin.
    """
    return 0.0 if epoch < warmup_epochs else target_lambda
