"""Multi-metric evaluation for OD flow predictions.

Returns a dict with 7 standard metrics. Use to compare model variants on the
same (predicted, observed) flow tensors.

CPC alone is not enough — it's robust to scale but insensitive to magnitude
distribution. The complementary metrics:

  CPC          : Common Part of Commuters (Lenormand 2012). 0..1, higher better.
                 Robust to row-sum mismatch.
  RMSE         : Root mean squared error on flow counts. Lower better.
  MAE          : Mean absolute error on flow counts. Lower better, less
                 sensitive to outliers than RMSE.
  Pearson r    : Linear correlation between predicted and observed flow vectors.
                 Captures whether high-flow pairs are predicted high.
  Spearman ρ   : Rank correlation. Captures ordering preservation, robust to
                 monotone transforms.
  KL(obs‖pred) : KL divergence per-origin (predicted-distribution fit, in nats).
                 Lower better; complementary to CPC because it penalises P=0
                 where F>0 heavily.
  Top-K acc    : Fraction of (origin, hour) pairs whose top-K observed
                 destinations are predicted in the top-K. K=10 by default.
"""
from __future__ import annotations

from typing import Dict

import torch


def _mask_select(F: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Slice (T, N, N) flow on origin axis by 1-D bool mask of length N."""
    T, N, _ = F.shape
    return F[:, mask]                                   # (T, n_mask, N)


def cpc(pred: torch.Tensor, obs: torch.Tensor) -> float:
    """CPC = 2 Σ min(pred, obs) / (Σ pred + Σ obs)."""
    num = 2.0 * torch.minimum(pred, obs).sum()
    den = (pred.sum() + obs.sum()).clamp(min=1.0)
    return float(num / den)


def rmse(pred: torch.Tensor, obs: torch.Tensor) -> float:
    return float(torch.sqrt(((pred - obs) ** 2).mean()))


def mae(pred: torch.Tensor, obs: torch.Tensor) -> float:
    return float((pred - obs).abs().mean())


def pearson_r(pred: torch.Tensor, obs: torch.Tensor) -> float:
    p = pred.flatten()
    o = obs.flatten()
    pm = p - p.mean()
    om = o - o.mean()
    num = (pm * om).sum()
    den = (pm.norm() * om.norm()).clamp(min=1e-12)
    return float(num / den)


def spearman_rho(pred: torch.Tensor, obs: torch.Tensor) -> float:
    """Spearman ρ via rank-correlation. Uses argsort-of-argsort for ranking."""
    p_rank = pred.flatten().argsort().argsort().float()
    o_rank = obs.flatten().argsort().argsort().float()
    return pearson_r(p_rank, o_rank)


def kl_obs_to_pred(pred: torch.Tensor, obs: torch.Tensor, eps: float = 1e-9) -> float:
    """KL(obs‖pred) per origin, then averaged over origins.

    Treats each origin's destination distribution as a categorical:
        q_o(j | i, t) = obs[t, i, j] / Σ_j obs[t, i, j]
        p_o(j | i, t) = pred[t, i, j] / Σ_j pred[t, i, j]
    KL = Σ_j q log(q / p). Active only on (t, i) with positive observed mass.
    """
    T, N_o, N_d = obs.shape
    obs_row = obs.sum(dim=2, keepdim=True).clamp(min=eps)
    pred_row = pred.sum(dim=2, keepdim=True).clamp(min=eps)
    q = obs / obs_row
    p = pred / pred_row
    # mask cells with q == 0 (KL contribution is 0 there)
    mask_pos = q > 0
    log_qp = torch.log(q.clamp(min=eps) / p.clamp(min=eps))
    kl_per_cell = q * log_qp
    kl_per_cell = torch.where(mask_pos, kl_per_cell, torch.zeros_like(kl_per_cell))
    kl_per_origin = kl_per_cell.sum(dim=2)             # (T, N_o)
    active = (obs.sum(dim=2) > 0)                      # (T, N_o)
    denom = max(int(active.sum().item()), 1)
    return float(kl_per_origin.sum() / denom)


def topk_accuracy(pred: torch.Tensor, obs: torch.Tensor, K: int = 10) -> float:
    """Mean over (t, i) of: |top-K-pred ∩ top-K-obs| / K.

    Skip (t, i) with fewer than K positive observed flows.
    """
    T, N_o, N_d = obs.shape
    K = min(K, N_d)
    obs_topk = obs.topk(K, dim=2).indices                # (T, N_o, K)
    pred_topk = pred.topk(K, dim=2).indices
    # broadcast equal-set check via union sort
    overlap = torch.zeros(T, N_o, dtype=torch.float32, device=pred.device)
    for k in range(K):
        # for each pred-top-k, check if it's in obs-top-K
        match = (pred_topk[..., k:k+1] == obs_topk).any(dim=-1).float()
        overlap += match
    overlap = overlap / K                                 # fraction overlap
    # only count (t, i) where origin has at least K positive flows
    active = (obs > 0).sum(dim=2) >= K
    denom = max(int(active.sum().item()), 1)
    return float(overlap[active].sum() / denom)


def all_metrics(
    pred: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    K_top: int = 10,
) -> Dict[str, float]:
    """Compute all 7 metrics on origins selected by ``mask`` (1-D bool, length N).

    Inputs
    ------
    pred  : (T, N, N) predicted flow counts
    obs   : (T, N, N) observed flow counts
    mask  : (N,) bool — only origins with mask=True are evaluated
    """
    p = _mask_select(pred, mask)        # (T, n_mask, N)
    o = _mask_select(obs, mask)
    return {
        "cpc": cpc(p, o),
        "rmse": rmse(p, o),
        "mae": mae(p, o),
        "pearson_r": pearson_r(p, o),
        "spearman_rho": spearman_rho(p, o),
        "kl_obs_to_pred": kl_obs_to_pred(p, o),
        f"topK_acc_K{K_top}": topk_accuracy(p, o, K=K_top),
    }
