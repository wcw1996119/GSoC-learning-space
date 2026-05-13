"""Extra OD-flow evaluation metrics for v2.2.

All metrics take ``pred`` and ``obs`` torch tensors of shape ``(T, N, N)``
and an optional 1-D ``mask`` of shape ``(N,)`` selecting the origin grids
to evaluate over (e.g. the spatial test holdout).

Conventions
-----------
- ``pred`` are *expected flow counts* on the same scale as ``obs``
  (i.e. P(j|i,t) multiplied by the observed origin total flow).
- "Per pair" metrics (MAE, RMSE, Spearman) operate on the set of
  (t, i, j) triples where ``obs > 0`` and the origin ``i`` is selected
  by the mask. This is the natural support for OD evaluation: zero-cells
  dominate the tensor and would otherwise drown the signal.
- ``kl_per_origin`` is computed per (i, t) origin/hour slice on the
  *normalised* destination distributions, then averaged over slices.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def _origin_mask(mask: Optional[torch.Tensor], N: int, device) -> torch.Tensor:
    if mask is None:
        return torch.ones(N, dtype=torch.bool, device=device)
    m = mask.to(device=device)
    if m.dtype != torch.bool:
        m = m.bool()
    return m


def _flat_nonzero(pred: torch.Tensor, obs: torch.Tensor,
                  mask: Optional[torch.Tensor]):
    """Return masked, observed-positive pred/obs as 1-D tensors."""
    T, N, _ = obs.shape
    m = _origin_mask(mask, N, obs.device)
    p = pred[:, m, :]
    o = obs[:, m, :]
    nz = o > 0
    return p[nz], o[nz]


def mae_per_pair(pred: torch.Tensor, obs: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> float:
    """Mean absolute error over (t, i, j) triples where obs > 0."""
    p, o = _flat_nonzero(pred, obs, mask)
    if p.numel() == 0:
        return float("nan")
    return float((p - o).abs().mean())


def rmse_per_pair(pred: torch.Tensor, obs: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> float:
    """Root mean squared error over (t, i, j) triples where obs > 0."""
    p, o = _flat_nonzero(pred, obs, mask)
    if p.numel() == 0:
        return float("nan")
    return float(torch.sqrt(((p - o) ** 2).mean()))


def spearman_flat(pred: torch.Tensor, obs: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> float:
    """Spearman rank correlation on flattened nonzero OD pairs.

    Uses average-rank tie handling so ties (very common in integer flows)
    do not corrupt the coefficient.
    """
    p, o = _flat_nonzero(pred, obs, mask)
    if p.numel() < 2:
        return float("nan")
    rp = _avg_rank(p)
    ro = _avg_rank(o)
    rp = rp - rp.mean()
    ro = ro - ro.mean()
    denom = torch.sqrt((rp ** 2).sum() * (ro ** 2).sum()) + 1e-12
    return float((rp * ro).sum() / denom)


def _avg_rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (handles ties), 1-indexed."""
    x = x.flatten().double()
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    # initial dense ranks (1..n)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    # average over ties
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # 1-indexed average of [i+1..j+1]
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def kl_per_origin(pred: torch.Tensor, obs: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  eps: float = 1e-10) -> float:
    """Mean of KL(P_pred(j|i,t) || P_obs(j|i,t)) over masked (i, t) slices.

    Each slice is normalised to a probability distribution over j
    (skipping slices where the obs distribution sums to zero).
    """
    T, N, _ = obs.shape
    m = _origin_mask(mask, N, obs.device)
    p = pred[:, m, :]
    o = obs[:, m, :]

    p_sum = p.sum(dim=-1, keepdim=True)
    o_sum = o.sum(dim=-1, keepdim=True)
    valid = (o_sum > 0).squeeze(-1) & (p_sum > 0).squeeze(-1)
    if valid.sum() == 0:
        return float("nan")

    P = p / (p_sum + eps)
    Q = o / (o_sum + eps)
    P = P.clamp_min(eps)
    Q = Q.clamp_min(eps)
    kl = (P * (P.log() - Q.log())).sum(dim=-1)  # (T, |mask|)
    return float(kl[valid].mean())


def cpc(pred: torch.Tensor, obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> float:
    """Common Part of Commuters (Sørensen / Bray–Curtis style)."""
    T, N, _ = obs.shape
    m = _origin_mask(mask, N, obs.device)
    p = pred[:, m, :]
    o = obs[:, m, :]
    return float(torch.minimum(p, o).sum() / (o.sum() + 1e-8))


def all_metrics(pred: torch.Tensor, obs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Bundle MAE, RMSE, Spearman, KL, and CPC into a single dict."""
    return {
        "mae": mae_per_pair(pred, obs, mask),
        "rmse": rmse_per_pair(pred, obs, mask),
        "spearman": spearman_flat(pred, obs, mask),
        "kl": kl_per_origin(pred, obs, mask),
        "cpc": cpc(pred, obs, mask),
    }


__all__ = [
    "mae_per_pair", "rmse_per_pair", "spearman_flat",
    "kl_per_origin", "cpc", "all_metrics",
]
