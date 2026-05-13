"""Inequality indices over a per-grid quantity (typically accessibility A_i).

Three indices, all weighted by population w_i (or unit weights):

  Gini       — 0 (egalitarian) to 1 (one person has everything).
                Formula: Σ_i Σ_j w_i w_j |x_i - x_j| / (2 (Σw)^2 mean(x))
  Palma      — top 10% / bottom 40% of population-weighted x. >2 ≈ unequal.
  Atkinson   — 1 - (Σ w_i (x_i / x̄)^(1-ε))^(1/(1-ε)) for ε != 1.
                ε = 0.5 default (mild aversion); ε = 1 uses log form.

All accept either numpy or torch arrays; numpy is used internally for
percentile / sort because torch's quantile semantics under weighting are
restrictive.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_numpy(a: ArrayLike) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def gini(x: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    """Population-weighted Gini coefficient. Uses the absolute-difference form
    rather than the sorted-cumulative form — slower but unambiguous under
    fractional weights and zeros."""
    x = _as_numpy(x).astype(np.float64)
    n = len(x)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _as_numpy(w).astype(np.float64)
    W = w.sum()
    if W <= 0:
        return float("nan")
    mean_x = (w * x).sum() / W
    if mean_x <= 0:
        return float("nan")
    # Σ_i Σ_j w_i w_j |x_i - x_j|
    diff = np.abs(x[:, None] - x[None, :])               # (N, N)
    weighted_diff = (w[:, None] * w[None, :] * diff).sum()
    return float(weighted_diff / (2.0 * (W ** 2) * mean_x))


def palma_ratio(x: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    """Palma = (sum of x in top 10% of population) / (sum of x in bottom 40%).
    Robust to outliers; >2 considered unequal in income literature."""
    x = _as_numpy(x).astype(np.float64)
    n = len(x)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _as_numpy(w).astype(np.float64)
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    W = cum_w[-1]
    # Bottom 40% mass cutoff
    cutoff_bot = 0.40 * W
    bot_idx = cum_w <= cutoff_bot
    bot_sum = (w_sorted[bot_idx] * x_sorted[bot_idx]).sum()
    # Top 10% mass cutoff (population from the top)
    cutoff_top = 0.90 * W
    top_idx = cum_w >= cutoff_top
    top_sum = (w_sorted[top_idx] * x_sorted[top_idx]).sum()
    if bot_sum <= 0:
        return float("inf")
    return float(top_sum / bot_sum)


def atkinson(x: ArrayLike, w: Optional[ArrayLike] = None, eps: float = 0.5) -> float:
    """Atkinson index, ε > 0 (inequality aversion). 0 (egalitarian) to 1.
    eps=0.5 → mild aversion; eps=1 → log form (geometric mean ratio)."""
    x = _as_numpy(x).astype(np.float64)
    n = len(x)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _as_numpy(w).astype(np.float64)
    W = w.sum()
    mean_x = (w * x).sum() / W
    if mean_x <= 0:
        return float("nan")
    pos = x > 0
    if not pos.all():
        # Atkinson undefined with zero or negative entries; drop them and
        # rescale weights. Caveat: this changes the universe.
        x = x[pos]; w = w[pos]; W = w.sum()
        mean_x = (w * x).sum() / W
    if abs(eps - 1.0) < 1e-9:
        # Log / geometric form
        log_mean = (w * np.log(x)).sum() / W
        ede = np.exp(log_mean)
    else:
        p = 1.0 - eps
        moment = (w * (x ** p)).sum() / W
        ede = moment ** (1.0 / p)
    return float(1.0 - ede / mean_x)


def all_indices(x: ArrayLike, w: Optional[ArrayLike] = None,
                atkinson_eps: float = 0.5) -> dict:
    return {
        "mean": float(np.average(_as_numpy(x), weights=_as_numpy(w) if w is not None else None)),
        "gini": gini(x, w),
        "palma": palma_ratio(x, w),
        "atkinson": atkinson(x, w, atkinson_eps),
    }


__all__ = ["gini", "palma_ratio", "atkinson", "all_indices"]
