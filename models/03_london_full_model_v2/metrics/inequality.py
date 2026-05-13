"""Inequality metrics for accessibility / income / commute distributions.

Ported and generalised from v1 (`02_london_commuting_model/model.py`).

All functions take a 1-D array of NON-NEGATIVE values (e.g. accessibility per grid).
Negative values, NaNs, and zeros are filtered out before the computation, since the
Gini coefficient and Palma ratio are only defined for non-negative magnitudes.

Functions
---------
gini(values)         -> float : standard Gini coefficient in [0, 1]
palma(values)        -> float : top-10% mean / bottom-40% mean
lorenz_curve(values) -> tuple : (cum_pop_share, cum_value_share) for plotting
"""
from __future__ import annotations

import numpy as np


def _clean(values: np.ndarray) -> np.ndarray:
    """Filter to finite, strictly-positive values, returned sorted ascending."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    arr.sort()
    return arr


def gini(values: np.ndarray) -> float:
    """Standard Gini coefficient on non-negative values.

    Uses the discrete formula

        G = (2 Σ i·x_i) / (n Σ x_i)  −  (n + 1) / n

    on values sorted ascending (i = 1, ..., n). Returns 0.0 for empty input.
    """
    arr = _clean(values)
    n = arr.size
    if n == 0:
        return 0.0
    total = arr.sum()
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * arr)) / (n * total) - (n + 1.0) / n)


def palma(values: np.ndarray) -> float:
    """Palma ratio = mean(top 10%) / mean(bottom 40%).

    Higher = more unequal at the tails. Returns 0.0 if either tail is empty
    (e.g., n < 10) or the bottom 40% mean is 0.
    """
    arr = _clean(values)
    n = arr.size
    if n == 0:
        return 0.0
    bottom = arr[: int(n * 0.4)]
    top = arr[int(n * 0.9) :]
    if bottom.size == 0 or top.size == 0:
        return 0.0
    bot_mean = float(bottom.mean())
    if bot_mean == 0:
        return 0.0
    return float(top.mean() / bot_mean)


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve points for plotting.

    Returns
    -------
    cum_pop_share   : np.ndarray of shape (n+1,)
        Cumulative population share, starting at 0.0 and ending at 1.0.
    cum_value_share : np.ndarray of shape (n+1,)
        Cumulative value share, starting at 0.0 and ending at 1.0.

    The 45° line y = x is the perfect-equality reference. The Gini coefficient
    equals twice the area between this line and the curve.
    """
    arr = _clean(values)
    n = arr.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    total = arr.sum()
    if total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum_value = np.cumsum(arr) / total
    cum_pop = np.arange(1, n + 1, dtype=float) / n
    cum_value = np.concatenate(([0.0], cum_value))
    cum_pop = np.concatenate(([0.0], cum_pop))
    return cum_pop, cum_value
