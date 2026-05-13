"""Spatial autocorrelation metrics for grid-level variables.

Lightweight numpy / scipy.sparse implementation — avoids libpysal which has
heavy install footprint and version quirks.

Functions
---------
build_kNN_weight(coords, k=8)
    Build a row-standardised kNN spatial weight matrix.
morans_I(values, W)
    Global Moran's I + permutation p-value.
lisa(values, W)
    Local Moran's I (one value per location).

References
----------
- Anselin, L. (1995). Local indicators of spatial association — LISA.
  Geographical Analysis, 27(2), 93-115.
- Moran, P.A.P. (1950). Notes on continuous stochastic phenomena.
  Biometrika, 37(1/2), 17-23.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Weight matrix construction
# ---------------------------------------------------------------------------

def build_kNN_weight(coords: np.ndarray, k: int = 8) -> sp.csr_matrix:
    """Row-standardised kNN spatial weight matrix.

    Parameters
    ----------
    coords : (N, 2) ndarray
        Projected planar coordinates (e.g. BNG / EPSG:27700).
    k : int
        Number of neighbours per location (excluding self).

    Returns
    -------
    W : (N, N) scipy.sparse.csr_matrix
        Row-standardised so that W.sum(axis=1) == 1 for every row.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    tree = cKDTree(coords)
    # +1 because cKDTree returns the point itself as the first neighbour
    _, idx = tree.query(coords, k=k + 1)
    # Drop the self-index (first column, distance 0)
    nbr = idx[:, 1:]  # (N, k)
    rows = np.repeat(np.arange(n), k)
    cols = nbr.reshape(-1)
    data = np.full(rows.shape, 1.0 / k, dtype=np.float64)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return W


# ---------------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------------

def _moran_I_from_z(z: np.ndarray, W: sp.csr_matrix) -> float:
    """Compute Moran's I given centred values z = x - mean(x).

    With row-standardised W, S0 = sum_ij w_ij = N, so
        I = (z' W z) / (z' z)
    """
    Wz = W @ z
    num = float(z @ Wz)
    den = float(z @ z)
    if den <= 0:
        return 0.0
    return num / den


def morans_I(
    values: np.ndarray,
    weight_matrix: sp.csr_matrix,
    n_permutations: int = 999,
    seed: int | None = 42,
) -> Tuple[float, float]:
    """Global Moran's I + permutation p-value.

    Parameters
    ----------
    values : (N,) ndarray
        The variable.
    weight_matrix : (N, N) sparse, row-standardised
        Spatial weights.
    n_permutations : int
        Number of random permutations for the reference distribution.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    I : float
        Moran's I statistic.
    p_value : float
        Two-sided pseudo p-value
        = (1 + #{|I_perm| >= |I_obs|}) / (1 + n_permutations).
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    z = x - x.mean()
    I_obs = _moran_I_from_z(z, weight_matrix)

    rng = np.random.default_rng(seed)
    n = z.shape[0]
    z_perm = z.copy()
    perm_I = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        rng.shuffle(z_perm)
        perm_I[i] = _moran_I_from_z(z_perm, weight_matrix)

    # Two-sided test: how often does the permuted |I| match or exceed |I_obs|?
    extreme = int(np.sum(np.abs(perm_I) >= abs(I_obs)))
    p_value = (1 + extreme) / (1 + n_permutations)
    return float(I_obs), float(p_value)


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------

def lisa(values: np.ndarray, weight_matrix: sp.csr_matrix) -> np.ndarray:
    """Local Moran's I per grid.

    Defined as
        I_i = z_i * sum_j w_ij * z_j / m2,
    where m2 = sum_i z_i^2 / N is the variance estimator.

    Parameters
    ----------
    values : (N,) ndarray
    weight_matrix : (N, N) sparse, row-standardised

    Returns
    -------
    I_local : (N,) ndarray
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    z = x - x.mean()
    n = z.shape[0]
    m2 = (z * z).sum() / n
    if m2 <= 0:
        return np.zeros(n, dtype=np.float64)
    Wz = weight_matrix @ z
    return z * Wz / m2


def lisa_with_pvalues(
    values: np.ndarray,
    weight_matrix: sp.csr_matrix,
    n_permutations: int = 999,
    seed: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """LISA + per-location pseudo p-value via conditional permutation.

    Conditional permutation: for each location i, fix z_i and randomly
    permute the remaining z values, then recompute the local statistic.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    z = x - x.mean()
    n = z.shape[0]
    m2 = (z * z).sum() / n
    if m2 <= 0:
        return np.zeros(n), np.ones(n)

    Wz = weight_matrix @ z
    I_obs = z * Wz / m2

    rng = np.random.default_rng(seed)
    # Convert to LIL/CSR pattern: for each i, neighbours and weights
    Wcsr = weight_matrix.tocsr()
    p = np.empty(n, dtype=np.float64)
    extreme = np.zeros(n, dtype=np.int64)

    # Pre-extract per-row indices/data
    indptr = Wcsr.indptr
    indices = Wcsr.indices
    data = Wcsr.data

    for i in range(n):
        nbr_w = data[indptr[i]:indptr[i + 1]]
        n_nbr = nbr_w.shape[0]
        if n_nbr == 0:
            p[i] = 1.0
            continue
        # Sample neighbours from population excluding i
        pool = np.delete(np.arange(n), i)
        zi = z[i]
        # Vectorised: draw n_permutations samples of n_nbr neighbours
        # without-replacement per draw is most defensible; use rng.permuted
        for k in range(n_permutations):
            sel = rng.choice(pool, size=n_nbr, replace=False)
            wi_zi = float(zi * (nbr_w * z[sel]).sum() / m2)
            if abs(wi_zi) >= abs(I_obs[i]):
                extreme[i] += 1
        p[i] = (1 + extreme[i]) / (1 + n_permutations)

    return I_obs, p


def lisa_classify(
    values: np.ndarray,
    weight_matrix: sp.csr_matrix,
    p_local: np.ndarray | None = None,
    p_threshold: float = 0.05,
) -> np.ndarray:
    """Classify each location by LISA quadrant.

    Returns
    -------
    labels : (N,) int array
        0 = not significant
        1 = high-high (hot spot)
        2 = low-low  (cold spot)
        3 = high-low (anomaly: high surrounded by low)
        4 = low-high (anomaly: low surrounded by high)
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    z = x - x.mean()
    Wz = weight_matrix @ z
    n = z.shape[0]
    labels = np.zeros(n, dtype=np.int64)

    sig = (
        np.ones(n, dtype=bool)
        if p_local is None
        else (p_local < p_threshold)
    )
    hi = z > 0
    lo = z < 0
    hi_lag = Wz > 0
    lo_lag = Wz < 0

    labels[sig & hi & hi_lag] = 1  # HH
    labels[sig & lo & lo_lag] = 2  # LL
    labels[sig & hi & lo_lag] = 3  # HL
    labels[sig & lo & hi_lag] = 4  # LH
    return labels
