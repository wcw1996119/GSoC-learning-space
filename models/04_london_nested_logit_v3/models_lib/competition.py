"""Shen 1998 competition effect — effective demand D_j at each destination.

D_j = Σ_i n_workers_i · f(d_ij)

where:
    n_workers_i = workforce living at origin i (proxy: total outflow F_i)
    f(d)        = impedance / distance-decay kernel
    D_j         = effective number of workers competing for jobs at j

Two standard kernel choices:
    power-law:    f(d) = d^β        (Cervero 1999 uses β = -0.35)
    exponential:  f(d) = exp(β·d)   (alternative; sharper near-field decay)

Output `log(D_j)` is used as a destination-level utility feature with
negative expected coefficient ν (more competition → lower utility).

Anchor: Shen Q (1998) "Location characteristics of inner-city neighborhoods
and employment accessibility of low-wage workers" E&P B 25:345-365.

Merlin LA, Hu L (2017) extended this to LA — confirms competition
materially shifts conclusions about who is disadvantaged.

Beijing portability: same formula; workers from 七普 居住地从业人员 by
grid (or borough-level + spatial allocation), distance from Amap.
"""
from __future__ import annotations

import numpy as np


def shen_effective_demand(n_workers: np.ndarray,
                          dist_km: np.ndarray,
                          beta_decay: float = -0.35,
                          kernel: str = "power") -> np.ndarray:
    """Compute Shen-style effective demand D_j at each destination.

    Args:
      n_workers:  (N,)   workforce at each origin
      dist_km:    (N, N) distance i→j in km (or any consistent unit)
      beta_decay: decay parameter. Cervero 1999 uses -0.35 for log_d (power law)
                  or -0.05 to -0.1 for raw km (exponential)
      kernel:     "power" → f = (d+1)^β   (avoids log(0) for i=j by adding 1)
                  "exp"   → f = exp(β·d)

    Returns:
      D_j: (N,) effective demand per destination
    """
    if kernel == "power":
        decay = np.power(dist_km + 1.0, beta_decay)
    elif kernel == "exp":
        decay = np.exp(beta_decay * dist_km)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    # D_j = sum over i of workers_i × decay(i, j)
    return n_workers @ decay  # (N,) = (N,) @ (N, N)


def log_effective_demand(n_workers: np.ndarray,
                         dist_km: np.ndarray,
                         beta_decay: float = -0.35,
                         kernel: str = "power",
                         eps: float = 1.0) -> np.ndarray:
    """Convenience: log(D_j + eps) as destination feature."""
    D_j = shen_effective_demand(n_workers, dist_km, beta_decay, kernel)
    return np.log(D_j + eps)
