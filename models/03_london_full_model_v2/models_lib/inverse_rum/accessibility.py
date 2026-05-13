"""Hansen-style job accessibility A_i = Σ_j E_j · exp(β_t · t_ij).

Convention: pass the *signed* recovered β_t (negative). Then
``exp(β_t · t)`` decays in t as expected.

A_i = Σ_j E_j · exp(β_t · t_ij)

Companion to D5: the same call signature is reused for baseline / scenario /
post-BPR comparisons in scenario_A_accessibility.py.
"""
from __future__ import annotations

from typing import Optional

import torch


def hansen_accessibility(
    E_j: torch.Tensor,
    t_ij: torch.Tensor,
    beta_t: float,
    self_loop: bool = False,
) -> torch.Tensor:
    """Compute (N,) Hansen accessibility from an OD travel-time matrix.

    Parameters
    ----------
    E_j     : (N,) destination employment (raw counts; not z-scored).
    t_ij    : (N, N) or (T, N, N) travel time. If 3D, averaged over the
              leading axis (assumes daily aggregation).
    beta_t  : signed coefficient on t in the utility (negative for normal
              behaviour). Pass e.g. recovered ``trainer.rum.beta_t.mean()``.
    self_loop : if False (default), zero out the diagonal so a grid does not
                count its own employment.

    Returns
    -------
    A_i : (N,) accessible jobs from each origin.
    """
    if t_ij.dim() == 3:
        t_ij = t_ij.mean(0)
    decay = torch.exp(beta_t * t_ij)                      # (N, N)
    if not self_loop:
        decay = decay.clone()
        decay.fill_diagonal_(0.0)
    A = decay @ E_j                                        # (N, N) @ (N,) -> (N,)
    return A


__all__ = ["hansen_accessibility"]
