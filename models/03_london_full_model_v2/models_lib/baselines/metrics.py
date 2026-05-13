"""Metrics for OD-flow baseline evaluation.

CPC (common part of commuters, Lenormand 2016), MAE on log(1+flow),
Spearman rank correlation, and per-origin KL divergence.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cpc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Common part of commuters: 2*sum(min(T,P)) / (sum(T)+sum(P))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = y_true.sum() + y_pred.sum()
    if denom <= 0:
        return float("nan")
    return float(2.0 * np.minimum(y_true, y_pred).sum() / denom)


def mae_log1p(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error on log(1+flow)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(np.log1p(y_true) - np.log1p(np.maximum(y_pred, 0.0)))))


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if rho is not None else float("nan")


def per_origin_kl(
    origins: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean KL(P_obs || P_pred) computed per-origin then averaged.

    P_obs(j|i) is the observed outflow distribution from origin i;
    P_pred(j|i) is the predicted distribution. Origins with zero observed
    or zero predicted total flow are skipped.
    """
    origins = np.asarray(origins)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    df = pd.DataFrame({"o": origins, "t": y_true, "p": np.maximum(y_pred, 0.0)})
    kls = []
    for _, g in df.groupby("o", sort=False):
        t_sum = g["t"].sum()
        p_sum = g["p"].sum()
        if t_sum <= 0 or p_sum <= 0:
            continue
        pt = (g["t"].to_numpy() / t_sum) + eps
        pp = (g["p"].to_numpy() / p_sum) + eps
        pt = pt / pt.sum()
        pp = pp / pp.sum()
        kls.append(float(np.sum(pt * np.log(pt / pp))))
    if not kls:
        return float("nan")
    return float(np.mean(kls))


def all_metrics(
    origins: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "CPC": cpc(y_true, y_pred),
        "MAE_log1p": mae_log1p(y_true, y_pred),
        "Spearman": spearman_rho(y_true, y_pred),
        "KL_per_origin": per_origin_kl(origins, y_true, y_pred),
    }
