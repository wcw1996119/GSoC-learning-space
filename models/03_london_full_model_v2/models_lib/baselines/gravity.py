"""Wilson 1971 doubly-constrained gravity model via Poisson regression.

We use a Poisson GLM with log link and offsets log(O_i)+log(D_j); the only
free parameter is the deterrence on log(t_ij+1). Poisson is preferred over
OLS-on-log because OD flows are non-negative integer counts with many
zeros, and a Poisson GLM is exactly equivalent to maximising the
multinomial likelihood used in classical Wilson entropy maximisation
(Flowerdew & Aitkin 1982). OLS-on-log would drop zeros and bias β toward
zero.

After fitting, predicted means are scaled by per-origin and per-destination
balancing factors (one round of IPF) so that the predictions satisfy the
doubly-constrained marginals on the training set, matching the spirit of
Wilson 1971.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


class GravityModel:
    def __init__(self, ipf_iters: int = 30) -> None:
        self.beta_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.ipf_iters = ipf_iters
        self._result = None

    def fit(
        self,
        O_i: np.ndarray,
        D_j: np.ndarray,
        t_ij: np.ndarray,
        observed_flow: np.ndarray,
    ) -> "GravityModel":
        """Fit Poisson GLM. All inputs are 1D arrays of equal length (one
        per OD pair). O_i is origin total (e.g. population), D_j the
        destination total (e.g. employment), t_ij the travel time."""
        O_i = np.asarray(O_i, dtype=float)
        D_j = np.asarray(D_j, dtype=float)
        t_ij = np.asarray(t_ij, dtype=float)
        y = np.asarray(observed_flow, dtype=float)

        # Guard against zeros in offsets.
        offset = np.log(np.maximum(O_i, 1e-6)) + np.log(np.maximum(D_j, 1e-6))
        X = np.column_stack(
            [np.ones_like(t_ij), np.log(t_ij + 1.0)]
        )

        model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        try:
            self._result = model.fit(maxiter=100)
            self.intercept_ = float(self._result.params[0])
            self.beta_ = float(self._result.params[1])
        except Exception:
            # Fallback: log-OLS on positive flows.
            mask = y > 0
            if mask.sum() < 5:
                self.intercept_, self.beta_ = 0.0, -1.0
            else:
                A = X[mask]
                b = np.log(y[mask]) - offset[mask]
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                self.intercept_ = float(coef[0])
                self.beta_ = float(coef[1])
        return self

    def _raw_mean(
        self, O_i: np.ndarray, D_j: np.ndarray, t_ij: np.ndarray
    ) -> np.ndarray:
        offset = np.log(np.maximum(O_i, 1e-6)) + np.log(np.maximum(D_j, 1e-6))
        eta = self.intercept_ + self.beta_ * np.log(t_ij + 1.0) + offset
        return np.exp(eta)

    def predict(
        self,
        O_i: np.ndarray,
        D_j: np.ndarray,
        t_ij: np.ndarray,
        origins: Optional[np.ndarray] = None,
        dests: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict OD flows. If origins/dests labels are passed we apply
        IPF balancing so that row/col marginals match O_i/D_j."""
        mu = self._raw_mean(np.asarray(O_i, float), np.asarray(D_j, float),
                             np.asarray(t_ij, float))
        if origins is None or dests is None:
            return mu

        df = pd.DataFrame({"o": origins, "d": dests, "mu": mu,
                           "O": O_i, "D": D_j})
        # Target marginals per unique origin/dest.
        target_O = df.groupby("o")["O"].first()
        target_D = df.groupby("d")["D"].first()
        for _ in range(self.ipf_iters):
            cur_O = df.groupby("o")["mu"].sum()
            scale_o = (target_O / cur_O.replace(0, np.nan)).fillna(1.0)
            df["mu"] = df["mu"] * df["o"].map(scale_o).to_numpy()
            cur_D = df.groupby("d")["mu"].sum()
            scale_d = (target_D / cur_D.replace(0, np.nan)).fillna(1.0)
            df["mu"] = df["mu"] * df["d"].map(scale_d).to_numpy()
        return df["mu"].to_numpy()
