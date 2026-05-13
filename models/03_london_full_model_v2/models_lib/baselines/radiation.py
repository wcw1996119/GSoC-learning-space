"""Simini et al. 2012 parameter-free radiation model.

T_ij = T_i * (m_i * n_j) / [ (m_i + s_ij) * (m_i + n_j + s_ij) ]

where m_i = origin "mass" (population), n_j = destination "mass"
(employment), s_ij = total destination mass within distance d_ij of i,
*excluding* both i and j. T_i is total outflow from i.

This implementation works on a precomputed distance matrix. Distances are
computed from grid centroids using a haversine approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class RadiationModel:
    def __init__(self) -> None:
        self.s_matrix_: np.ndarray | None = None
        self.grid_index_: dict[str, int] | None = None

    def fit(
        self,
        grid_features: pd.DataFrame,
        m_col: str = "population",
        n_col: str = "total_employment",
    ) -> "RadiationModel":
        """Precompute the s_ij intervening-opportunities matrix."""
        gf = grid_features.reset_index(drop=True)
        N = len(gf)
        self.grid_index_ = {g: i for i, g in enumerate(gf["grid_id"])}
        lat = gf["centroid_lat"].to_numpy()
        lon = gf["centroid_lon"].to_numpy()
        n = gf[n_col].to_numpy(dtype=float)
        self.m_ = gf[m_col].to_numpy(dtype=float)
        self.n_ = n

        # Pairwise distance (N,N).
        d = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        self.d_ = d

        # s_ij = sum of n_k for k != i, k != j, d_ik < d_ij.
        # Vectorised: rank destinations within each origin row by distance,
        # cumulative sum of n along that order, then subtract n_j and n_i.
        order = np.argsort(d, axis=1)  # (N,N) indices sorted by dist asc
        n_sorted = n[order]            # (N,N)
        cum = np.cumsum(n_sorted, axis=1)
        # s for the k-th nearest dest = cum[k] - n_sorted[k]  (excludes j)
        s_in_order = cum - n_sorted
        # Place back into (i,j) layout: invert the order permutation.
        inv_order = np.argsort(order, axis=1)
        s = np.take_along_axis(s_in_order, inv_order, axis=1)
        # Exclude origin i itself from s_ij (subtract n_i if i was within the radius).
        # When d_ii=0, i is always the first in the sort, so its n_i is in the
        # subtracted-out slot already. To be safe, subtract n_i where d_ii<=d_ij,
        # which is always true; so subtract n_i from every s[i,j].
        s = s - n[:, None]
        s = np.maximum(s, 0.0)
        np.fill_diagonal(s, 0.0)
        self.s_matrix_ = s
        return self

    def predict(
        self,
        origins: np.ndarray,
        dests: np.ndarray,
        T_i: np.ndarray,
    ) -> np.ndarray:
        """Predict flows for given OD-pair labels.

        Parameters
        ----------
        origins, dests : 1-D arrays of grid_id strings.
        T_i            : 1-D array of total outflow from each origin (same
                         length as `origins`); typically supplied as the
                         observed origin total from the training data.
        """
        if self.s_matrix_ is None or self.grid_index_ is None:
            raise RuntimeError("RadiationModel.fit must be called first")
        gi = self.grid_index_
        oi = np.array([gi[o] for o in origins])
        dj = np.array([gi[d] for d in dests])
        m = self.m_[oi]
        n = self.n_[dj]
        s = self.s_matrix_[oi, dj]

        denom = (m + s) * (m + n + s)
        denom = np.where(denom <= 0, np.nan, denom)
        prob = (m * n) / denom
        prob = np.nan_to_num(prob, nan=0.0)

        # Normalise per origin so that probs sum to 1, then scale by T_i.
        df = pd.DataFrame({"o": origins, "p": prob, "T": T_i})
        sums = df.groupby("o")["p"].transform("sum").replace(0, np.nan)
        df["p"] = (df["p"] / sums).fillna(0.0)
        return (df["p"] * df["T"]).to_numpy()
