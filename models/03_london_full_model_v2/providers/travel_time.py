"""TravelTimeProvider — abstract interface + concrete implementations per mode.

Architecture
------------
Each city × mode pair gets its own provider class. The abstract `TravelTimeProvider`
exposes only `get_matrix(hour) -> (N, N) minutes`, so the model is mode-agnostic.

London providers
~~~~~~~~~~~~~~~~
- `OSMnxCarProvider`     — real road network, OSMnx + Dijkstra, TomTom hourly congestion
- `TfLPTProvider`        — GTFS-based transit routing (Stage A2)
- `EuclideanActiveProvider` — walk/cycle, Euclidean × 1.3 / fixed speed

Beijing providers (future, when ethics approved)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `AmapCarProvider`      — 高德 directions API
- `BMTUPTProvider`       — Beijing transit feed
- `EuclideanActiveProvider` (reusable)

Legacy
~~~~~~
- `LondonBPRProvider` (kept for backward compatibility with v2.1-v2.3 demo_cache.npz;
  do NOT use for v2.4+ — it has a 4× speed bias against JTS).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class TravelTimeProvider(ABC):
    """Returns (N, N) travel-time matrix in **minutes** at a given hour."""

    @abstractmethod
    def get_matrix(self, hour: int) -> np.ndarray:
        ...

    def get_freeflow_matrix(self) -> np.ndarray:
        """Return hour-independent free-flow (N, N). Override for mode w/ no congestion."""
        return self.get_matrix(hour=3)  # 3am ≈ free-flow proxy


# ---------------------------------------------------------------------------
# London — Car (Stage A1)
# ---------------------------------------------------------------------------
class OSMnxCarProvider(TravelTimeProvider):
    """Car travel time = OSMnx free-flow × destination-borough hourly TomTom ratio.

    Free-flow is precomputed via `data/scripts/build_car_travel_matrix.py`:
    Greater London drive network from OSM, edge weights = length / posted speed,
    multi-source Dijkstra from each grid centroid.

    Hourly congestion: TomTom-derived ratio (real_time / free_flow), already in
    `grid_hourly_congestion.csv`. We use destination's hour-t ratio as proxy
    for the route congestion (matches LondonBPRProvider's choice for continuity).
    """

    def __init__(
        self,
        freeflow_path: str | Path,
        grid_hourly_congestion_long: "pd.DataFrame",
        grid_id_order: list,
    ):
        import pandas as pd

        path = Path(freeflow_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `data/scripts/build_car_travel_matrix.py` first."
            )
        self.t0 = np.load(path).astype(np.float32)  # (N, N) minutes, free-flow
        self.N = self.t0.shape[0]

        # Build (N, T=24) congestion ratio matrix in canonical grid order
        df = grid_hourly_congestion_long
        # Pivot to wide
        wide = df.pivot(index="grid_id", columns="hour", values="congestion_ratio")
        # Reindex to canonical order
        wide = wide.reindex(index=grid_id_order)
        if wide.isna().any().any():
            n_missing = int(wide.isna().any(axis=1).sum())
            warnings.warn(
                f"OSMnxCarProvider: {n_missing} grids missing congestion → filling 1.20"
            )
            wide = wide.fillna(1.20)
        self.cong_jt = wide.to_numpy().astype(np.float32)  # (N, 24)
        if self.cong_jt.shape[0] != self.N:
            raise ValueError(
                f"Free-flow matrix has N={self.N} grids but congestion has "
                f"{self.cong_jt.shape[0]} — grid_id_order mismatch."
            )

    def get_matrix(self, hour: int) -> np.ndarray:
        """t_ij(hour) = t0[i, j] × cong_j[hour] — destination-side congestion proxy."""
        ratio_j = self.cong_jt[:, hour]      # (N,)
        return self.t0 * ratio_j[None, :]    # (N, N)

    def get_freeflow_matrix(self) -> np.ndarray:
        return self.t0


# ---------------------------------------------------------------------------
# London — Active (walk / cycle, Stage A1 trivial)
# ---------------------------------------------------------------------------
class EuclideanActiveProvider(TravelTimeProvider):
    """Walk/cycle: Euclidean × circuity / fixed speed. No congestion."""

    def __init__(
        self,
        grid_centroids_bng_m: np.ndarray,  # (N, 2)
        speed_kmh: float = 5.0,            # walk default; cycle ~15
        circuity: float = 1.3,
    ):
        self.coords = grid_centroids_bng_m
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        d_m = np.sqrt((diff ** 2).sum(axis=-1))
        d_km = d_m / 1000.0
        self.t0 = (d_km * circuity / (speed_kmh / 60.0)).astype(np.float32)  # minutes

    def get_matrix(self, hour: int) -> np.ndarray:
        return self.t0  # hour-independent

    def get_freeflow_matrix(self) -> np.ndarray:
        return self.t0


# ---------------------------------------------------------------------------
# Legacy (DEPRECATED — kept only for v2.1-v2.3 reproducibility)
# ---------------------------------------------------------------------------
class LondonBPRProvider(TravelTimeProvider):
    """DEPRECATED. Euclidean × 1.3 / 20 km/h × BPR-style multiplier.

    Bias documented in `evaluation_outputs/v22_jts_validation.npz`:
        MAPE vs JTS PT = 78%, vs JTS Car = 60%, mean predicted ≈ 5 min vs JTS 25 min.
    Replaced by `OSMnxCarProvider` for v2.4+.
    """

    def __init__(
        self,
        grid_centroids_bng_m: np.ndarray,
        grid_borough_idx: np.ndarray,
        borough_hourly_congestion: np.ndarray,
        avg_speed_kmh: float = 20.0,
        alpha: float = 0.15,
        bpr_beta: float = 4.0,
        circuity: float = 1.3,
    ):
        warnings.warn(
            "LondonBPRProvider is deprecated; use OSMnxCarProvider for v2.4+",
            DeprecationWarning,
            stacklevel=2,
        )
        self.coords = grid_centroids_bng_m
        self.grid_borough_idx = grid_borough_idx
        self.borough_cong = borough_hourly_congestion
        self.v0_km_per_min = avg_speed_kmh / 60.0
        self.alpha = alpha
        self.beta = bpr_beta
        self.circuity = circuity

        diff = self.coords[:, None, :] - self.coords[None, :, :]
        d_m = np.sqrt((diff ** 2).sum(axis=-1))
        d_km = d_m / 1000.0
        self.t0 = d_km * self.circuity / self.v0_km_per_min

    def get_matrix(self, hour: int) -> np.ndarray:
        cong_j = self.borough_cong[self.grid_borough_idx, hour]
        bpr_factor = 1.0 + self.alpha * (cong_j ** self.beta)
        return self.t0 * bpr_factor[None, :]

    def get_freeflow_matrix(self) -> np.ndarray:
        return self.t0
