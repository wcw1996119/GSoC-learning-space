"""NodeFeatureProvider — abstract interface + London implementation.

Per methodology/03_node_features.md: 21 static + 4 temporal = 25 dims per (grid, hour).

Static features (21):
- sec1..sec8 (8 sectors employment)
- total_employment (1)
- population (1)
- poi_total (1)
- poi_<8 categories> (8)
- subway_station_count (1)
- centroid_lat (1)
- centroid_lon (1)

Temporal features (4):
- congestion_ratio (1)
- hour_sin (1)
- hour_cos (1)
- workplace_population_estimate (1) — synthesized from OD × NTS departure-time

All features are z-score normalized using training set stats.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd

V2_ROOT = Path(__file__).resolve().parent.parent

# Static feature column order (canonical)
STATIC_COLS = [
    "sec1_primary", "sec2_manufacturing", "sec3_construction", "sec4_retail",
    "sec5_fnb", "sec6_info_finance", "sec7_public", "sec8_other",
    "total_employment",
    "population",
    "poi_total",
    "poi_commercial", "poi_education", "poi_fnb", "poi_healthcare",
    "poi_office", "poi_public", "poi_retail", "poi_transport",
    "subway_station_count",
    "centroid_lat", "centroid_lon",
]
LOG_TRANSFORM_COLS = {
    "sec1_primary", "sec2_manufacturing", "sec3_construction", "sec4_retail",
    "sec5_fnb", "sec6_info_finance", "sec7_public", "sec8_other",
    "total_employment", "population", "poi_total",
    "poi_commercial", "poi_education", "poi_fnb", "poi_healthcare",
    "poi_office", "poi_public", "poi_retail", "poi_transport",
}

N_STATIC = len(STATIC_COLS)
N_TEMPORAL = 4  # congestion, hour_sin, hour_cos, workplace_pop


class NodeFeatureProvider(ABC):
    @abstractmethod
    def n_grids(self) -> int: ...

    @abstractmethod
    def grid_ids(self) -> list: ...

    @abstractmethod
    def get_static(self) -> np.ndarray:
        """(N, F_static) — all grids, static features, normalized."""
        ...

    @abstractmethod
    def get_temporal(self, hour: int) -> np.ndarray:
        """(N, F_temporal) — all grids, hourly features, normalized."""
        ...


class LondonFeatureProvider(NodeFeatureProvider):
    def __init__(self, grid_features_path=None, congestion_path=None,
                 train_mask: np.ndarray | None = None):
        if grid_features_path is None:
            grid_features_path = V2_ROOT / "data" / "processed" / "grid_static_features.csv"
        if congestion_path is None:
            congestion_path = V2_ROOT / "data" / "processed" / "grid_hourly_congestion.csv"

        self.grid = pd.read_csv(grid_features_path).reset_index(drop=True)
        self._grid_ids = self.grid["grid_id"].tolist()
        self._n = len(self.grid)

        cong = pd.read_csv(congestion_path)
        # Pivot to (N, 24) array aligned to self.grid order
        cong_pivot = cong.pivot(index="grid_id", columns="hour", values="congestion_ratio")
        cong_pivot = cong_pivot.reindex(self._grid_ids)
        self.congestion_matrix = cong_pivot.values  # (N, 24)

        # Build raw static feature matrix
        raw = self.grid[STATIC_COLS].values.astype(np.float64)  # (N, F_static)
        # Apply log1p to skewed cols
        log_idx = [i for i, c in enumerate(STATIC_COLS) if c in LOG_TRANSFORM_COLS]
        raw[:, log_idx] = np.log1p(raw[:, log_idx])

        # Normalization stats (use train_mask if provided, else use all grids)
        mask = train_mask if train_mask is not None else np.ones(self._n, dtype=bool)
        self.static_mean = raw[mask].mean(axis=0)
        self.static_std = raw[mask].std(axis=0) + 1e-8

        self.static_norm = (raw - self.static_mean) / self.static_std

        # Congestion stats
        cong_train = self.congestion_matrix[mask]
        self.cong_mean = float(np.nanmean(cong_train))
        self.cong_std = float(np.nanstd(cong_train) + 1e-8)
        # Replace NaN with mean (some borough may lack data; fill with 1.2 default already)
        cong_filled = np.where(np.isnan(self.congestion_matrix), self.cong_mean, self.congestion_matrix)
        self.cong_norm = (cong_filled - self.cong_mean) / self.cong_std

        # Workplace population estimate placeholder (zeros for now — synthesize later)
        # In v2.1: load synthesized workplace_pop[grid][hour] from data/processed/
        self.workplace_pop_norm = np.zeros((self._n, 24), dtype=np.float32)

    def n_grids(self) -> int:
        return self._n

    def grid_ids(self) -> list:
        return self._grid_ids

    def get_static(self) -> np.ndarray:
        return self.static_norm.astype(np.float32)  # (N, F_static)

    def get_temporal(self, hour: int) -> np.ndarray:
        cong = self.cong_norm[:, hour]  # (N,)
        wpop = self.workplace_pop_norm[:, hour]  # (N,)
        sin = np.full(self._n, np.sin(2 * np.pi * hour / 24), dtype=np.float32)
        cos = np.full(self._n, np.cos(2 * np.pi * hour / 24), dtype=np.float32)
        return np.stack([cong, sin, cos, wpop], axis=1).astype(np.float32)  # (N, 4)
