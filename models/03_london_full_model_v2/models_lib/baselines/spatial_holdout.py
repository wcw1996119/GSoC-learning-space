"""Spatial + temporal holdout splitter for OD-flow baselines.

Spatial: hold out a set of London boroughs by seed.
Temporal: hold out specific hours (the model + baselines are evaluated at
hour=8 by default in the orchestrator, but the function returns the list
of holdout hours so each baseline can iterate over them consistently).
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def make_holdout(
    grid_borough_df: pd.DataFrame,
    od_df: pd.DataFrame,
    n_holdout_boroughs: int = 4,
    holdout_hours: Tuple[int, ...] = (5, 14, 22),
    seed: int = 42,
) -> Dict[str, object]:
    """Build spatial + temporal holdout masks for OD-pair rows.

    Parameters
    ----------
    grid_borough_df : DataFrame with columns ['grid_id', 'borough'].
    od_df           : DataFrame with at least ['grid_home', 'grid_work'].
    n_holdout_boroughs : number of boroughs to hold out (random, seeded).
    holdout_hours      : hour-of-day blocks to hold out for temporal split.
    seed               : RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        'train_mask'        : np.ndarray[bool] over od_df rows (spatial train)
        'test_mask'         : np.ndarray[bool] over od_df rows (spatial test)
        'holdout_boroughs'  : list[str]
        'holdout_hours'     : tuple[int]
        'all_boroughs'      : list[str]
    """
    rng = np.random.default_rng(seed)

    g2b = dict(zip(grid_borough_df["grid_id"], grid_borough_df["borough"]))
    all_boroughs = sorted(set(g2b.values()))
    if n_holdout_boroughs >= len(all_boroughs):
        raise ValueError(
            f"n_holdout_boroughs={n_holdout_boroughs} too large; "
            f"only {len(all_boroughs)} boroughs available"
        )

    holdout_idx = rng.choice(
        len(all_boroughs), size=n_holdout_boroughs, replace=False
    )
    holdout_boroughs = [all_boroughs[i] for i in holdout_idx]
    holdout_set = set(holdout_boroughs)

    home_b = od_df["grid_home"].map(g2b)
    work_b = od_df["grid_work"].map(g2b)

    in_holdout = home_b.isin(holdout_set) | work_b.isin(holdout_set)
    test_mask = in_holdout.to_numpy(dtype=bool)
    train_mask = ~test_mask

    # Drop OD rows whose grid is unmapped (NaN borough) from both sides.
    valid = home_b.notna().to_numpy() & work_b.notna().to_numpy()
    train_mask = train_mask & valid
    test_mask = test_mask & valid

    return {
        "train_mask": train_mask,
        "test_mask": test_mask,
        "holdout_boroughs": holdout_boroughs,
        "holdout_hours": tuple(holdout_hours),
        "all_boroughs": all_boroughs,
    }


def grid_index_map(grid_ids: Iterable[str]) -> Dict[str, int]:
    """Helper: stable grid_id -> 0..N-1 row index in feature matrix."""
    return {g: i for i, g in enumerate(grid_ids)}
