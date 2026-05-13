"""Bundle minimal data for Colab Deep Gravity training.

Extracts what Deep Gravity needs (X_static, F_ij_t, masks, log_d) into a single
~15 MB npz file that can be uploaded to Colab.

Run:
  python experiments/paper_a/prep_dg_colab_bundle.py

Output:
  evaluation_outputs/paper_a/dg_colab_bundle.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    cache = np.load(ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True)
    od = np.load(ROOT / "data" / "processed" / "grid_hourly_od_2019.npz")

    X_static = cache["static_features"].astype(np.float32)         # (1725, 22)
    train_mask = cache["train_mask"].astype(np.bool_)              # (1725,)
    val_mask = cache["val_mask"].astype(np.bool_)
    test_mask = cache["test_mask"].astype(np.bool_)
    coords = cache["coords_bng"].astype(np.float32)                # (1725, 2)
    F_ij_t = od["F_ij_t"].astype(np.float32)                       # (24, 1725, 1725)

    diff = coords[None, :, :] - coords[:, None, :]
    dist_m = np.linalg.norm(diff, axis=-1)
    dist_km = np.maximum(dist_m / 1000.0, 0.1)
    log_d = np.log(dist_km).astype(np.float32)

    out_path = ROOT / "evaluation_outputs" / "paper_a" / "dg_colab_bundle.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X_static=X_static,
        F_ij_t=F_ij_t,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        log_d=log_d,
        dist_km=dist_km.astype(np.float32),
        coords=coords,
    )
    print(f"Wrote {out_path}")
    print(f"  size = {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  N = {X_static.shape[0]}, T = {F_ij_t.shape[0]}")
    print(f"  train origins = {train_mask.sum()}/1725")
    print(f"  val   origins = {val_mask.sum()}/1725")
    print(f"  test  origins = {test_mask.sum()}/1725")
    print(f"  F_ij_t total = {F_ij_t.sum():.0f}")


if __name__ == "__main__":
    main()
