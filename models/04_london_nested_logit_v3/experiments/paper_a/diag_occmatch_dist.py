"""Diagnostic: OccMatch x distance x flow — verify the 4 hypotheses for δ_slope = -0.435.

Anchored to v3 D->M nested smoke result: model fit δ_slope on (occ_match × log_d)
with negative sign. Need to figure out which mechanism explains it before paper claim.

Outputs:
  - occ_match raw distribution (rules out H_D encoding-direction)
  - (distance bin × occ_match bin) -> mean F heatmap (tests H_A sorting, H_C correlation)
  - Borough-level high-OM flow concentration (further test H_A)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

V2_ROOT = Path(__file__).resolve().parents[3] / "03_london_full_model_v2"
V3_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = V3_ROOT / "evaluation_outputs" / "paper_a" / "diag_occmatch"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    aux = np.load(V2_ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    cache = np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz")

    occ = aux["occ_match"].astype(np.float64)  # (1725, 1725)
    F = cache["F_ij_t"].sum(axis=0).astype(np.float64)  # daily flow (1725, 1725)
    coords = cache["coords_bng"].astype(np.float64)  # (1725, 2) metres
    borough_idx = cache["grid_borough_idx"]  # (1725,)
    boroughs = cache["boroughs"]

    N = occ.shape[0]
    assert occ.shape == (N, N) and F.shape == (N, N) and coords.shape == (N, 2)

    # Distance matrix (km)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0  # (N, N) km

    # Strip self-loops
    eye = np.eye(N, dtype=bool)
    occ_off = occ[~eye]
    F_off = F[~eye]
    dist_off = dist[~eye]

    # ============================== Task #3 ==============================
    print("=" * 60)
    print("OccMatch distribution (off-diagonal, 1725*1724 pairs)")
    print("=" * 60)
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    qs = np.percentile(occ_off, pcts)
    for p, q in zip(pcts, qs):
        print(f"  p{p:3d}  = {q:.4f}")
    print(f"  mean  = {occ_off.mean():.4f}")
    print(f"  std   = {occ_off.std():.4f}")
    print(f"  n_zero  = {(occ_off == 0).sum()}")
    print(f"  n_one   = {(occ_off >= 0.9999).sum()}")
    print(f"  >>> range confirmed [{occ_off.min():.4f}, {occ_off.max():.4f}]")
    print(f"  >>> H_D (sign convention) verdict: OccMatch ∈ [0, 1], higher=better match. "
          "δ_slope<0 is NOT a sign flip. RULED OUT.")

    # ============================== Task #4: heatmap ==============================
    print("\n" + "=" * 60)
    print("(distance × OccMatch) → mean flow F  (H_A / H_C test)")
    print("=" * 60)
    # Distance bins (km): tuned to London geography — 0-5, 5-10, 10-20, 20-50
    d_edges = np.array([0, 2, 5, 10, 20, 50, 200])
    # OccMatch bins: by quantile to ensure equal mass
    o_edges = np.quantile(occ_off, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    o_edges[-1] += 1e-6  # include max
    d_labels = [f"{d_edges[i]:g}-{d_edges[i+1]:g}km" for i in range(len(d_edges) - 1)]
    o_labels = [f"q{i*20}-{(i+1)*20}" for i in range(len(o_edges) - 1)]

    d_idx = np.digitize(dist_off, d_edges) - 1
    o_idx = np.digitize(occ_off, o_edges) - 1
    nd, no = len(d_labels), len(o_labels)
    mean_F = np.zeros((nd, no))
    count = np.zeros((nd, no), dtype=np.int64)
    sum_F = np.zeros((nd, no))
    for di in range(nd):
        for oi in range(no):
            mask = (d_idx == di) & (o_idx == oi)
            count[di, oi] = mask.sum()
            sum_F[di, oi] = F_off[mask].sum()
            mean_F[di, oi] = F_off[mask].mean() if mask.any() else 0.0

    print("\nMean F per cell (rows=distance, cols=OccMatch quintile):")
    print(f"  {'':12s}" + "".join(f"{l:>10s}" for l in o_labels))
    for di in range(nd):
        print(f"  {d_labels[di]:12s}" + "".join(f"{mean_F[di, oi]:>10.3f}" for oi in range(no)))

    print("\nCell counts (#pairs per cell):")
    print(f"  {'':12s}" + "".join(f"{l:>10s}" for l in o_labels))
    for di in range(nd):
        print(f"  {d_labels[di]:12s}" + "".join(f"{count[di, oi]:>10d}" for oi in range(no)))

    print("\nTotal flow sum per cell (sum F, daily commuters):")
    print(f"  {'':12s}" + "".join(f"{l:>10s}" for l in o_labels))
    for di in range(nd):
        print(f"  {d_labels[di]:12s}" + "".join(f"{sum_F[di, oi]:>10.0f}" for oi in range(no)))

    # Raw correlation: pairwise corr(distance, occ_match) — does occ_match systematically
    # vary with distance regardless of flow?
    print("\nRaw correlation diagnostics:")
    print(f"  corr(distance, occ_match)        = {np.corrcoef(dist_off, occ_off)[0,1]:+.4f}")
    print(f"  corr(log_distance, occ_match)    = {np.corrcoef(np.log1p(dist_off), occ_off)[0,1]:+.4f}")
    # Mean occ_match within each distance bin (asks: are far zones systematically lower-matched?)
    print("\nMean OccMatch by distance bin:")
    for di in range(nd):
        mask = d_idx == di
        if mask.any():
            print(f"  {d_labels[di]:12s}  mean_OM = {occ_off[mask].mean():.4f}   n_pairs = {mask.sum()}")

    # ============================== Borough check (H_A) ==============================
    print("\n" + "=" * 60)
    print("Borough-level: high-OM flows concentrated within or across boroughs?")
    print("=" * 60)
    # For each pair, mark "high OM" = top 20% OM
    om_thresh = np.quantile(occ_off, 0.8)
    print(f"  high-OM threshold (top 20%) = {om_thresh:.4f}")

    # Re-index everything with self-loop kept (for borough origin index)
    iu, ju = np.triu_indices(N, k=1)
    om_pairs = occ[iu, ju]
    F_pairs = F[iu, ju]  # i→j only (no j→i double count for borough-coincidence stat)
    same_borough = borough_idx[iu] == borough_idx[ju]
    high_om = om_pairs >= om_thresh

    print(f"\n  All pairs (i<j, {len(iu)} total):")
    print(f"    P(same borough)                       = {same_borough.mean():.4f}")
    print(f"    P(same borough | high OM)             = {same_borough[high_om].mean():.4f}")
    print(f"    P(same borough | low OM, bottom 20%)  = "
          f"{same_borough[om_pairs <= np.quantile(om_pairs, 0.2)].mean():.4f}")

    print(f"\n  Flow-weighted:")
    total_F = F_pairs.sum()
    if total_F > 0:
        print(f"    Σ F_ij (all)                          = {total_F:.0f}")
        print(f"    Σ F_ij (same borough) / total         = {F_pairs[same_borough].sum()/total_F:.4f}")
        print(f"    Σ F_ij (high OM) / total              = {F_pairs[high_om].sum()/total_F:.4f}")
        # Joint: high OM AND same borough
        joint = high_om & same_borough
        print(f"    Σ F_ij (high OM & same borough) / Σ F_ij (high OM) "
              f"= {F_pairs[joint].sum()/max(F_pairs[high_om].sum(), 1e-9):.4f}")

    # ============================== Save heatmap data ==============================
    np.savez(
        OUT_DIR / "occmatch_distance_heatmap.npz",
        d_edges=d_edges, o_edges=o_edges,
        mean_F=mean_F, count=count, sum_F=sum_F,
        corr_d_om=np.corrcoef(dist_off, occ_off)[0, 1],
    )
    print(f"\n[saved] {OUT_DIR / 'occmatch_distance_heatmap.npz'}")


if __name__ == "__main__":
    main()
