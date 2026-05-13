"""v2 counterfactual evaluation — implements methodology/02 three-evidence framework.

1. Plausibility distance — k-NN distance from scenario input to training distribution.
2. Magnitude scan — sweep employment perturbation magnitude, plot key outcomes.
3. Multi-model cross-check — run gravity baseline + observed-prior RUM + STGNN.

Outputs: figures + summary table to evaluation_outputs/
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from data_loader import load_v2_data, build_features_tensor, spatial_holdout
from providers.travel_time import LondonBPRProvider
from providers.graph_builder import build_knn_graph
from models_lib.stgnn import V2_STGNN
from models_lib.rum import rum_closure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = V2_ROOT / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)


def plausibility_score(scenario_input: np.ndarray, training_data: np.ndarray, k=10) -> float:
    """Mean L2 distance to k nearest training samples."""
    diffs = training_data - scenario_input[None, :]
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    return float(np.partition(dists, k)[:k].mean())


def gravity_baseline(F_ij_t: torch.Tensor, t_ij_t: torch.Tensor, beta: float = 0.07):
    """Simple gravity: P(j|i,t) ∝ exp(-beta * t_ij(t))."""
    T, N, _ = F_ij_t.shape
    log_p = torch.log_softmax(-beta * t_ij_t.expand(T, N, N), dim=2)
    return log_p


def observed_prior(F_ij_t: torch.Tensor, t_ij_t: torch.Tensor, beta: float = 0.07):
    """RUM with V_j = log(observed_inflow_j) — uses OD data as prior, no GNN."""
    T, N, _ = F_ij_t.shape
    inflow = F_ij_t.sum(dim=(0, 1))  # (N,) total inflow per destination
    V_j = torch.log1p(inflow)  # (N,)
    V_jt = V_j[None, :].expand(T, N)
    return rum_closure(V_jt, t_ij_t, beta=beta)


def cpc(pred: torch.Tensor, obs: torch.Tensor) -> float:
    return float(torch.minimum(pred, obs).sum() / (obs.sum() + 1e-8))


def predict_F(log_p: torch.Tensor, origin_total_t: torch.Tensor) -> torch.Tensor:
    """Distribute origin total flow over destinations by P(j|i,t)."""
    return log_p.exp() * origin_total_t


def main(args):
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading data...")
    d = load_v2_data()
    N, T = d["N"], d["T"]
    x_seq = build_features_tensor(d["feat_provider"], T=T).to(DEVICE)
    edge_index, edge_attr = build_knn_graph(d["coords_bng"], K=10, add_self_loop=True)
    edge_index = edge_index.to(DEVICE)
    edge_attr = edge_attr.to(DEVICE)
    tt = LondonBPRProvider(d["coords_bng"], d["grid_borough_idx"], d["borough_hourly_congestion"])
    t_ij_t = torch.tensor(np.stack([tt.get_matrix(h) for h in range(T)], axis=0)).to(DEVICE)
    F_ij_t = torch.tensor(d["F_ij_t"]).to(DEVICE)
    train_mask, val_mask, test_mask = spatial_holdout(N, seed=42)
    test_mask_t = torch.tensor(test_mask).to(DEVICE)

    # Load trained model
    print("Loading trained STGNN...")
    node_dim = x_seq.shape[-1]
    edge_dim = edge_attr.shape[1]
    model = V2_STGNN(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden_dim=args.hidden, gat_heads=args.heads, gru_hidden=args.gru_hidden,
    ).to(DEVICE)
    model.load_state_dict(torch.load(V2_ROOT / "best_model.pt", map_location=DEVICE))
    model.eval()

    origin_total_t = F_ij_t.sum(dim=2, keepdim=True)  # (T, N, 1)

    # ====== EVIDENCE 3: Multi-model cross-check (baseline) ======
    print("\n=== Evidence 3: Multi-model cross-check (baseline) ===")
    with torch.no_grad():
        # GNN
        V_jt = model(x_seq, edge_index, edge_attr)
        log_p_gnn = rum_closure(V_jt, t_ij_t, beta=args.beta)
        pred_gnn = predict_F(log_p_gnn, origin_total_t)
        cpc_gnn = cpc(pred_gnn[:, test_mask_t], F_ij_t[:, test_mask_t])

        # Gravity
        log_p_grv = gravity_baseline(F_ij_t, t_ij_t, beta=args.beta)
        pred_grv = predict_F(log_p_grv, origin_total_t)
        cpc_grv = cpc(pred_grv[:, test_mask_t], F_ij_t[:, test_mask_t])

        # Observed-prior RUM
        log_p_obs = observed_prior(F_ij_t, t_ij_t, beta=args.beta)
        pred_obs = predict_F(log_p_obs, origin_total_t)
        cpc_obs = cpc(pred_obs[:, test_mask_t], F_ij_t[:, test_mask_t])

    print(f"  STGNN+RUM    CPC (test): {cpc_gnn:.4f}")
    print(f"  Gravity      CPC (test): {cpc_grv:.4f}")
    print(f"  Observed-RUM CPC (test): {cpc_obs:.4f}")

    # Save table
    with open(OUT_DIR / "cross_check_table.txt", "w") as f:
        f.write("Multi-model cross-check (test set)\n")
        f.write("=" * 50 + "\n")
        f.write(f"  STGNN + RUM     CPC = {cpc_gnn:.4f}\n")
        f.write(f"  Gravity         CPC = {cpc_grv:.4f}\n")
        f.write(f"  Observed prior  CPC = {cpc_obs:.4f}\n")

    # ====== EVIDENCE 1: Plausibility ======
    # We treat each grid's static feature vector as a "sample" in training distribution.
    # When we perturb a grid (e.g. +N employment), we measure k-NN distance from
    # perturbed feature vector to all training-grid feature vectors.
    print("\n=== Evidence 1: Plausibility distance (per scenario) ===")
    static = d["feat_provider"].get_static()  # (N, F_static) normalized
    train_static = static[train_mask]
    print(f"  training distribution: {train_static.shape}")

    # ====== EVIDENCE 2: Magnitude scan on a candidate "副中心" grid ======
    # Pick the grid with highest current employment as the "副中心" candidate
    feat_raw = d["feat_provider"].grid[["grid_id", "total_employment", "centroid_lat", "centroid_lon"]]
    target_idx = int(np.argmax(feat_raw["total_employment"].values))
    target_grid = feat_raw.iloc[target_idx]
    print(f"\n=== Evidence 2: Magnitude scan ===")
    print(f"  target grid: {target_grid['grid_id']} "
          f"(current employment ~{target_grid['total_employment']:.0f}, "
          f"lat={target_grid['centroid_lat']:.4f})")

    magnitudes = [0, 5_000, 10_000, 20_000, 50_000, 100_000]
    results = []

    # Target grid base feature row (raw, log-transformed-then-zscore in get_static)
    # For magnitude scan, we need to perturb in *raw* space then re-normalize via
    # the same log+zscore pipeline. Implementation:
    # Use the raw grid DataFrame directly, modify the target row, re-create features.
    grid_df = d["feat_provider"].grid.copy()
    base_emp = float(grid_df.loc[target_idx, "total_employment"])
    base_sec_office = float(grid_df.loc[target_idx, "sec6_info_finance"])

    # Save baseline V_j for reference
    with torch.no_grad():
        V_jt_base = model(x_seq, edge_index, edge_attr)  # (T, N)

    for delta in magnitudes:
        # Perturb: increase total_employment AND sec6_info_finance by delta
        # (treat new center as office-cluster style — like Canary Wharf / Silicon Roundabout)
        new_grid_df = grid_df.copy()
        new_grid_df.loc[target_idx, "total_employment"] = base_emp + delta
        new_grid_df.loc[target_idx, "sec6_info_finance"] = base_sec_office + delta

        # Rebuild features through the provider's normalization pipeline
        from providers.features import LondonFeatureProvider, STATIC_COLS, LOG_TRANSFORM_COLS
        new_provider = object.__new__(LondonFeatureProvider)
        # Manually init with this new grid
        new_provider.grid = new_grid_df
        new_provider._grid_ids = d["feat_provider"]._grid_ids
        new_provider._n = d["feat_provider"]._n
        new_provider.congestion_matrix = d["feat_provider"].congestion_matrix
        new_provider.cong_mean = d["feat_provider"].cong_mean
        new_provider.cong_std = d["feat_provider"].cong_std
        new_provider.cong_norm = d["feat_provider"].cong_norm
        new_provider.workplace_pop_norm = d["feat_provider"].workplace_pop_norm
        new_provider.static_mean = d["feat_provider"].static_mean
        new_provider.static_std = d["feat_provider"].static_std
        # Recompute static normalized using SAME stats (don't re-fit)
        raw = new_grid_df[STATIC_COLS].values.astype(np.float64)
        log_idx = [i for i, c in enumerate(STATIC_COLS) if c in LOG_TRANSFORM_COLS]
        raw[:, log_idx] = np.log1p(raw[:, log_idx])
        new_provider.static_norm = (raw - new_provider.static_mean) / new_provider.static_std

        # New feature tensor
        x_seq_new = build_features_tensor(new_provider, T=T).to(DEVICE)

        with torch.no_grad():
            V_jt_new = model(x_seq_new, edge_index, edge_attr)
            log_p_new = rum_closure(V_jt_new, t_ij_t, beta=args.beta)
            P_new = log_p_new.exp()
            # Pred flow into target grid (sum over hours, all origins)
            inflow_target = float(P_new[:, :, target_idx].sum() * origin_total_t.mean())

        # Plausibility on the target grid's static vector
        plaus = plausibility_score(new_provider.static_norm[target_idx], train_static, k=10)
        # Reference: distribution of training plausibility distances
        if delta == 0:
            train_plaus_dists = np.array([
                plausibility_score(train_static[i], np.delete(train_static, i, axis=0), k=10)
                for i in range(min(50, len(train_static)))
            ])
            train_plaus_pctile_thresh = np.percentile(train_plaus_dists, [50, 90])
            print(f"  Training plausibility 50th pct: {train_plaus_pctile_thresh[0]:.3f}, 90th pct: {train_plaus_pctile_thresh[1]:.3f}")

        results.append({"delta": delta, "plausibility": plaus, "inflow_target": inflow_target})
        print(f"  delta=+{delta:>7,d}  plaus={plaus:6.3f}  inflow_target={inflow_target:8.1f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    deltas = [r["delta"] for r in results]
    plaus_arr = [r["plausibility"] for r in results]
    inflow_arr = [r["inflow_target"] for r in results]

    axes[0].plot(deltas, inflow_arr, "o-", color="tab:blue")
    axes[0].set_xlabel("Employment delta in target grid")
    axes[0].set_ylabel("Predicted inflow into target")
    axes[0].set_title(f"Magnitude scan: target {target_grid['grid_id']}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(deltas, plaus_arr, "s-", color="tab:orange")
    axes[1].axhline(train_plaus_pctile_thresh[0], color="green", linestyle="--", label="50th pct (in-dist)")
    axes[1].axhline(train_plaus_pctile_thresh[1], color="red", linestyle="--", label="90th pct (far OOD)")
    axes[1].set_xlabel("Employment delta")
    axes[1].set_ylabel("Plausibility distance (k-NN, k=10)")
    axes[1].set_title("Plausibility evolution")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "magnitude_scan.png", dpi=120)
    print(f"\nSaved plot -> {OUT_DIR / 'magnitude_scan.png'}")

    # Summary table
    with open(OUT_DIR / "magnitude_scan.csv", "w") as f:
        f.write("delta,plausibility,inflow_target,credibility_tier\n")
        for r in results:
            tier = "in-dist" if r["plausibility"] < train_plaus_pctile_thresh[0] else (
                "near-dist" if r["plausibility"] < train_plaus_pctile_thresh[1] else "far-OOD"
            )
            f.write(f"{r['delta']},{r['plausibility']:.4f},{r['inflow_target']:.2f},{tier}\n")

    print("\n=== Done ===")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--gru_hidden", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.07)
    args = parser.parse_args()
    main(args)
