"""v2 STGNN training script — first runnable demo.

Loads:
- 1km grid + features + edges
- synthesized F_ij^t
- spatial holdout (75/10/15)
- production-constrained MNL loss

Trains a small T-GCN-style model. Logs NLL + CPC each epoch.

Usage:
    python train.py [--epochs 30] [--hidden 64] [--lr 1e-3]
"""
import argparse
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from data_loader import load_v2_data, build_features_tensor, spatial_holdout
from providers.travel_time import LondonBPRProvider
from providers.graph_builder import build_knn_graph
from models_lib.stgnn import V2_STGNN
from models_lib.rum import rum_closure
from models_lib.loss import production_constrained_mnl_nll

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpc(pred_flows: torch.Tensor, obs_flows: torch.Tensor) -> float:
    """Common Part of Commuters: Σ min(pred, obs) / Σ obs."""
    return float(torch.minimum(pred_flows, obs_flows).sum() / (obs_flows.sum() + 1e-8))


def train(args):
    print(f"Device: {DEVICE}")

    # 1. Load data
    print("\n=== Loading data ===")
    d = load_v2_data()
    N, T = d["N"], d["T"]

    # 2. Build features tensor
    print("\n=== Building features tensor ===")
    x_seq = build_features_tensor(d["feat_provider"], T=T).to(DEVICE)
    print(f"x_seq shape: {tuple(x_seq.shape)}")

    # 3. Build graph
    print("\n=== Building kNN(K=10) graph ===")
    edge_index, edge_attr = build_knn_graph(d["coords_bng"], K=args.K, add_self_loop=True)
    edge_index = edge_index.to(DEVICE)
    edge_attr = edge_attr.to(DEVICE)
    print(f"edge_index: {tuple(edge_index.shape)}, edge_attr: {tuple(edge_attr.shape)}")

    # 4. Travel time provider
    print("\n=== Building travel time provider ===")
    tt = LondonBPRProvider(
        grid_centroids_bng_m=d["coords_bng"],
        grid_borough_idx=d["grid_borough_idx"],
        borough_hourly_congestion=d["borough_hourly_congestion"],
    )
    # Precompute t_ij(t) for all hours: (T, N, N)
    print("Precomputing t_ij(t) for 24 hours...")
    t_ij_t = np.stack([tt.get_matrix(h) for h in range(T)], axis=0).astype(np.float32)
    t_ij_t = torch.tensor(t_ij_t).to(DEVICE)
    print(f"t_ij_t shape: {tuple(t_ij_t.shape)}, mean: {t_ij_t.mean():.2f} min")

    # 5. F_ij^t target tensor
    F_ij_t = torch.tensor(d["F_ij_t"]).to(DEVICE)
    print(f"F_ij_t shape: {tuple(F_ij_t.shape)}")

    # 6. Spatial holdout
    print("\n=== Spatial holdout 75/10/15 ===")
    train_mask, val_mask, test_mask = spatial_holdout(N, seed=args.seed)
    print(f"Train: {train_mask.sum()} grids, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    train_mask_t = torch.tensor(train_mask).to(DEVICE)
    val_mask_t = torch.tensor(val_mask).to(DEVICE)
    test_mask_t = torch.tensor(test_mask).to(DEVICE)

    # 7. Build model — derive dims from actual tensors
    node_dim = x_seq.shape[-1]
    edge_dim = edge_attr.shape[1]

    print(f"\n=== Building model (node_dim={node_dim}, edge_dim={edge_dim}, hidden={args.hidden}) ===")
    model = V2_STGNN(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden_dim=args.hidden, gat_heads=args.heads, gru_hidden=args.gru_hidden,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    beta = args.beta  # WebTAG VOT, ~0.07/min

    # 8. Training loop
    print(f"\n=== Training {args.epochs} epochs (lr={args.lr}, beta={beta}) ===")
    best_val_nll = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        V_jt = model(x_seq, edge_index, edge_attr)  # (T, N)
        log_p = rum_closure(V_jt, t_ij_t, beta=beta)  # (T, N, N)
        loss = production_constrained_mnl_nll(log_p, F_ij_t, origin_mask=train_mask_t)
        loss.backward()
        # Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            V_jt = model(x_seq, edge_index, edge_attr)
            log_p = rum_closure(V_jt, t_ij_t, beta=beta)
            val_nll = production_constrained_mnl_nll(log_p, F_ij_t, origin_mask=val_mask_t).item()
            test_nll = production_constrained_mnl_nll(log_p, F_ij_t, origin_mask=test_mask_t).item()

            # CPC: per origin, distribute origin total flow over destinations by P(j|i,t)
            # Compute predicted F_ij^t (full)
            P = log_p.exp()  # (T, N, N)
            origin_total_t = F_ij_t.sum(dim=2, keepdim=True)  # (T, N, 1)
            pred_F = P * origin_total_t  # (T, N, N)
            cpc_train = cpc(
                pred_F[:, train_mask_t].reshape(-1),
                F_ij_t[:, train_mask_t].reshape(-1),
            )
            cpc_val = cpc(
                pred_F[:, val_mask_t].reshape(-1),
                F_ij_t[:, val_mask_t].reshape(-1),
            )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(model.state_dict(), V2_ROOT / "best_model.pt")
            star = "*"
        else:
            star = " "

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | train NLL {loss.item():8.4f} | val NLL {val_nll:8.4f}{star} | "
            f"test NLL {test_nll:8.4f} | CPC train {cpc_train:.4f} val {cpc_val:.4f} | {elapsed:.1f}s"
        )

    print(f"\nBest val NLL: {best_val_nll:.4f}")
    print(f"Saved -> {V2_ROOT / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
