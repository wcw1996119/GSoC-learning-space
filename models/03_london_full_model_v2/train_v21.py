"""Train Linear utility for v2.1 (Step 4 Stage 2 calibration).

Uses cached STGNN V_j(t) as fixed prior. Fits 5 scalar params:
  V_ij^t = alpha_V·V_j(t) + alpha_occ·OccMatch_ij - beta·t_ij(t) - gamma·log d_ij + bias

Training: aggregate F_ij^t with MNL log-likelihood per (origin, hour).
Output: linear_utility_best.pt
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.linear_utility import LinearUtility, per_hour_log_p
from models_lib.occupation_match import (
    aggregate_occupation_match, grid_industry_vector
)


def build_aggregate_om(grid_features_df, occupation_df, grid_msoa):
    soc_cols = [f"prop_soc{i}" for i in range(1, 10)]
    msoa_soc = occupation_df[["MSOA21CD"] + soc_cols].copy()
    df = grid_msoa[["grid_id", "MSOA21CD"]].merge(msoa_soc, on="MSOA21CD", how="left")
    df = df.set_index("grid_id").reindex(grid_features_df["grid_id"].values).reset_index()
    soc_props = df[soc_cols].fillna(1.0 / 9).values
    soc_props = soc_props / (soc_props.sum(axis=1, keepdims=True) + 1e-12)
    gx = grid_industry_vector(grid_features_df)
    om_avg = aggregate_occupation_match(gx, soc_props)
    return om_avg, soc_props


def main(args):
    print("Loading cache...")
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    V_jt = torch.tensor(cache["V_jt_baseline"], dtype=torch.float32)
    t_ij_t = torch.tensor(cache["t_ij_t"], dtype=torch.float32)
    F_ij_t = torch.tensor(cache["F_ij_t"], dtype=torch.float32)
    coords_bng = cache["coords_bng"]
    train_mask = torch.tensor(cache["train_mask"])
    val_mask = torch.tensor(cache["val_mask"])
    test_mask = torch.tensor(cache["test_mask"])
    T, N = V_jt.shape
    print(f"T={T}, N={N}")

    # Log distance matrix
    diff = coords_bng[:, None, :] - coords_bng[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0
    log_d_ij = torch.tensor(np.log1p(d_km), dtype=torch.float32)

    # OccMatch
    print("Building OccMatch matrix...")
    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    grid_msoa = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_msoa_primary.csv")
    occupation = pd.read_csv(
        V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "london_occupation_msoa.csv"
    )
    om_avg, soc_props = build_aggregate_om(grid_feats, occupation, grid_msoa)
    om_avg_t = torch.tensor(om_avg, dtype=torch.float32)
    print(f"OccMatch: shape={om_avg.shape}, mean={om_avg.mean():.3f}")

    # Z-score normalize features (helps optimizer)
    V_norm = (V_jt - V_jt.mean()) / (V_jt.std() + 1e-8)
    t_norm = torch.log1p(t_ij_t)
    t_norm = (t_norm - t_norm.mean()) / (t_norm.std() + 1e-8)
    d_norm = (log_d_ij - log_d_ij.mean()) / (log_d_ij.std() + 1e-8)
    om_norm = (om_avg_t - om_avg_t.mean()) / (om_avg_t.std() + 1e-8)

    # Linear utility model
    util = LinearUtility()
    print(f"Linear utility params: {sum(p.numel() for p in util.parameters())}")

    optimizer = torch.optim.Adam(util.parameters(), lr=args.lr)

    print(f"\n=== Training {args.epochs} epochs ===")
    best_val = float("inf")
    cpc_den_train = float(F_ij_t[:, train_mask].sum() + 1e-8)
    cpc_den_val = float(F_ij_t[:, val_mask].sum() + 1e-8)

    for epoch in range(args.epochs):
        t0 = time.time()
        util.train()
        optimizer.zero_grad()

        nll_train_total = torch.tensor(0.0)
        active_train = 0
        for h in range(T):
            log_p_h = per_hour_log_p(util, V_norm[h], om_norm, t_norm[h], d_norm)
            flow_h = F_ij_t[h] * train_mask.view(-1, 1).float()
            nll_train_total = nll_train_total + (-(flow_h * log_p_h).sum())
            active_train += int((flow_h.sum(dim=1) > 0).sum())

        nll_train = nll_train_total / max(active_train, 1)
        nll_train.backward()
        torch.nn.utils.clip_grad_norm_(util.parameters(), max_norm=1.0)
        optimizer.step()

        util.eval()
        with torch.no_grad():
            nll_val_total = torch.tensor(0.0)
            active_val = 0
            cpc_num_train = 0.0
            cpc_num_val = 0.0
            for h in range(T):
                log_p_h = per_hour_log_p(util, V_norm[h], om_norm, t_norm[h], d_norm)
                flow_v_h = F_ij_t[h] * val_mask.view(-1, 1).float()
                nll_val_total = nll_val_total + (-(flow_v_h * log_p_h).sum())
                active_val += int((flow_v_h.sum(dim=1) > 0).sum())
                P_h = log_p_h.exp()
                origin_total_h = F_ij_t[h].sum(dim=1, keepdim=True)
                pred_F_h = P_h * origin_total_h
                cpc_num_train += float(torch.minimum(pred_F_h[train_mask], F_ij_t[h][train_mask]).sum())
                cpc_num_val += float(torch.minimum(pred_F_h[val_mask], F_ij_t[h][val_mask]).sum())
            nll_val = (nll_val_total / max(active_val, 1)).item()
            cpc_train = cpc_num_train / cpc_den_train
            cpc_val = cpc_num_val / cpc_den_val

        if nll_val < best_val:
            best_val = nll_val
            torch.save(util.state_dict(), V2_ROOT / "linear_utility_best.pt")
            star = "*"
        else:
            star = " "

        elapsed = time.time() - t0
        params_str = " ".join(f"{n}={p.item():.3f}" for n, p in util.named_parameters())
        print(
            f"Epoch {epoch:3d} | NLL t {nll_train:.3f} v {nll_val:.3f}{star} | "
            f"CPC t {cpc_train:.3f} v {cpc_val:.3f} | {params_str} | {elapsed:.1f}s"
        )

    print(f"\nBest val NLL: {best_val:.4f}")

    # Save aux for inference
    np.savez_compressed(
        V2_ROOT / "data" / "processed" / "training_aux.npz",
        soc_props=soc_props,
        om_avg=om_avg,
        log_d_ij=log_d_ij.numpy(),
        V_jt_mean=float(V_jt.mean()), V_jt_std=float(V_jt.std()),
        t_log_mean=float(torch.log1p(t_ij_t).mean()), t_log_std=float(torch.log1p(t_ij_t).std()),
        d_log_mean=float(log_d_ij.mean()), d_log_std=float(log_d_ij.std()),
        om_mean=float(om_avg_t.mean()), om_std=float(om_avg_t.std()),
    )
    print(f"Saved training_aux.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-2)
    args = parser.parse_args()
    main(args)
