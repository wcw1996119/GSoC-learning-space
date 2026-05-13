"""Train v2.2 HeterogeneousUtility (Step 4 Stage 2 calibration).

Key v2.2 differences from v2.1:
- β is heterogeneous (fixed lookup), aggregated per origin grid for training
- α_V, α_occ, γ are softplus-positive (no negative values possible)
- α_occ has strong positive prior (per supervisor: OccMatch is decisive)

Output: heterogeneous_utility_best.pt
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

from models_lib.heterogeneous_utility import HeterogeneousUtility, compute_beta_per_origin
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

    # Aggregate β per origin grid (population-weighted from agents.csv)
    print("Computing aggregate β per origin grid...")
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    beta_origin = compute_beta_per_origin(agents_df, n_grids=N)
    print(f"β_origin: mean={beta_origin.mean():.4f}, range=[{beta_origin.min():.4f}, {beta_origin.max():.4f}]")
    # Grids without agents get default = 0.07 (mid×car)
    print(f"Grids with population-weighted β: {(beta_origin != 0.07).sum().item()} / {N}")

    # Z-score normalize features (helps optimizer)
    V_norm = (V_jt - V_jt.mean()) / (V_jt.std() + 1e-8)
    t_log = torch.log1p(t_ij_t)
    t_norm = (t_log - t_log.mean()) / (t_log.std() + 1e-8)
    d_norm = (log_d_ij - log_d_ij.mean()) / (log_d_ij.std() + 1e-8)
    om_norm = (om_avg_t - om_avg_t.mean()) / (om_avg_t.std() + 1e-8)

    # Note: β acts on RAW (un-normalized) t_ij in minutes. For training, we need to use
    # raw t_ij but then re-scale back. To keep things consistent with normalized features,
    # we'll fit β implicitly via softplus + normalized t_ij:
    #   utility uses t_norm (z-scored log t_ij)
    #   the heterogeneous β table represents "per minute" disutility on RAW t_ij
    # We translate: β_normalized = β_raw × t_log_std (rough approximation for log-scaled features)
    # For training, we'll use β_norm = β_origin × t_log_std (per-grid effective coefficient on z-scored t_log)
    t_log_std_val = float(t_log.std())
    beta_origin_for_training = beta_origin * t_log_std_val
    print(f"β_origin scaled for normalized features: mean={beta_origin_for_training.mean():.3f}")

    # Heterogeneous utility
    util = HeterogeneousUtility(
        alpha_V_init=0.5, alpha_occ_init=0.7, gamma_init=0.5
    )
    print(f"v2.2 utility init: alpha_V={util.alpha_V.item():.3f}, "
          f"alpha_occ={util.alpha_occ.item():.3f}, gamma={util.gamma.item():.3f}")

    optimizer = torch.optim.Adam(util.parameters(), lr=args.lr)

    # Strong positive prior for alpha_occ: encourage value near 1.0 (supervisor's belief)
    alpha_occ_prior = 1.0
    prior_weight = 0.05  # regularization strength

    print(f"\n=== Training {args.epochs} epochs (v2.2 heterogeneous) ===")
    best_val = float("inf")
    cpc_den_train = float(F_ij_t[:, train_mask].sum() + 1e-8)
    cpc_den_val = float(F_ij_t[:, val_mask].sum() + 1e-8)

    def per_hour_log_p(util_module, t_idx):
        Vj_b = V_norm[t_idx][None, :].expand(N, N)
        # broadcast aggregate β to (N, N) — β depends on origin only
        beta_b = beta_origin_for_training.unsqueeze(1).expand(N, N)
        V_ij = util_module.utility_aggregate(
            V_j_t=Vj_b,
            occ_match_avg=om_norm,
            t_ij_t=t_norm[t_idx],
            log_d_ij=d_norm,
            beta_origin=beta_b,
        )
        return torch.log_softmax(V_ij, dim=1)

    for epoch in range(args.epochs):
        t0 = time.time()
        util.train()
        optimizer.zero_grad()

        nll_total = torch.tensor(0.0)
        active = 0
        for h in range(T):
            log_p_h = per_hour_log_p(util, h)
            flow_h = F_ij_t[h] * train_mask.view(-1, 1).float()
            nll_total = nll_total + (-(flow_h * log_p_h).sum())
            active += int((flow_h.sum(dim=1) > 0).sum())

        nll_train = nll_total / max(active, 1)

        # Add prior regularization on alpha_occ (pull toward 1.0)
        prior_loss = prior_weight * (util.alpha_occ - alpha_occ_prior).pow(2)
        loss = nll_train + prior_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(util.parameters(), max_norm=1.0)
        optimizer.step()

        util.eval()
        with torch.no_grad():
            nll_val_total = torch.tensor(0.0)
            active_val = 0
            cpc_num_train, cpc_num_val = 0.0, 0.0
            for h in range(T):
                log_p_h = per_hour_log_p(util, h)
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
            torch.save(util.state_dict(), V2_ROOT / "heterogeneous_utility_best.pt")
            star = "*"
        else:
            star = " "

        elapsed = time.time() - t0
        a_V = util.alpha_V.item()
        a_occ = util.alpha_occ.item()
        g = util.gamma.item()
        print(
            f"Ep {epoch:3d} | NLL t {nll_train:.3f} v {nll_val:.3f}{star} | "
            f"CPC t {cpc_train:.3f} v {cpc_val:.3f} | "
            f"αV={a_V:.3f} αocc={a_occ:.3f} γ={g:.3f} bias={util.bias.item():.3f} | "
            f"prior={prior_loss.item():.4f} | {elapsed:.1f}s"
        )

    print(f"\nBest val NLL: {best_val:.4f}")

    # Save aux for inference (extends v2.1 aux)
    np.savez_compressed(
        V2_ROOT / "data" / "processed" / "training_aux_v22.npz",
        soc_props=soc_props,
        om_avg=om_avg,
        log_d_ij=log_d_ij.numpy(),
        beta_origin=beta_origin.numpy(),
        beta_origin_for_training=beta_origin_for_training.numpy(),
        V_jt_mean=float(V_jt.mean()), V_jt_std=float(V_jt.std()),
        t_log_mean=float(t_log.mean()), t_log_std=float(t_log.std()),
        d_log_mean=float(log_d_ij.mean()), d_log_std=float(log_d_ij.std()),
        om_mean=float(om_avg_t.mean()), om_std=float(om_avg_t.std()),
    )
    print(f"Saved training_aux_v22.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-2)
    args = parser.parse_args()
    main(args)
