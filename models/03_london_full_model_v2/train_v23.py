"""Train v2.3 WageUtility on aggregate F_ij^t.

5 learnable params (α_V, α_occ, α_wage, γ, bias).
β is non-learnable — agent-level lookup at inference.
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

from models_lib.wage_utility import WageUtility, compute_beta_per_origin_v23
from models_lib.occupation_match import (
    aggregate_occupation_match, grid_industry_vector
)
from models_lib.wage_priors import LONDON_MEDIAN_WAGE


def main(args):
    print("Loading...")
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    V_jt = torch.tensor(cache["V_jt_baseline"], dtype=torch.float32)
    t_ij_t = torch.tensor(cache["t_ij_t"], dtype=torch.float32)
    F_ij_t = torch.tensor(cache["F_ij_t"], dtype=torch.float32)
    coords_bng = cache["coords_bng"]
    train_mask = torch.tensor(cache["train_mask"])
    val_mask = torch.tensor(cache["val_mask"])
    T, N = V_jt.shape

    # log distance
    diff = coords_bng[:, None, :] - coords_bng[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0
    log_d_ij = torch.tensor(np.log1p(d_km), dtype=torch.float32)

    # OccMatch
    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features_v23.csv")
    grid_msoa = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_msoa_primary.csv")
    occ = pd.read_csv(V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "london_occupation_msoa.csv")
    soc_cols = [f"prop_soc{i}" for i in range(1, 10)]
    df = grid_msoa[["grid_id","MSOA21CD"]].merge(occ[["MSOA21CD"] + soc_cols], on="MSOA21CD", how="left")
    df = df.set_index("grid_id").reindex(grid_feats["grid_id"]).reset_index()
    soc_props = df[soc_cols].fillna(1/9).values
    soc_props = soc_props / soc_props.sum(axis=1, keepdims=True)
    gx = grid_industry_vector(grid_feats)
    om_avg = aggregate_occupation_match(gx, soc_props)
    om_avg_t = torch.tensor(om_avg, dtype=torch.float32)

    # wage_j (NEW for v2.3)
    log_wage_norm = (
        np.log(grid_feats["wage_j_weekly"].values) - np.log(LONDON_MEDIAN_WAGE)
    ).astype(np.float32)
    log_wage_norm_t = torch.tensor(log_wage_norm)
    print(f"log_wage_norm: mean={log_wage_norm.mean():.3f}, std={log_wage_norm.std():.3f}, "
          f"range=[{log_wage_norm.min():.3f}, {log_wage_norm.max():.3f}]")

    # β per origin (from agents' real income)
    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population_v23.csv")
    beta_origin = compute_beta_per_origin_v23(agents_df, n_grids=N)
    print(f"β_origin: mean={beta_origin.mean():.4f}, range=[{beta_origin.min():.4f}, {beta_origin.max():.4f}]")

    # Z-score features
    V_norm = (V_jt - V_jt.mean()) / (V_jt.std() + 1e-8)
    t_log = torch.log1p(t_ij_t)
    t_norm = (t_log - t_log.mean()) / (t_log.std() + 1e-8)
    d_norm = (log_d_ij - log_d_ij.mean()) / (log_d_ij.std() + 1e-8)
    om_norm = (om_avg_t - om_avg_t.mean()) / (om_avg_t.std() + 1e-8)
    # log_wage already centred at 0 since /median
    t_log_std = float(t_log.std())
    beta_origin_train = beta_origin * t_log_std

    util = WageUtility(alpha_V_init=0.5, alpha_occ_init=0.7,
                       alpha_wage_init=0.3, gamma_init=0.5)
    optim = torch.optim.Adam(util.parameters(), lr=args.lr)

    print(f"\n=== Training v2.3 (5 params, {args.epochs} epochs) ===")
    cpc_den_train = float(F_ij_t[:, train_mask].sum() + 1e-8)
    cpc_den_val = float(F_ij_t[:, val_mask].sum() + 1e-8)

    def per_hour_log_p(util_module, h):
        Vj_b = V_norm[h][None, :].expand(N, N)
        wage_b = log_wage_norm_t[None, :].expand(N, N)
        beta_b = beta_origin_train.unsqueeze(1).expand(N, N)
        V_ij = util_module.utility_aggregate(
            V_j_t=Vj_b, occ_match_avg=om_norm,
            log_wage_norm=wage_b,
            t_ij_t=t_norm[h], log_d_ij=d_norm,
            beta_origin=beta_b,
        )
        return torch.log_softmax(V_ij, dim=1)

    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        util.train(); optim.zero_grad()
        nll = torch.tensor(0.0); active = 0
        for h in range(T):
            log_p = per_hour_log_p(util, h)
            flow = F_ij_t[h] * train_mask.view(-1, 1).float()
            nll = nll + (-(flow * log_p).sum())
            active += int((flow.sum(dim=1) > 0).sum())
        nll_train = nll / max(active, 1)
        # priors: alpha_occ → 1.0, alpha_wage → 0.5
        prior = 0.05 * (util.alpha_occ - 1.0).pow(2) + 0.05 * (util.alpha_wage - 0.5).pow(2)
        loss = nll_train + prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_(util.parameters(), 1.0)
        optim.step()

        util.eval()
        with torch.no_grad():
            nll_v = torch.tensor(0.0); active_v = 0
            cpc_t, cpc_v = 0.0, 0.0
            for h in range(T):
                log_p = per_hour_log_p(util, h)
                flow_v = F_ij_t[h] * val_mask.view(-1, 1).float()
                nll_v = nll_v + (-(flow_v * log_p).sum())
                active_v += int((flow_v.sum(dim=1) > 0).sum())
                P = log_p.exp()
                pred_F = P * F_ij_t[h].sum(dim=1, keepdim=True)
                cpc_t += float(torch.minimum(pred_F[train_mask], F_ij_t[h][train_mask]).sum())
                cpc_v += float(torch.minimum(pred_F[val_mask], F_ij_t[h][val_mask]).sum())
            nll_val = (nll_v / max(active_v, 1)).item()
            cpc_train = cpc_t / cpc_den_train
            cpc_val = cpc_v / cpc_den_val

        star = "*" if nll_val < best_val else " "
        if nll_val < best_val:
            best_val = nll_val
            torch.save(util.state_dict(), V2_ROOT / "wage_utility_best.pt")

        print(
            f"Ep {ep:3d} | NLL t {nll_train:.3f} v {nll_val:.3f}{star} | "
            f"CPC t {cpc_train:.3f} v {cpc_val:.3f} | "
            f"αV={util.alpha_V:.3f} αocc={util.alpha_occ:.3f} αwage={util.alpha_wage:.3f} "
            f"γ={util.gamma:.3f} | {time.time()-t0:.1f}s"
        )

    print(f"\nBest val NLL: {best_val:.4f}")
    np.savez_compressed(
        V2_ROOT / "data" / "processed" / "training_aux_v23.npz",
        log_wage_norm=log_wage_norm,
        beta_origin=beta_origin.numpy(),
        beta_origin_train=beta_origin_train.numpy(),
        soc_props=soc_props, om_avg=om_avg, log_d_ij=log_d_ij.numpy(),
        V_jt_mean=float(V_jt.mean()), V_jt_std=float(V_jt.std()),
        t_log_mean=float(t_log.mean()), t_log_std=float(t_log.std()),
        d_log_mean=float(log_d_ij.mean()), d_log_std=float(log_d_ij.std()),
        om_mean=float(om_avg_t.mean()), om_std=float(om_avg_t.std()),
    )
    print(f"Saved training_aux_v23.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-2)
    args = p.parse_args()
    main(args)
