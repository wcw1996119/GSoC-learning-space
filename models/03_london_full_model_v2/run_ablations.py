"""Run 4 utility ablations × 3 seeds = 12 training runs on the v2.2 utility.

Ablations:
  full           — STGNN V_j(t) + OccMatch + heterogeneous β  (baseline)
  no_gnn         — V_j replaced by zeros (gravity-style: only β·t and γ·log d)
  no_occmatch    — OccMatch replaced by zeros (α_occ inactive)
  no_heterogeneity — single β=0.07 for all agents (no income×mode lookup)

Outputs:
  evaluation_outputs/ablation_results.csv
  evaluation_outputs/ablation_plot.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.heterogeneous_utility import HeterogeneousUtility, compute_beta_per_origin
from models_lib.occupation_match import (
    aggregate_occupation_match, grid_industry_vector
)

ABLATIONS = {
    "full":             {"gnn": True,  "occ": True,  "hetero": True},
    "no_gnn":           {"gnn": False, "occ": True,  "hetero": True},
    "no_occmatch":      {"gnn": True,  "occ": False, "hetero": True},
    "no_heterogeneity": {"gnn": True,  "occ": True,  "hetero": False},
}

SEEDS = [42, 123, 2024]
EPOCHS = 30
LR = 5e-2


def build_aux():
    """Load and prepare common inputs (same for all configs/seeds)."""
    cache = dict(np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz", allow_pickle=True))
    V_jt = torch.tensor(cache["V_jt_baseline"], dtype=torch.float32)
    t_ij_t = torch.tensor(cache["t_ij_t"], dtype=torch.float32)
    F_ij_t = torch.tensor(cache["F_ij_t"], dtype=torch.float32)
    coords_bng = cache["coords_bng"]
    train_mask = torch.tensor(cache["train_mask"])
    val_mask = torch.tensor(cache["val_mask"])
    test_mask = torch.tensor(cache["test_mask"])
    T, N = V_jt.shape

    diff = coords_bng[:, None, :] - coords_bng[None, :, :]
    d_km = np.sqrt((diff ** 2).sum(axis=-1)) / 1000.0
    log_d_ij = torch.tensor(np.log1p(d_km), dtype=torch.float32)

    grid_feats = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_static_features.csv")
    grid_msoa = pd.read_csv(V2_ROOT / "data" / "processed" / "grid_msoa_primary.csv")
    occupation = pd.read_csv(
        V2_ROOT.parent / "02_london_commuting_model" / "data" / "processed" / "london_occupation_msoa.csv"
    )
    soc_cols = [f"prop_soc{i}" for i in range(1, 10)]
    msoa_soc = occupation[["MSOA21CD"] + soc_cols].copy()
    df = grid_msoa[["grid_id", "MSOA21CD"]].merge(msoa_soc, on="MSOA21CD", how="left")
    df = df.set_index("grid_id").reindex(grid_feats["grid_id"].values).reset_index()
    soc_props = df[soc_cols].fillna(1.0 / 9).values
    soc_props = soc_props / (soc_props.sum(axis=1, keepdims=True) + 1e-12)
    gx = grid_industry_vector(grid_feats)
    om_avg = aggregate_occupation_match(gx, soc_props)
    om_avg_t = torch.tensor(om_avg, dtype=torch.float32)

    agents_df = pd.read_csv(V2_ROOT / "data" / "processed" / "agent_population.csv")
    beta_origin = compute_beta_per_origin(agents_df, n_grids=N)

    # Z-score normalize features (same for all configs)
    V_norm = (V_jt - V_jt.mean()) / (V_jt.std() + 1e-8)
    t_log = torch.log1p(t_ij_t)
    t_norm = (t_log - t_log.mean()) / (t_log.std() + 1e-8)
    d_norm = (log_d_ij - log_d_ij.mean()) / (log_d_ij.std() + 1e-8)
    om_norm = (om_avg_t - om_avg_t.mean()) / (om_avg_t.std() + 1e-8)

    t_log_std_val = float(t_log.std())
    beta_origin_for_training = beta_origin * t_log_std_val
    # Single β for "no_heterogeneity": uniform = mid×car × t_log_std
    beta_uniform = torch.full((N,), 0.07 * t_log_std_val, dtype=torch.float32)

    return {
        "V_norm": V_norm, "t_norm": t_norm, "d_norm": d_norm, "om_norm": om_norm,
        "F_ij_t": F_ij_t, "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "beta_origin": beta_origin_for_training, "beta_uniform": beta_uniform,
        "T": T, "N": N,
    }


def train_one(aux, config, seed, verbose=False):
    """Run one training trial. Returns dict with final val/test CPC + best val NLL.

    True multi-seed: init values perturbed by Gaussian noise per seed → real std.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    T, N = aux["T"], aux["N"]
    V_norm = aux["V_norm"] if config["gnn"] else torch.zeros_like(aux["V_norm"])
    om_norm = aux["om_norm"] if config["occ"] else torch.zeros_like(aux["om_norm"])
    beta = aux["beta_origin"] if config["hetero"] else aux["beta_uniform"]

    # Per-seed random init perturbation (σ=0.1) around literature priors
    util = HeterogeneousUtility(
        alpha_V_init=max(0.05, 0.5 + float(rng.normal(0, 0.1))),
        alpha_occ_init=max(0.05, 0.7 + float(rng.normal(0, 0.1))),
        gamma_init=max(0.05, 0.5 + float(rng.normal(0, 0.1))),
    )
    optim = torch.optim.Adam(util.parameters(), lr=LR)

    train_mask, val_mask, test_mask = aux["train_mask"], aux["val_mask"], aux["test_mask"]
    F_ij_t = aux["F_ij_t"]
    cpc_den_train = float(F_ij_t[:, train_mask].sum() + 1e-8)
    cpc_den_val = float(F_ij_t[:, val_mask].sum() + 1e-8)
    cpc_den_test = float(F_ij_t[:, test_mask].sum() + 1e-8)

    def per_hour_log_p(t_idx):
        Vj_b = V_norm[t_idx][None, :].expand(N, N)
        beta_b = beta.unsqueeze(1).expand(N, N)
        V_ij = util.utility_aggregate(
            V_j_t=Vj_b, occ_match_avg=om_norm,
            t_ij_t=aux["t_norm"][t_idx], log_d_ij=aux["d_norm"],
            beta_origin=beta_b,
        )
        return torch.log_softmax(V_ij, dim=1)

    best_val_nll = float("inf")
    best_test_cpc = 0.0
    best_val_cpc = 0.0
    for epoch in range(EPOCHS):
        util.train()
        optim.zero_grad()
        nll_total = torch.tensor(0.0)
        active = 0
        for h in range(T):
            log_p_h = per_hour_log_p(h)
            flow_h = F_ij_t[h] * train_mask.view(-1, 1).float()
            nll_total = nll_total + (-(flow_h * log_p_h).sum())
            active += int((flow_h.sum(dim=1) > 0).sum())
        nll_train = nll_total / max(active, 1)

        # Prior on alpha_occ (only relevant when OccMatch is on)
        prior_loss = 0.05 * (util.alpha_occ - 1.0).pow(2) if config["occ"] else 0.0
        loss = nll_train + prior_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(util.parameters(), max_norm=1.0)
        optim.step()

        util.eval()
        with torch.no_grad():
            nll_val_total = torch.tensor(0.0)
            active_val = 0
            cpc_v, cpc_te = 0.0, 0.0
            for h in range(T):
                log_p_h = per_hour_log_p(h)
                flow_v = F_ij_t[h] * val_mask.view(-1, 1).float()
                nll_val_total = nll_val_total + (-(flow_v * log_p_h).sum())
                active_val += int((flow_v.sum(dim=1) > 0).sum())
                P_h = log_p_h.exp()
                origin_total_h = F_ij_t[h].sum(dim=1, keepdim=True)
                pred_F_h = P_h * origin_total_h
                cpc_v += float(torch.minimum(pred_F_h[val_mask], F_ij_t[h][val_mask]).sum())
                cpc_te += float(torch.minimum(pred_F_h[test_mask], F_ij_t[h][test_mask]).sum())
            nll_val = (nll_val_total / max(active_val, 1)).item()
            cpc_v = cpc_v / cpc_den_val
            cpc_te = cpc_te / cpc_den_test

        if nll_val < best_val_nll:
            best_val_nll = nll_val
            best_val_cpc = cpc_v
            best_test_cpc = cpc_te

        if verbose:
            print(f"  ep{epoch:2d} train {nll_train:.3f} val {nll_val:.3f} cpc_v {cpc_v:.3f} cpc_te {cpc_te:.3f}")

    return {
        "best_val_nll": best_val_nll,
        "best_val_cpc": best_val_cpc,
        "best_test_cpc": best_test_cpc,
        "alpha_V": util.alpha_V.item(),
        "alpha_occ": util.alpha_occ.item(),
        "gamma": util.gamma.item(),
        "bias": util.bias.item(),
    }


def main():
    print("Loading shared inputs...")
    aux = build_aux()
    print(f"T={aux['T']}, N={aux['N']}")

    print(f"\n=== Running {len(ABLATIONS)} configs × {len(SEEDS)} seeds = {len(ABLATIONS)*len(SEEDS)} runs ===")
    rows = []
    overall_t0 = time.time()
    for config_name, config in ABLATIONS.items():
        for seed in SEEDS:
            t0 = time.time()
            res = train_one(aux, config, seed)
            elapsed = time.time() - t0
            print(
                f"  [{config_name:18s} seed={seed:5d}] "
                f"val NLL={res['best_val_nll']:.3f} | "
                f"val CPC={res['best_val_cpc']:.3f} | test CPC={res['best_test_cpc']:.3f} | "
                f"αV={res['alpha_V']:.2f} αocc={res['alpha_occ']:.2f} γ={res['gamma']:.2f} | "
                f"{elapsed:.1f}s"
            )
            rows.append({"config": config_name, "seed": seed, **res})
    print(f"\nTotal time: {(time.time() - overall_t0):.1f}s")

    df = pd.DataFrame(rows)
    out_csv = V2_ROOT / "evaluation_outputs" / "ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Aggregate mean ± std
    agg = df.groupby("config").agg(
        val_cpc_mean=("best_val_cpc", "mean"),
        val_cpc_std=("best_val_cpc", "std"),
        test_cpc_mean=("best_test_cpc", "mean"),
        test_cpc_std=("best_test_cpc", "std"),
        val_nll_mean=("best_val_nll", "mean"),
    ).reset_index()
    print("\n=== Mean ± std across seeds ===")
    print(agg.to_string(index=False))

    # Bar plot
    order = ["full", "no_gnn", "no_occmatch", "no_heterogeneity"]
    agg = agg.set_index("config").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    width = 0.35
    bars1 = ax.bar(x - width/2, agg["val_cpc_mean"], width,
                   yerr=agg["val_cpc_std"], label="Val CPC", capsize=5, color="#1976d2")
    bars2 = ax.bar(x + width/2, agg["test_cpc_mean"], width,
                   yerr=agg["test_cpc_std"], label="Test CPC", capsize=5, color="#388e3c")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel("CPC")
    ax.set_title(f"Ablation study: {len(SEEDS)} seeds, {EPOCHS} epochs each")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean, std in zip(bars1, agg["val_cpc_mean"], agg["val_cpc_std"]):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.005,
                f"{mean:.3f}", ha="center", fontsize=9)
    for bar, mean, std in zip(bars2, agg["test_cpc_mean"], agg["test_cpc_std"]):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.005,
                f"{mean:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_png = V2_ROOT / "evaluation_outputs" / "ablation_plot.png"
    plt.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
