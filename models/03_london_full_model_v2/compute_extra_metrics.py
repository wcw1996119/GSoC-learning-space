"""Re-run the 4 × 3 ablation trials and capture extra OD metrics on the test mask.

The original ``run_ablations.py`` only stored CPC/NLL summaries and did not
checkpoint per-trial utility weights, so we have to re-train. Each trial is
fast (~30-60s on CPU) and the training procedure is fully deterministic given
the seed, so re-running reproduces the original CPC values exactly while
allowing us to compute MAE / RMSE / Spearman / KL on the predicted hourly
F_ij^t at the best-validation-NLL checkpoint.

Outputs:
  evaluation_outputs/ablation_results_extra.csv   (per-trial)
  evaluation_outputs/ablation_results_extra_agg.csv (mean ± std per config)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from models_lib.heterogeneous_utility import HeterogeneousUtility
from metrics_extra import all_metrics
from run_ablations import ABLATIONS, SEEDS, EPOCHS, LR, build_aux


def train_one_with_metrics(aux, config, seed):
    """Train one trial; at the best-val-NLL epoch, also record extra metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    T, N = aux["T"], aux["N"]
    V_norm = aux["V_norm"] if config["gnn"] else torch.zeros_like(aux["V_norm"])
    om_norm = aux["om_norm"] if config["occ"] else torch.zeros_like(aux["om_norm"])
    beta = aux["beta_origin"] if config["hetero"] else aux["beta_uniform"]

    util = HeterogeneousUtility(alpha_V_init=0.5, alpha_occ_init=0.7, gamma_init=0.5)
    optim = torch.optim.Adam(util.parameters(), lr=LR)

    train_mask = aux["train_mask"]
    val_mask = aux["val_mask"]
    test_mask = aux["test_mask"]
    F_ij_t = aux["F_ij_t"]
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

    def predict_F_full():
        """Build (T, N, N) predicted-flow tensor at current weights."""
        with torch.no_grad():
            pred = torch.zeros_like(F_ij_t)
            for h in range(T):
                log_p_h = per_hour_log_p(h)
                P_h = log_p_h.exp()
                origin_total_h = F_ij_t[h].sum(dim=1, keepdim=True)
                pred[h] = P_h * origin_total_h
        return pred

    best_val_nll = float("inf")
    best_metrics_test = None
    best_metrics_val = None
    best_val_cpc = 0.0
    best_test_cpc = 0.0

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
            cpc_v /= cpc_den_val
            cpc_te /= cpc_den_test

        if nll_val < best_val_nll:
            best_val_nll = nll_val
            best_val_cpc = cpc_v
            best_test_cpc = cpc_te
            pred_F = predict_F_full()
            best_metrics_test = all_metrics(pred_F, F_ij_t, mask=test_mask)
            best_metrics_val = all_metrics(pred_F, F_ij_t, mask=val_mask)

    out = {
        "best_val_nll": best_val_nll,
        "best_val_cpc": best_val_cpc,
        "best_test_cpc": best_test_cpc,
        "alpha_V": util.alpha_V.item(),
        "alpha_occ": util.alpha_occ.item(),
        "gamma": util.gamma.item(),
        "bias": util.bias.item(),
    }
    for k, v in best_metrics_test.items():
        out[f"test_{k}"] = v
    for k, v in best_metrics_val.items():
        out[f"val_{k}"] = v
    return out


def main():
    print("Loading shared inputs...")
    aux = build_aux()
    print(f"T={aux['T']}, N={aux['N']}")

    print(f"\n=== Running {len(ABLATIONS)} configs × {len(SEEDS)} seeds = "
          f"{len(ABLATIONS) * len(SEEDS)} trials (with extra metrics) ===")
    rows = []
    overall_t0 = time.time()
    for config_name, config in ABLATIONS.items():
        for seed in SEEDS:
            t0 = time.time()
            res = train_one_with_metrics(aux, config, seed)
            elapsed = time.time() - t0
            print(
                f"  [{config_name:18s} seed={seed:5d}] "
                f"CPC test={res['best_test_cpc']:.3f} | "
                f"MAE={res['test_mae']:.3f} RMSE={res['test_rmse']:.3f} "
                f"ρ={res['test_spearman']:.3f} KL={res['test_kl']:.3f} | "
                f"{elapsed:.1f}s"
            )
            rows.append({"config": config_name, "seed": seed, **res})
    print(f"\nTotal time: {(time.time() - overall_t0):.1f}s")

    df = pd.DataFrame(rows)
    out_csv = V2_ROOT / "evaluation_outputs" / "ablation_results_extra.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    agg_cols = ["best_test_cpc", "test_mae", "test_rmse", "test_spearman", "test_kl"]
    agg = df.groupby("config")[agg_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    out_agg = V2_ROOT / "evaluation_outputs" / "ablation_results_extra_agg.csv"
    agg.to_csv(out_agg, index=False)
    print(f"Saved {out_agg}")

    print("\n=== Mean ± std across seeds (test set) ===")
    for _, r in agg.iterrows():
        print(f"  {r['config']:18s} "
              f"CPC={r['best_test_cpc_mean']:.3f}±{r['best_test_cpc_std']:.3f}  "
              f"MAE={r['test_mae_mean']:.3f}±{r['test_mae_std']:.3f}  "
              f"RMSE={r['test_rmse_mean']:.3f}±{r['test_rmse_std']:.3f}  "
              f"ρ={r['test_spearman_mean']:.3f}±{r['test_spearman_std']:.3f}  "
              f"KL={r['test_kl_mean']:.3f}±{r['test_kl_std']:.3f}")


if __name__ == "__main__":
    main()
