"""B-2b smoke: verify trainer can recover multi-tier β via mixture logit.

Synthesize OD with known (β_low, β_mid, β_high, γ, δ) under the mixture
formulation:
    P(j|i) = Σ_k π_{k|i} · softmax_j( α V_j + β_{t,k} t_ij + γ log d + δ OccMatch )

Train inverse model and check per-tier β recovery.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from models_lib.inverse_rum import (
    InverseRUMTrainer, StructuralGNN, hessian_ci_rum,
)
from experiments.paper_a.synthetic_recovery import (
    load_grid_features, build_edge_index,
)


# Truth: |β| INCREASES with income (high income → time-sensitive, WebTAG style).
BETA_TRUE_PER_TIER = np.array([-0.04, -0.07, -0.10])    # low, mid, high
GAMMA_TRUE = -0.5
DELTA_TRUE = 0.5
ALPHA_TRUE = 0.30


def synthesize_od_mixture(grid_features, t_ij, log_d_ij, edge_index,
                           occ_match, income_tier_props,
                           n_total_trips, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, Fdim = grid_features.shape
    K = income_tier_props.shape[1]

    teacher = StructuralGNN(in_features=Fdim, hidden=64, out=1, depth=3).to(device).eval()
    with torch.no_grad():
        V_j = teacher(grid_features, edge_index).squeeze(-1)            # (N,)

        logits_per_tier = torch.zeros(K, N, N, device=device)
        for k in range(K):
            logits_per_tier[k] = (
                ALPHA_TRUE * V_j.unsqueeze(0)
                + BETA_TRUE_PER_TIER[k] * t_ij
                + GAMMA_TRUE * log_d_ij
                + DELTA_TRUE * occ_match
            )
        log_p_per_tier = torch.log_softmax(logits_per_tier, dim=-1)     # (K, N, N)

        log_pi = torch.log(income_tier_props.clamp(min=1e-30))          # (N, K)
        log_pi_b = log_pi.t().view(K, N, 1)                              # (K, N, 1)
        log_p = torch.logsumexp(log_p_per_tier + log_pi_b, dim=0)       # (N, N)
        P_true = log_p.exp()

    origin_mass = torch.full((N,), n_total_trips / N, device=device)
    F_mean = origin_mass.unsqueeze(1) * P_true
    F_obs = torch.poisson(F_mean.clamp(min=0.0))
    return F_obs


def main():
    device = "cpu"
    print("=" * 60)
    print("  B-2b smoke: multi-tier β recovery test")
    print("=" * 60)
    print(f"Truth: β_low={BETA_TRUE_PER_TIER[0]}, "
          f"β_mid={BETA_TRUE_PER_TIER[1]}, β_high={BETA_TRUE_PER_TIER[2]}")
    print(f"       γ={GAMMA_TRUE}, δ={DELTA_TRUE}, α={ALPHA_TRUE}")

    grid_features, t_ij, log_d_ij = load_grid_features(device)
    edge_index = build_edge_index(t_ij, k=10)
    aux = np.load(V2_ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    occ_match = torch.tensor(aux["occ_match"], dtype=torch.float32, device=device)
    income_tier_props = torch.tensor(aux["income_tier_props"], dtype=torch.float32, device=device)
    K = income_tier_props.shape[1]
    print(f"\nN={grid_features.shape[0]}, K={K} (income tiers)")
    print(f"Income tier global proportion: "
          f"{income_tier_props.mean(dim=0).cpu().numpy()}")

    for noise_label, n_trips in [("low", 1_000_000), ("medium", 100_000)]:
        print(f"\n--- {noise_label} (N_trips={n_trips:,}) ---")
        F_obs = synthesize_od_mixture(
            grid_features, t_ij, log_d_ij, edge_index,
            occ_match, income_tier_props,
            n_total_trips=n_trips, seed=0, device=device,
        )
        print(f"F_obs.sum = {float(F_obs.sum()):.0f}")

        Fdim = grid_features.shape[-1]
        student = StructuralGNN(in_features=Fdim, hidden=64, out=1, depth=3).to(device)
        trainer = InverseRUMTrainer(
            grid_features=grid_features, edge_index=edge_index,
            observed_OD=F_obs, t_ij_t=t_ij, log_d_ij=log_d_ij,
            K=50, utility_net=student, device=device, seed=0,
            epochs=150, patience=150, residualise=False,
            occ_match=occ_match, income_tier_props=income_tier_props,
        )
        assert trainer._enable_mixture, "mixture not enabled!"
        assert trainer._enable_occ_match, "occ_match not enabled!"
        print(f"trainer mixture={trainer._enable_mixture}, "
              f"occ_match={trainer._enable_occ_match}, K={trainer.n_income_tiers}")

        t0 = time.time()
        theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
        elapsed = time.time() - t0

        # beta_t_hat shape (T, K) = (1, 3)
        beta_per_tier = beta_t_hat[0].cpu().numpy()
        delta_hat = float(trainer.rum.delta.item())
        gamma_val = float(gamma_hat)

        print(f"\nElapsed: {elapsed:.1f}s")
        print("Recovery:")
        for k, label in enumerate(["low", "mid", "high"]):
            truth = BETA_TRUE_PER_TIER[k]
            hat = float(beta_per_tier[k])
            bias_pct = (hat - truth) / abs(truth) * 100
            print(f"  beta_{label:>4}: truth={truth:+.4f}, "
                  f"hat={hat:+.4f}, bias={bias_pct:+.1f}%")
        print(f"  gamma:    truth={GAMMA_TRUE:+.4f}, hat={gamma_val:+.4f}, "
              f"bias={(gamma_val - GAMMA_TRUE) / abs(GAMMA_TRUE) * 100:+.1f}%")
        print(f"  delta:    truth={DELTA_TRUE:+.4f}, hat={delta_hat:+.4f}, "
              f"bias={(delta_hat - DELTA_TRUE) / abs(DELTA_TRUE) * 100:+.1f}%")

        try:
            hci = hessian_ci_rum(trainer, confidence=0.95)
            print("\nHessian 95% CIs:")
            for k, label in enumerate(["low", "mid", "high"]):
                lo, hi = hci["beta_mean_per_tier_ci"][k]
                in_ci = lo <= BETA_TRUE_PER_TIER[k] <= hi
                print(f"  beta_{label:>4}: [{lo:+.4f}, {hi:+.4f}], in_CI? {in_ci}")
            if "delta_ci" in hci:
                lo, hi = hci["delta_ci"]
                in_ci = lo <= DELTA_TRUE <= hi
                print(f"  delta:   [{lo:+.4f}, {hi:+.4f}], in_CI? {in_ci}")
        except Exception as e:
            print(f"\nHessian CI failed: {e}")


if __name__ == "__main__":
    main()
