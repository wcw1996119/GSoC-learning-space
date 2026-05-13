"""B-2a smoke: verify trainer can recover δ (OccMatch coefficient).

Synthesize OD with known (β, γ, δ_true) — including the OccMatch term —
and check inverse trainer recovers δ_hat ≈ δ_true.

Sanity check before doing full W4 with multi-tier β (B-2b).
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
    BETA_STAR, ALPHA_STAR, GAMMA_STAR,
    load_grid_features, build_edge_index,
)


DELTA_STAR = 0.5    # truth: δ * OccMatch ranges from 0.08 to 0.48 (since OccMatch ∈ [0.16, 0.97])


def synthesize_od_with_occmatch(grid_features, t_ij, log_d_ij, edge_index, occ_match,
                                 n_total_trips, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, Fdim = grid_features.shape

    # Teacher GNN with frozen random weights
    teacher = StructuralGNN(in_features=Fdim, hidden=64, out=1, depth=3).to(device).eval()

    with torch.no_grad():
        V_j = teacher(grid_features, edge_index).squeeze(-1)  # (N,)
        # Build full logits: alpha V + beta t + gamma log_d + delta * occ_match
        logits = (
            ALPHA_STAR * V_j.unsqueeze(0)
            + BETA_STAR * t_ij
            + GAMMA_STAR * log_d_ij
            + DELTA_STAR * occ_match
        )
        log_p = torch.log_softmax(logits, dim=1)
        P_true = log_p.exp()

    origin_mass = torch.full((N,), n_total_trips / N, device=device)
    F_mean = origin_mass.unsqueeze(1) * P_true
    F_obs = torch.poisson(F_mean.clamp(min=0.0))
    return F_obs


def main():
    device = "cpu"
    print("=" * 60)
    print("  B-2a smoke: trainer δ recovery test")
    print("=" * 60)
    print(f"Truth: β={BETA_STAR}, γ={GAMMA_STAR}, δ={DELTA_STAR}")

    # Load data
    grid_features, t_ij, log_d_ij = load_grid_features(device)
    edge_index = build_edge_index(t_ij, k=10)
    aux = np.load(V2_ROOT / "data" / "processed" / "paperA_v23_aux.npz")
    occ_match = torch.tensor(aux["occ_match"], dtype=torch.float32, device=device)
    print(f"\nLoaded: N={grid_features.shape[0]}, occ_match={tuple(occ_match.shape)}")
    print(f"  occ_match range: [{occ_match.min():.3f}, {occ_match.max():.3f}], "
          f"mean={occ_match.mean():.3f}")

    for noise_label, n_trips in [("low", 1_000_000), ("medium", 100_000)]:
        print(f"\n--- {noise_label} (N_trips={n_trips:,}) ---")
        F_obs = synthesize_od_with_occmatch(
            grid_features, t_ij, log_d_ij, edge_index, occ_match,
            n_total_trips=n_trips, seed=0, device=device,
        )
        print(f"F_obs.sum = {float(F_obs.sum()):.0f}")

        # Train inverse model with OccMatch enabled
        Fdim = grid_features.shape[-1]
        student = StructuralGNN(in_features=Fdim, hidden=64, out=1, depth=3).to(device)
        trainer = InverseRUMTrainer(
            grid_features=grid_features,
            edge_index=edge_index,
            observed_OD=F_obs,
            t_ij_t=t_ij,
            log_d_ij=log_d_ij,
            K=50,
            utility_net=student,
            device=device,
            seed=0,
            epochs=150,
            patience=150,
            residualise=False,
            occ_match=occ_match,    # ⭐ enable OccMatch
        )
        # Sanity: confirm it's enabled
        assert trainer._enable_occ_match, "occ_match did not enable!"
        print(f"trainer._enable_occ_match = True [OK]")

        t0 = time.time()
        theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
        elapsed = time.time() - t0

        delta_hat = float(trainer.rum.delta.item())
        beta_hat = float(beta_t_hat.mean().item())
        gamma_hat_val = float(gamma_hat)

        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Recovery:")
        print(f"  β: truth={BETA_STAR:+.4f}, hat={beta_hat:+.4f}, "
              f"bias={(beta_hat - BETA_STAR) / abs(BETA_STAR) * 100:+.1f}%")
        print(f"  γ: truth={GAMMA_STAR:+.4f}, hat={gamma_hat_val:+.4f}, "
              f"bias={(gamma_hat_val - GAMMA_STAR) / abs(GAMMA_STAR) * 100:+.1f}%")
        print(f"  δ: truth={DELTA_STAR:+.4f}, hat={delta_hat:+.4f}, "
              f"bias={(delta_hat - DELTA_STAR) / abs(DELTA_STAR) * 100:+.1f}%")

        # Hessian CI for δ
        hci = hessian_ci_rum(trainer, confidence=0.95)
        if "delta_hat" in hci:
            print(f"  δ Hessian SE = {hci['delta_se']:.4f}")
            print(f"  δ 95% CI = [{hci['delta_ci'][0]:+.4f}, {hci['delta_ci'][1]:+.4f}]")
            in_ci = hci['delta_ci'][0] <= DELTA_STAR <= hci['delta_ci'][1]
            print(f"  δ_true in CI? {in_ci}")
        else:
            print("  WARNING: hessian_ci did not include delta")


if __name__ == "__main__":
    main()
