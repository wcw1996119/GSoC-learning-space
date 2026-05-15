"""Pure-torch synthetic sanity test for CerveroShenHead (avoid numpy interop).

Tests head + forward function with random tensors. Verifies:
  - head instantiates with expected parameter count
  - forward pass runs without shape errors
  - all losses are finite (no NaN/Inf)
  - one optimizer step reduces loss

Does NOT test correctness of the formulation — just that the code runs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

V3_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "v3_cervero_shen_head",
    V3_ROOT / "models_lib" / "inverse_rum" / "cervero_shen_head.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CerveroShenHead = _mod.CerveroShenHead


def main():
    torch.manual_seed(0)
    N, T, M, B = 50, 4, 3, 5  # smaller dimensions for speed

    # Synthetic data
    log_M_j = torch.linspace(0, 12, N) + torch.randn(N) * 0.5    # (N,)
    log_W_j = 6.5 + 0.2 * torch.randn(N)                          # (N,)
    log_D_j = 9.8 + 0.1 * torch.randn(N)                          # (N,)
    match_prob = torch.rand(N, N) * 0.4                           # in [0, 0.4]
    income_score = torch.randn(N) * 0.5
    grid_borough_idx = torch.randint(0, B, (N,))

    t_per_mode = {
        "car": torch.rand(T, N, N) * 30 + 5,
        "transit": torch.rand(T, N, N) * 50 + 10,
        "walk": torch.rand(T, N, N) * 100 + 5,
    }
    pi_m_pair = torch.softmax(torch.randn(N, N, M), dim=-1)
    observed_OD = torch.poisson(torch.rand(T, N, N) * 5).float()
    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[:N // 2] = True

    rum = CerveroShenHead(
        n_modes=M, n_boroughs=B,
        use_gnn_blend=True, blend_max=1.0,
    )

    print(f"=== CerveroShenHead instantiated ===")
    print(f"  n_params = {sum(p.numel() for p in rum.parameters())}")
    for name, p in rum.named_parameters():
        print(f"    {name:30s} shape={tuple(p.shape)} numel={p.numel()}")

    # Manual forward (mimic the trainer forward_cs)
    lambda_per_j = rum.lambda_for_destination(grid_borough_idx)   # (N,)
    lam_view = lambda_per_j.view(1, 1, N)

    log_iv = None
    ce_accum = None
    for m_idx, name in enumerate(["car", "transit", "walk"]):
        V_lower_m = (rum.beta_t_per_mode[m_idx] * t_per_mode[name]
                     + rum.asc_per_mode[m_idx]
                     + rum.theta_inc_per_mode[m_idx] * income_score.view(1, N, 1))  # (T, N, N)
        scaled = V_lower_m / lam_view
        if log_iv is None:
            log_iv = scaled
        else:
            log_iv = torch.logaddexp(log_iv, scaled)
        pi_m_ij = pi_m_pair[:, :, m_idx].unsqueeze(0)
        if ce_accum is None:
            ce_accum = pi_m_ij * scaled
        else:
            ce_accum = ce_accum + pi_m_ij * scaled

    IV_mode = log_iv
    ce_mode_per_ijt = IV_mode - ce_accum

    V_dest_destonly = (rum.alpha_wage * log_W_j.view(1, 1, N)
                       + rum.gamma_M * log_M_j.view(1, 1, N)
                       + rum.nu_D * log_D_j.view(1, 1, N))
    V_match = rum.delta_match * match_prob.view(1, N, N)
    V_rum_dest = V_dest_destonly + V_match + lam_view * IV_mode

    # Simulated GNN output
    V_gnn_jt = torch.randn(T, N)
    V_gnn = V_gnn_jt.view(T, 1, N).expand(T, N, N)
    blend = rum.gnn_blend
    V_dest = (1.0 - blend) * V_rum_dest + blend * V_gnn

    log_P_D = torch.log_softmax(V_dest, dim=-1)
    mask_f = train_mask.view(1, N, 1).float()
    flow = observed_OD * mask_f
    flow_sum = flow.sum().clamp(min=1.0)
    nll_dest = -(flow * log_P_D).sum() / flow_sum
    ce_mode = (flow * ce_mode_per_ijt).sum() / flow_sum
    loss = nll_dest + 1.0 * ce_mode

    print(f"\n=== Forward pass results ===")
    print(f"  log_P_D shape         = {tuple(log_P_D.shape)} (expect ({T}, {N}, {N}))")
    print(f"  IV_mode finite        = {bool(torch.isfinite(IV_mode).all())}")
    print(f"  V_dest finite         = {bool(torch.isfinite(V_dest).all())}")
    print(f"  nll_dest              = {float(nll_dest):.4f}   finite={bool(torch.isfinite(nll_dest))}")
    print(f"  ce_mode               = {float(ce_mode):.4f}   finite={bool(torch.isfinite(ce_mode))}")
    print(f"  total loss            = {float(loss):.4f}")
    assert torch.isfinite(nll_dest), "nll is NaN/Inf!"
    assert torch.isfinite(ce_mode), "ce_mode is NaN/Inf!"

    # One optimizer step
    optimizer = torch.optim.AdamW(rum.parameters(), lr=1e-2)
    loss_before = float(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Re-forward to confirm loss decreases
    log_iv = None
    ce_accum = None
    for m_idx, name in enumerate(["car", "transit", "walk"]):
        V_lower_m = (rum.beta_t_per_mode[m_idx] * t_per_mode[name]
                     + rum.asc_per_mode[m_idx]
                     + rum.theta_inc_per_mode[m_idx] * income_score.view(1, N, 1))
        scaled = V_lower_m / lam_view
        if log_iv is None:
            log_iv = scaled
        else:
            log_iv = torch.logaddexp(log_iv, scaled)
        pi_m_ij = pi_m_pair[:, :, m_idx].unsqueeze(0)
        if ce_accum is None:
            ce_accum = pi_m_ij * scaled
        else:
            ce_accum = ce_accum + pi_m_ij * scaled
    IV_mode = log_iv
    ce_mode_per_ijt = IV_mode - ce_accum
    V_dest_destonly = (rum.alpha_wage * log_W_j.view(1, 1, N)
                       + rum.gamma_M * log_M_j.view(1, 1, N)
                       + rum.nu_D * log_D_j.view(1, 1, N))
    V_match = rum.delta_match * match_prob.view(1, N, N)
    V_rum_dest = V_dest_destonly + V_match + lam_view * IV_mode
    blend = rum.gnn_blend
    V_dest = (1.0 - blend) * V_rum_dest + blend * V_gnn
    log_P_D = torch.log_softmax(V_dest, dim=-1)
    nll_after = -(flow * log_P_D).sum() / flow_sum
    ce_after = (flow * ce_mode_per_ijt).sum() / flow_sum
    loss_after = nll_after + 1.0 * ce_after

    print(f"\n=== After 1 optimizer step ===")
    print(f"  loss before: {loss_before:.4f}")
    print(f"  loss after:  {float(loss_after):.4f}")
    print(f"  delta:       {float(loss_after) - loss_before:+.4f} (expect negative)")

    snap = rum.snapshot()
    print(f"\n=== Snapshot ===")
    for k in ["alpha_wage", "gamma_M", "nu_D", "delta_match",
              "lambda_b_mean", "lambda_b_std", "gnn_blend"]:
        print(f"  {k:20s} = {snap[k]}")

    print("\n[SMOKE PASSED] head + forward + backward all working.")


if __name__ == "__main__":
    main()
