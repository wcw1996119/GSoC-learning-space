"""Smoke test for irm_penalty + trainer integration. Run locally on CPU
to catch import / shape / autograd issues before Colab GPU run.

Not part of paper pipeline — delete after sanity check passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models_lib.inverse_rum.irm_penalty import (
    compute_rex_penalty, compute_irmv1_penalty, warmup_lambda,
)


def test_rex():
    nll = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 1.8], requires_grad=True)
    env_idx = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = torch.tensor([True] * 6)
    p = compute_rex_penalty(nll, env_idx, mask, n_envs=3)
    # env_means = [1.5, 2.25, 2.15], var unbiased=False
    expected = torch.tensor([1.5, 2.25, 2.15]).var(unbiased=False).item()
    print(f"[V-REx] penalty={p.item():.4f}  expected={expected:.4f}  "
          f"{'OK' if abs(p.item() - expected) < 1e-4 else 'FAIL'}")

    # gradient flows back
    p.backward()
    print(f"[V-REx] nll.grad first 3: {nll.grad[:3].tolist()}  (non-zero means autograd works)")


def test_irmv1():
    nll_unscaled = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 1.8])
    dummy = torch.tensor(1.0, requires_grad=True)
    # Mimic trainer: per_origin_nll depends on dummy_scale
    nll = nll_unscaled * dummy
    env_idx = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = torch.tensor([True] * 6)
    p = compute_irmv1_penalty(nll, env_idx, mask, dummy, n_envs=3)
    # d (env_mean)/d dummy = env_mean_of_unscaled at dummy=1
    # so penalty = mean of [1.5^2, 2.25^2, 2.15^2] = (2.25 + 5.0625 + 4.6225)/3 = 3.978
    expected = (1.5**2 + 2.25**2 + 2.15**2) / 3
    print(f"[IRM-v1] penalty={p.item():.4f}  expected={expected:.4f}  "
          f"{'OK' if abs(p.item() - expected) < 1e-3 else 'FAIL'}")


def test_warmup():
    assert warmup_lambda(0, 50, 10) == 0.0
    assert warmup_lambda(49, 50, 10) == 0.0
    assert warmup_lambda(50, 50, 10) == 10.0
    assert warmup_lambda(199, 50, 10) == 10.0
    assert warmup_lambda(0, 0, 10) == 10.0  # no warmup
    print("[warmup] all assertions OK")


def test_trainer_irm_construct():
    """Build trainer with IRM enabled — checks __init__ wiring without training."""
    from models_lib.inverse_rum.dual_branch_mixture_trainer import DualBranchMixtureTrainer

    N, T, F_s, F_d, M, K = 20, 4, 5, 3, 3, 3
    X_static = torch.randn(N, F_s)
    X_dynamic = torch.randn(T, N, F_d)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    F_ij = torch.rand(T, N, N).clamp(min=0.1) * 10
    t_per_mode = {
        "car": torch.rand(T, N, N) * 30,
        "transit": torch.rand(T, N, N) * 40,
        "walk": torch.rand(T, N, N) * 60,
    }
    ms = torch.rand(N, M) + 0.1
    ms = ms / ms.sum(dim=1, keepdim=True)
    ip = torch.rand(N, K) + 0.1
    ip = ip / ip.sum(dim=1, keepdim=True)
    log_d = torch.rand(N, N) * 2 + 0.1
    occ = torch.randn(N, N)
    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[:14] = True
    val_mask = torch.zeros(N, dtype=torch.bool); val_mask[14:] = True
    env_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])

    for variant, lam in [("rex", 1.0), ("irmv1", 1.0)]:
        trainer = DualBranchMixtureTrainer(
            X_static=X_static, X_dynamic=X_dynamic, edge_index=edge_index,
            observed_OD=F_ij, t_ij_per_mode=t_per_mode,
            mode_share_per_origin=ms, income_tier_props=ip,
            log_d_ij=log_d, occ_match=occ,
            train_mask=train_mask, val_mask=val_mask,
            device="cpu", seed=0, epochs=2, patience=10, verbose=True,
            irm_variant=variant, irm_lambda=lam,
            irm_warmup_epochs=0,  # no warmup — penalty active from ep 0
            origin_env_idx=env_idx,
        )
        print(f"\n[trainer {variant}] fit (2 epochs)...")
        encoder, head, log = trainer.fit()
        print(f"[trainer {variant}] train_nll: {log.train_nll}")
        print(f"[trainer {variant}] val_nll: {log.val_nll}")
        print(f"[trainer {variant}] cpc: {log.cpc_val}  → OK\n")


if __name__ == "__main__":
    print("=" * 60)
    print("V-REx unit test")
    print("=" * 60)
    test_rex()

    print("\n" + "=" * 60)
    print("IRM-v1 unit test")
    print("=" * 60)
    test_irmv1()

    print("\n" + "=" * 60)
    print("warmup_lambda test")
    print("=" * 60)
    test_warmup()

    print("\n" + "=" * 60)
    print("Trainer integration (2 epochs, tiny synthetic)")
    print("=" * 60)
    test_trainer_irm_construct()
