"""Tests for models_lib.inverse_rum.

Run:
    python -m pytest tests/test_inverse_rum.py -q

The smoke + identifiability tests train on tiny synthetic data so they
finish in a few seconds on CPU.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from models_lib.inverse_rum import (
    ImplicitSoftmax,
    implicit_softmax,
    nested_logit_logsum,
    StructuralGNN,
    TopKChoiceSet,
    make_choice_set,
    InverseRUMTrainer,
)


# --------------------------------------------------------------- implicit softmax

def test_implicit_softmax_matches_torch_softmax():
    torch.manual_seed(0)
    z = torch.randn(4, 8, dtype=torch.float64)
    p_imp = implicit_softmax(z, tau=1.0)
    p_ref = torch.softmax(z, dim=-1)
    assert torch.allclose(p_imp, p_ref, atol=1e-10)


def test_implicit_softmax_gradcheck():
    torch.manual_seed(0)
    z = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(1.0, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda zz: ImplicitSoftmax.apply(zz, tau), (z,), eps=1e-6, atol=1e-4
    )


def test_implicit_softmax_temperature():
    z = torch.tensor([[1.0, 2.0, 3.0]])
    p_low = implicit_softmax(z, tau=0.1)   # peaked
    p_high = implicit_softmax(z, tau=50.0)  # very flat
    assert p_low[0, 2] > 0.95
    assert (p_high[0].max() - p_high[0].min()).item() < 0.02


def test_nested_logit_consistency():
    # alternatives 0,1 in nest 0; alternatives 2,3 in nest 1
    z = torch.tensor([[1.0, 0.5, 2.0, 1.5]])
    nest_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    log_p = nested_logit_logsum(z, nest_idx, n_nests=2,
                                tau_within=1.0, tau_between=1.0)
    p = log_p.exp()
    # probabilities sum to 1
    assert abs(p.sum().item() - 1.0) < 1e-6
    # tau_within == tau_between collapses to plain softmax
    p_flat = torch.softmax(z, dim=-1)
    assert torch.allclose(p, p_flat, atol=1e-5)


# --------------------------------------------------------------- structural GNN

def _line_graph_edges(n: int) -> torch.Tensor:
    """0-1-2-...-(n-1) chain (undirected)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_structural_gnn_shape():
    torch.manual_seed(0)
    N, T, F = 10, 3, 4
    gnn = StructuralGNN(node_dim=F, hidden_dim=8, n_layers=2, gru_hidden=4)
    X = torch.randn(T, N, F)
    edge_index = _line_graph_edges(N)
    V = gnn(X, edge_index)
    assert V.shape == (T, N)


def test_structural_gnn_intervention_locality():
    """Intervening on node 0 should NOT affect a node far away on the chain
    (modulo the global normalisation applied at the end of forward).

    With 2 SAGE layers the receptive field is 2 hops, so node 5 on a 10-node
    chain (3 hops away from node 0) must be unaffected by a do(X_0) clamp on
    the *pre-normalisation* features. The forward pass now applies a
    zero-mean/unit-std rescaling globally, so an intervention on node 0
    propagates to far nodes via the global mean/std — but the *relative*
    pattern across far nodes (i.e. V_do[far] reproduces V_base[far] up to a
    global affine transform) must be preserved.
    """
    torch.manual_seed(0)
    N, T, F = 10, 2, 3
    gnn = StructuralGNN(node_dim=F, hidden_dim=8, n_layers=2, gru_hidden=4)
    gnn.eval()
    X = torch.randn(T, N, F)
    edge_index = _line_graph_edges(N)
    V_base = gnn(X, edge_index)

    mask = torch.zeros(N, F, dtype=torch.bool); mask[0, :] = True
    new_val = torch.zeros(N, F); new_val[0, :] = 99.0
    V_do = gnn.forward_under_intervention(X, edge_index, mask, new_val)

    # node 0..2 should differ (within k-hop)
    diff_local = (V_do[:, :3] - V_base[:, :3]).abs().max()
    assert diff_local > 1e-3

    # node 5..9 should differ from the base only by a global affine — i.e.
    # V_do[far] = a * V_base[far] + b for some scalars (a, b) per time step.
    # Equivalently, after re-standardising the far slice, the two should match.
    def _restd(z):
        z = z - z.mean()
        return z / (z.std() + 1e-8)
    for t in range(T):
        z_base = _restd(V_base[t, 5:])
        z_do = _restd(V_do[t, 5:])
        assert torch.allclose(z_base, z_do, atol=1e-4), (
            f"far slice at t={t} not affine-invariant: "
            f"max diff = {(z_base - z_do).abs().max().item():.2e}"
        )


# --------------------------------------------------------------- top-K choice

def test_topk_reproducibility():
    torch.manual_seed(0)
    V = torch.randn(50)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    idx1, _ = make_choice_set(0, V, K=20, rng=rng1, chosen_idx=7)
    idx2, _ = make_choice_set(0, V, K=20, rng=rng2, chosen_idx=7)
    assert torch.equal(idx1, idx2)
    assert (idx1 == 7).any().item(), "chosen alternative must be present"


def test_topk_correction_uniform_q():
    """When q is uniform, the McFadden correction term is constant across
    alternatives, so the *relative* corrected log-probabilities recover the
    full-universe softmax (up to a constant).

    Concretely: with q uniform, log q(j) = -log N for every j in C, and
    log P(j|C) - log q = log P(j|C) + log N, so the *softmax over corrected
    terms inside the choice set* equals the softmax restricted to C.  We
    verify that the relative ordering/values match across many random
    choice sets.
    """
    rng = np.random.default_rng(0)
    N, K = 30, 10
    torch.manual_seed(0)
    logits = torch.randn(N)

    # uniform q: pass zero-vector as prior — all alts equally likely
    uniform_prior = torch.zeros(N)

    helper = TopKChoiceSet(K=K, rng=rng, include_chosen=False)
    rel_errs = []
    for _ in range(500):
        idx, log_q = helper.sample(0, uniform_prior)
        # log_q under uniform prior is -log(N) for every j
        assert torch.allclose(log_q, log_q[0] * torch.ones_like(log_q), atol=1e-6)
        # corrected log-probs (within choice set) — softmax over them
        # should equal softmax of the original logits restricted to idx
        sub_logits = logits[idx]
        log_p_in_set = torch.log_softmax(sub_logits, dim=-1)
        corrected = helper.correction_logp(log_p_in_set, log_q)
        # softmax of corrected = softmax(log_p_in_set + const) = softmax(log_p_in_set)
        p_corrected = torch.softmax(corrected, dim=-1)
        p_in_set = torch.softmax(sub_logits, dim=-1)
        rel_errs.append(float((p_corrected - p_in_set).abs().max()))

    assert max(rel_errs) < 1e-5, f"max rel err {max(rel_errs)}"


def test_topk_correction_returns_correct_shape():
    """correction_logp returns same-shape tensor as inputs."""
    helper = TopKChoiceSet(K=10, rng=np.random.default_rng(0))
    log_p = torch.randn(10)
    log_q = torch.randn(10)
    out = helper.correction_logp(log_p, log_q)
    assert out.shape == log_p.shape
    assert torch.allclose(out, log_p - log_q)


# --------------------------------------------------------------- inverse trainer

# B5 sign convention: trainer reparam is beta_t = -softplus(raw), always
# negative. Synthetic generators must use a *negative* beta_true so the
# identifiability check compares like with like.
BETA_T_STAR = -0.07


def _make_synthetic(N=20, T=3, n_pairs=1000, beta_true=BETA_T_STAR, alpha_true=0.6,
                    gamma_true=-0.4, seed=0):
    """Generate synthetic OD flows from a known data-generating process."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    F_dim = 4
    # node features
    X_static = torch.randn(N, F_dim)
    X = X_static.unsqueeze(0).expand(T, N, F_dim).clone()
    # add a bit of hour variation
    X = X + 0.05 * torch.randn(T, N, F_dim)

    # grid edges: ring + a few random extras (so receptive fields are non-trivial)
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # true V_jt = some smooth function of features
    W_true = torch.randn(F_dim)
    V_jt = (X @ W_true)                          # (T, N)
    # travel time t_ij and distance d_ij
    coords = torch.randn(N, 2) * 5.0
    diff = coords[:, None, :] - coords[None, :, :]
    d_ij = (diff ** 2).sum(-1).sqrt()             # (N, N)
    log_d_ij = torch.log1p(d_ij)
    t_base = d_ij * 2.0 + 1.0                     # (N, N)
    cong = 1.0 + 0.2 * torch.randn(T, 1, 1)
    t_ij_t = t_base.unsqueeze(0) * cong            # (T, N, N)

    # true utility and choice probs
    V_b = V_jt.unsqueeze(1).expand(T, N, N)
    d_b = log_d_ij.unsqueeze(0).expand(T, N, N)
    U = alpha_true * V_b + beta_true * t_ij_t + gamma_true * d_b
    P = torch.softmax(U, dim=2)                   # (T, N, N)

    # observed flows: equal mass per origin per hour, allocated by P
    F_ij_t = torch.zeros(T, N, N)
    flows_per_origin = max(1, n_pairs // (T * N))
    for t in range(T):
        for i in range(N):
            probs = P[t, i].numpy()
            choices = rng.multinomial(flows_per_origin, probs)
            F_ij_t[t, i] = torch.tensor(choices, dtype=torch.float32)

    return {
        "X": X, "edge_index": edge_index, "t_ij_t": t_ij_t,
        "log_d_ij": log_d_ij, "F_ij_t": F_ij_t,
        "alpha_true": alpha_true, "beta_true": beta_true,
        "gamma_true": gamma_true, "N": N, "T": T, "F": F_dim,
    }


def test_inverse_trainer_smoke():
    """Smoke test: the loss decreases over 50 epochs on tiny synthetic data.

    Exercises the canonical API:
      * kwargs constructor (grid_features, edge_index, observed_OD, t_ij_t, ...)
      * fit() returns 6-tuple (theta, alpha, beta_t, beta_c, gamma, log)
    """
    syn = _make_synthetic(N=20, T=3, n_pairs=1000, seed=0)
    val_mask = torch.zeros(syn["N"], dtype=torch.bool)
    val_mask[::4] = True

    # X comes in as (T, N, F); pass as static (N, F) by averaging.
    X_static = syn["X"].mean(0)
    trainer = InverseRUMTrainer(
        grid_features=X_static,
        edge_index=syn["edge_index"],
        observed_OD=syn["F_ij_t"],
        t_ij_t=syn["t_ij_t"],
        log_d_ij=syn["log_d_ij"],
        K=50,                  # <= N triggers full-N softmax fallback (N=20)
        val_mask=val_mask,
        device="cpu",
        seed=0,
        lr_theta=5e-3,
        lr_rum=5e-2,
        epochs=50,
        patience=50,
        residualise=True,
        verbose=False,
    )
    out = trainer.fit()
    assert len(out) == 6, f"fit() must return 6-tuple; got {len(out)}"
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = out
    assert isinstance(theta_hat, torch.nn.Module)
    assert isinstance(alpha_hat, float)
    assert isinstance(beta_t_hat, torch.Tensor)
    assert isinstance(beta_c_hat, float)
    assert isinstance(gamma_hat, float)
    # sign convention: beta_t and beta_c are constrained negative
    assert (beta_t_hat <= 0).all().item(), "beta_t must be <= 0"
    assert beta_c_hat <= 0, f"beta_c must be <= 0; got {beta_c_hat}"

    assert len(log.train_nll) >= 5
    # loss must decrease meaningfully from epoch 0 to the best
    assert min(log.train_nll) < log.train_nll[0] - 0.05, (
        f"train_nll did not decrease: {log.train_nll[:5]}"
    )


def test_inverse_trainer_predict_OD_shape():
    """predict_OD() exists and returns (T, N, N)."""
    syn = _make_synthetic(N=20, T=3, n_pairs=500, seed=0)
    X_static = syn["X"].mean(0)
    trainer = InverseRUMTrainer(
        grid_features=X_static,
        edge_index=syn["edge_index"],
        observed_OD=syn["F_ij_t"],
        t_ij_t=syn["t_ij_t"],
        log_d_ij=syn["log_d_ij"],
        K=50,
        device="cpu",
        seed=0,
        epochs=3,
        patience=3,
        residualise=False,
    )
    trainer.fit()
    pred = trainer.predict_OD()
    assert pred.shape == (syn["T"], syn["N"], syn["N"])

    # override path: pass a different cost matrix and check shape preserved.
    new_t = syn["t_ij_t"] * 1.5
    pred_override = trainer.predict_OD(t_ij_override=new_t)
    assert pred_override.shape == (syn["T"], syn["N"], syn["N"])
    # outputs should differ when costs differ.
    assert not torch.allclose(pred, pred_override, atol=1e-6)


def test_inverse_trainer_topk_invoked():
    """When N > K the trainer must call make_choice_set per (t, origin)."""
    # Build a tiny problem with N=60 > K=20 so the top-K path fires.
    syn = _make_synthetic(N=60, T=2, n_pairs=2000, seed=2)
    X_static = syn["X"].mean(0)
    trainer = InverseRUMTrainer(
        grid_features=X_static,
        edge_index=syn["edge_index"],
        observed_OD=syn["F_ij_t"],
        t_ij_t=syn["t_ij_t"],
        log_d_ij=syn["log_d_ij"],
        K=20,
        device="cpu",
        seed=2,
        epochs=2,
        patience=2,
        residualise=False,
    )
    trainer.fit()
    # >= N * T per epoch * (1 epoch min before early stop).
    assert trainer._topk_calls >= syn["N"] * syn["T"], (
        f"top-K not invoked enough: {trainer._topk_calls} calls"
    )


@pytest.mark.slow
def test_inverse_trainer_identifiability():
    """Recover beta within 20% of the true value on tiny synthetic data.

    Marked slow because it needs ~300 epochs to settle.  Skipped by default
    via the ``slow`` marker; run with ``pytest -m slow``.

    The synthetic DGP has no confounder between V_jt and t_ij to partial out
    (there is no shared structural signal), so we run with residualise=False:
    the pure identifiability check is what matters here. The ridge-style
    penalty's finite-sample bias on N=20 is benchmarked separately on the
    full London grid (experiments/paper_a/synthetic_recovery.py).
    """
    beta_true = BETA_T_STAR     # -0.07
    syn = _make_synthetic(N=20, T=3, n_pairs=8000, seed=1, beta_true=beta_true)
    val_mask = torch.zeros(syn["N"], dtype=torch.bool); val_mask[::5] = True

    X_static = syn["X"].mean(0)
    trainer = InverseRUMTrainer(
        grid_features=X_static,
        edge_index=syn["edge_index"],
        observed_OD=syn["F_ij_t"],
        t_ij_t=syn["t_ij_t"],
        log_d_ij=syn["log_d_ij"],
        K=50,
        val_mask=val_mask,
        device="cpu",
        seed=1,
        lr_theta=2e-3,
        lr_rum=8e-2,
        epochs=300,
        patience=300,
        residualise=False,
        verbose=False,
    )
    _, _, beta_hat_t, _, _, _ = trainer.fit()
    beta_hat = float(beta_hat_t.mean().item())
    rel_err = abs(beta_hat - beta_true) / abs(beta_true)
    assert rel_err < 0.20, f"beta_hat = {beta_hat:.4f}, true = {beta_true}, rel_err {rel_err:.3f}"


def test_inverse_trainer_alpha_fixed_at_one():
    """alpha is a fixed buffer at 1.0 and is NOT trained.

    The fit() return must always have alpha_hat == 1.0 (within float tol),
    and the alpha attribute on _RUMHead must not require grad.
    """
    syn = _make_synthetic(N=20, T=2, n_pairs=500, seed=0)
    X_static = syn["X"].mean(0)
    trainer = InverseRUMTrainer(
        grid_features=X_static,
        edge_index=syn["edge_index"],
        observed_OD=syn["F_ij_t"],
        t_ij_t=syn["t_ij_t"],
        log_d_ij=syn["log_d_ij"],
        K=50,
        device="cpu",
        seed=0,
        epochs=5,
        patience=5,
        residualise=False,
    )
    # alpha must not be a Parameter
    assert not isinstance(trainer.rum.alpha, torch.nn.Parameter), (
        "alpha must be a fixed buffer, not a trainable Parameter"
    )
    # alpha is registered as a buffer at 1.0
    assert abs(float(trainer.rum.alpha.item()) - 1.0) < 1e-8
    # Confirm no parameter named anything alpha-like in the head
    rum_param_names = [n for n, _ in trainer.rum.named_parameters()]
    assert not any("alpha" in n for n in rum_param_names), (
        f"alpha must not be in parameters; got {rum_param_names}"
    )
    _, alpha_hat, _, _, _, log = trainer.fit()
    assert abs(alpha_hat - 1.0) < 1e-6
    # log.alpha tracks the fixed buffer — every entry should be 1.0
    for a in log.alpha:
        assert abs(a - 1.0) < 1e-6


def test_inverse_trainer_gamma_stays_negative():
    """gamma is constrained < 0 via -softplus reparam; it must remain negative
    across every training epoch (sample post-epoch via the TrainingLog).
    """
    syn = _make_synthetic(N=20, T=2, n_pairs=1000, seed=0)
    X_static = syn["X"].mean(0)
    trainer = InverseRUMTrainer(
        grid_features=X_static,
        edge_index=syn["edge_index"],
        observed_OD=syn["F_ij_t"],
        t_ij_t=syn["t_ij_t"],
        log_d_ij=syn["log_d_ij"],
        K=50,
        device="cpu",
        seed=0,
        lr_rum=1e-1,        # aggressive lr so the parameter actually moves
        epochs=20,
        patience=20,
        residualise=False,
    )
    _, _, _, _, gamma_hat, log = trainer.fit()
    assert gamma_hat < 0, f"gamma_hat must be < 0; got {gamma_hat}"
    # every logged gamma must be strictly negative
    assert len(log.gamma) >= 5
    for g in log.gamma:
        assert g < 0, f"gamma drifted non-negative during training: {log.gamma}"
