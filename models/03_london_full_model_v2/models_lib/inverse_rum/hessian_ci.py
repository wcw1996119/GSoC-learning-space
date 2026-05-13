"""Hessian-based asymptotic confidence intervals for InverseRUMTrainer.

Two CI flavors are implemented:

1. ``hessian_ci_rum`` — vanilla inverse-Hessian (Cramér-Rao) CI. Correct
   when the model is exactly well-specified. Fast: one Hessian compute.

2. ``sandwich_ci_rum`` — Huber-White robust CI:
       Cov_sandwich(θ̂) = H^{-1} · V · H^{-1}
   where V is the "meat" (sum of outer products of per-origin scores).
   Correct under model mis-specification (e.g., GNN θ̂ is a nuisance not
   estimated to perfection). Slower: requires per-origin gradient compute.

In the large-N regime, Poisson bootstrap underestimates variance because
resampling F_obs from Poisson(F_obs) introduces almost no signal change
when the per-cell mean is large. Hessian-based CIs use the Cramér-Rao
asymptotic bound directly:

    Cov(θ̂) ≈ I(θ̂)^{-1}
    SE(θ̂_k) = sqrt(diag(Cov)[k])
    CI_α(θ̂_k) = θ̂_k ± z_{1-α/2} · SE(θ̂_k)

where I(θ̂) is the observed Fisher Information (i.e., Hessian of the
total negative log-likelihood at the MLE).

Concentrated likelihood
-----------------------
GNN parameters θ are nuisance: we condition on θ̂ (frozen at convergence)
and compute the joint Hessian over the structural RUM parameters
(β_t, γ). β_c is the cost numeraire and does not appear in the logits
in the current trainer (see _per_hour_log_p_full); we therefore omit it
from the CI report.

Reparameterization
------------------
The trainer stores raw parameters (raw_β_t, raw_γ) and applies
β = -softplus(raw). We compute the Hessian directly in (β, γ) space
(treating them as free parameters), which gives the standard delta-method
CI on the post-softplus parameters. This is mathematically equivalent
because the Hessian transforms covariantly under reparameterization at
the optimum.

References
----------
- McFadden 1974, "Conditional logit analysis of qualitative choice behavior",
  appendix on asymptotic distribution of MLE.
- Train 2009, "Discrete Choice Methods with Simulation", ch. 8.
- Cox & Hinkley 1974, "Theoretical Statistics", §9.2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from .inverse_trainer import InverseRUMTrainer


_Z_95 = 1.959963984540054   # norm.ppf(0.975)


def hessian_ci_rum(
    trainer: "InverseRUMTrainer",
    confidence: float = 0.95,
    ridge: float = 1e-6,
) -> dict:
    """Compute joint asymptotic CIs for (β_t mean across hours, γ).

    Parameters
    ----------
    trainer    : InverseRUMTrainer at converged state.
    confidence : nominal CI level, default 0.95.
    ridge      : Tikhonov regularisation added to Hessian before inverting,
                 to handle near-singular cases (e.g., flat log-likelihood
                 directions). Default 1e-6 in the same scale as the per-
                 origin loss; the result is asymptotically negligible.

    Returns
    -------
    dict with keys
        beta_t_hat        : (T,) numpy array, post-softplus β per hour
        beta_t_se         : (T,) numpy array, per-hour SE
        beta_t_mean_hat   : float, mean β across hours
        beta_t_mean_se    : float, SE of the mean (using full T×T sub-cov)
        beta_t_mean_ci    : (lo, hi) tuple at the requested confidence
        gamma_hat         : float
        gamma_se          : float
        gamma_ci          : (lo, hi)
        hessian           : (T+1, T+1) numpy array
        cov               : (T+1, T+1) numpy array
        z                 : float, the normal quantile used
        confidence        : float, echoed back

    Caveats
    -------
    - Assumes θ̂ is at a local max of the likelihood. If trainer didn't
      converge, the Hessian may have negative eigenvalues; ridge stabilises
      but doesn't fix the underlying optimisation issue.
    - β_c is fixed numeraire — not in returned CIs.
    - Large-N asymptotic; for very small N (e.g., < 10k trips) the
      asymptotic approximation may understate the true sampling variance.
    """
    # avoid circular import at module load
    from .inverse_trainer import _fwl_inspired_ridge_penalty

    trainer.gnn.eval()
    trainer.rum.eval()

    # ---- Snapshot post-softplus parameters at convergence -----------------
    beta_t_hat = trainer.rum.beta_t.detach().clone()    # (T, K) Phase B: K=n_tiers
    gamma_hat = trainer.rum.gamma.detach().clone()       # scalar
    T, K = beta_t_hat.shape

    # Phase B: include δ if OccMatch is enabled.
    enable_occ = getattr(trainer, "_enable_occ_match", False) and trainer.occ_match is not None
    delta_hat = trainer.rum.delta.detach().clone() if enable_occ else None
    occ_match = trainer.occ_match if enable_occ else None

    # Phase B-2b: include mixture (income tier) if enabled.
    enable_mixture = getattr(trainer, "_enable_mixture", False) and trainer.income_tier_props is not None
    income_tier_props = trainer.income_tier_props if enable_mixture else None

    # Phase B-2c: include φ if income-interaction is enabled.
    enable_interaction = getattr(trainer, "_enable_income_interaction", False) and trainer.income_score_per_origin is not None
    phi_hat = trainer.rum.phi.detach().clone() if enable_interaction else None
    income_score = trainer.income_score_per_origin if enable_interaction else None
    # Phase B-2d: include ψ if dest-wage interaction is enabled.
    enable_dest_interaction = getattr(trainer, "_enable_dest_wage_interaction", False) and trainer.wage_score_per_dest is not None
    psi_hat = trainer.rum.psi.detach().clone() if enable_dest_interaction else None
    wage_score = trainer.wage_score_per_dest if enable_dest_interaction else None
    # Phase B-2e: include ξ if wage attraction is enabled.
    enable_wage_attr = getattr(trainer, "_enable_wage_attraction", False) and trainer.wage_score_per_dest is not None
    xi_hat = trainer.rum.xi.detach().clone() if enable_wage_attr else None

    # ---- Frozen V_jt at converged GNN -------------------------------------
    with torch.no_grad():
        V_jt = trainer._forward_V(trainer.X)
        if trainer.residualise:
            V_jt = _fwl_inspired_ridge_penalty(V_jt, trainer.t_ij_t, trainer.log_d_ij)
        V_jt = V_jt.detach()

    F_obs = trainer.observed_OD            # (T, N, N)
    t_ij_t = trainer.t_ij_t                # (T, N, N)
    log_d_ij = trainer.log_d_ij            # (N, N)
    alpha_buf = trainer.rum.alpha          # buffer 1.0
    N = V_jt.shape[1]

    # ---- Total NLL as function of (β_t..., γ [, δ] [, φ]) ------------------
    def neg_log_lik(params: torch.Tensor) -> torch.Tensor:
        """Total NLL (no per-origin averaging — required for CR Hessian)."""
        beta_t = params[:T * K].view(T, K)                             # (T, K)
        gamma = params[T * K]                                           # scalar
        idx = T * K + 1
        # Build logits (T, K, N, N) — same as trainer._per_hour_log_p_full
        V_b = V_jt.view(T, 1, 1, N).expand(T, K, N, N)
        t_b = t_ij_t.view(T, 1, N, N).expand(T, K, N, N)
        beta_b = beta_t.view(T, K, 1, 1).expand(T, K, N, N)
        d_b = log_d_ij.view(1, 1, N, N).expand(T, K, N, N)
        scale = None
        if enable_interaction:
            phi = params[idx]; idx += 1
            scale = phi * income_score.view(1, 1, N, 1)
        if enable_dest_interaction:
            psi = params[idx]; idx += 1
            term_d = psi * wage_score.view(1, 1, 1, N)
            scale = term_d if scale is None else scale + term_d
        if scale is not None:
            beta_b = beta_b * (1.0 + scale)
        logits = alpha_buf * V_b + beta_b * t_b + gamma * d_b
        if enable_occ:
            delta = params[idx]; idx += 1
            occ_b = occ_match.view(1, 1, N, N).expand(T, K, N, N)
            logits = logits + delta * occ_b
        if enable_wage_attr:
            xi = params[idx]; idx += 1
            wage_term = xi * wage_score.view(1, 1, 1, N)
            logits = logits + wage_term.expand(T, K, N, N)
        log_p_per_tier = torch.log_softmax(logits, dim=-1)             # (T, K, N, N)
        if enable_mixture:
            log_pi = torch.log(income_tier_props.clamp(min=1e-30))     # (N, K)
            log_pi_b = log_pi.t().view(1, K, N, 1)                      # (1, K, N, 1)
            log_p = torch.logsumexp(log_p_per_tier + log_pi_b, dim=1)  # (T, N, N)
        else:
            log_p = log_p_per_tier.squeeze(1)                           # (T, N, N)
        nll = -(F_obs * log_p).sum()
        return nll

    # Build params0 with consistent ordering: β, γ, [φ], [ψ], [δ], [ξ]
    parts = [beta_t_hat.flatten(), gamma_hat.unsqueeze(0)]
    if enable_interaction:
        parts.append(phi_hat.unsqueeze(0))
    if enable_dest_interaction:
        parts.append(psi_hat.unsqueeze(0))
    if enable_occ:
        parts.append(delta_hat.unsqueeze(0))
    if enable_wage_attr:
        parts.append(xi_hat.unsqueeze(0))
    params0 = torch.cat(parts)

    # ---- Compute Hessian via autograd -------------------------------------
    # torch.autograd.functional.hessian uses double backprop. The Hessian
    # is (T·K + 1) x (T·K + 1) [+1 if occ_match enabled], small for T=1, K=3.
    H = torch.autograd.functional.hessian(neg_log_lik, params0)

    p_dim = H.shape[0]
    H_reg = H + ridge * torch.eye(p_dim, dtype=H.dtype, device=H.device)

    # ---- Invert ------------------------------------------------------------
    try:
        cov = torch.linalg.inv(H_reg)
    except RuntimeError:
        cov = torch.linalg.pinv(H_reg)

    # Negative diag entries indicate non-PSD Hessian (i.e., not a true min).
    # Clamp at small positive value before sqrt; flag in `cov_psd` for caller.
    diag_cov = torch.diagonal(cov)
    cov_psd = bool((diag_cov >= 0).all().item())
    diag_clamped = diag_cov.clamp(min=1e-30)
    se_vec = torch.sqrt(diag_clamped)

    # ---- Z multiplier ------------------------------------------------------
    if abs(confidence - 0.95) < 1e-9:
        z = _Z_95
    else:
        try:
            from scipy.stats import norm
            z = float(norm.ppf((1 + confidence) / 2))
        except ImportError:
            # crude approximation if scipy unavailable
            alpha = 1 - confidence
            z = math.sqrt(2) * math.erfinv(1 - alpha)

    # ---- Per-parameter results --------------------------------------------
    # se_vec layout: [β flatten (T*K,), γ scalar, [δ scalar if occ]]
    beta_t_se = se_vec[:T * K].view(T, K)              # (T, K)
    gamma_se = se_vec[T * K]                            # scalar

    # SE of mean β per tier (averaged over hours)
    # Per-tier β_mean variance accounts for hour-covariance within tier.
    cov_beta = cov[:T * K, :T * K].view(T, K, T, K)    # block (T,K) x (T,K)
    beta_mean_per_tier = beta_t_hat.mean(dim=0)         # (K,)
    # variance of (1/T) Σ_t β_{t,k} = (1/T²) Σ_{t1,t2} cov[(t1,k), (t2,k)]
    beta_mean_var_per_tier = torch.zeros(K, dtype=cov.dtype, device=cov.device)
    for k in range(K):
        beta_mean_var_per_tier[k] = cov_beta[:, k, :, k].sum() / (T * T)
    beta_mean_se_per_tier = torch.sqrt(beta_mean_var_per_tier.clamp(min=1e-30))

    # Cross-tier-and-hour mean (paper A backward-compat scalar β reporting)
    beta_overall_mean = float(beta_t_hat.mean().item())
    beta_overall_var = cov_beta.sum() / ((T * K) ** 2)
    beta_overall_se = float(torch.sqrt(beta_overall_var.clamp(min=1e-30)).item())

    gamma_val = float(gamma_hat.item())

    result = {
        "beta_t_hat": beta_t_hat.cpu().numpy(),                      # (T, K)
        "beta_t_se": beta_t_se.cpu().numpy(),                        # (T, K)
        "beta_mean_per_tier_hat": beta_mean_per_tier.cpu().numpy(),  # (K,)
        "beta_mean_per_tier_se": beta_mean_se_per_tier.cpu().numpy(),# (K,)
        "beta_mean_per_tier_ci": [
            (float(beta_mean_per_tier[k].item()) - z * float(beta_mean_se_per_tier[k].item()),
             float(beta_mean_per_tier[k].item()) + z * float(beta_mean_se_per_tier[k].item()))
            for k in range(K)
        ],
        # Backward-compat: if K==1, beta_t_mean_hat is the same as scalar β.
        "beta_t_mean_hat": beta_overall_mean,
        "beta_t_mean_se": beta_overall_se,
        "beta_t_mean_ci": (
            beta_overall_mean - z * beta_overall_se,
            beta_overall_mean + z * beta_overall_se,
        ),
        "gamma_hat": gamma_val,
        "gamma_se": float(gamma_se.item()),
        "gamma_ci": (
            gamma_val - z * float(gamma_se.item()),
            gamma_val + z * float(gamma_se.item()),
        ),
        "hessian": H.detach().cpu().numpy(),
        "cov": cov.detach().cpu().numpy(),
        "cov_psd": cov_psd,
        "n_income_tiers": K,
        "z": z,
        "confidence": confidence,
    }
    # Index past β (T*K params) and γ (1 param)
    extra_idx = T * K + 1
    if enable_interaction:
        phi_se = se_vec[extra_idx]
        phi_val = float(phi_hat.item())
        result["phi_hat"] = phi_val
        result["phi_se"] = float(phi_se.item())
        result["phi_ci"] = (
            phi_val - z * float(phi_se.item()),
            phi_val + z * float(phi_se.item()),
        )
        extra_idx += 1
    if enable_dest_interaction:
        psi_se = se_vec[extra_idx]
        psi_val = float(psi_hat.item())
        result["psi_hat"] = psi_val
        result["psi_se"] = float(psi_se.item())
        result["psi_ci"] = (
            psi_val - z * float(psi_se.item()),
            psi_val + z * float(psi_se.item()),
        )
        extra_idx += 1
    if enable_occ:
        delta_se = se_vec[extra_idx]
        delta_val = float(delta_hat.item())
        result["delta_hat"] = delta_val
        result["delta_se"] = float(delta_se.item())
        result["delta_ci"] = (
            delta_val - z * float(delta_se.item()),
            delta_val + z * float(delta_se.item()),
        )
        extra_idx += 1
    if enable_wage_attr:
        xi_se = se_vec[extra_idx]
        xi_val = float(xi_hat.item())
        result["xi_hat"] = xi_val
        result["xi_se"] = float(xi_se.item())
        result["xi_ci"] = (
            xi_val - z * float(xi_se.item()),
            xi_val + z * float(xi_se.item()),
        )
    return result


def sandwich_ci_rum(
    trainer: "InverseRUMTrainer",
    confidence: float = 0.95,
    ridge: float = 1e-6,
) -> dict:
    """Compute Huber-White robust (sandwich) CIs for (β_t, γ).

    Cov_sandwich(θ̂) = H^{-1} V H^{-1}

    where:
        H = Hessian of total NLL at MLE   (same as hessian_ci_rum)
        V = Σ_i score_i · score_i^T       ("meat", outer product of
                                             per-origin scores)

    Per-origin score:
        score_i = ∂ ( Σ_j F[i,j] · log P(j|i) ) / ∂(β,γ)

    The sandwich form is the standard robust covariance for misspecified
    MLE (Huber 1967, White 1982). For correctly-specified models it
    coincides with H^{-1}; for misspecified models V > Fisher info, so
    Cov_sandwich > Cov_Fisher and CIs are wider.

    For our setup, model misspecification arises mainly from the GNN θ
    being treated as a nuisance (its estimation uncertainty is not
    propagated to (β, γ) in the concentrated likelihood). Sandwich SE
    captures this misspec source automatically.

    Returns same keys as hessian_ci_rum, with extra:
        meat            : (T+1, T+1) numpy array (V matrix)
        sandwich_correction : float, ratio of mean SE to inverse-Hessian SE
                              (>1 = sandwich wider than vanilla)

    Cost: O(N) autograd.grad calls (one per origin), so ~N=1725 in our
    setup. Empirically ~5-15 seconds for T=1.
    """
    from .inverse_trainer import _fwl_inspired_ridge_penalty

    trainer.gnn.eval()
    trainer.rum.eval()

    # ---- Snapshot post-softplus parameters at convergence -----------------
    beta_t_hat = trainer.rum.beta_t.detach().clone()    # (T,)
    gamma_hat = trainer.rum.gamma.detach().clone()       # scalar
    T = beta_t_hat.shape[0]

    # ---- Frozen V_jt at converged GNN -------------------------------------
    with torch.no_grad():
        V_jt = trainer._forward_V(trainer.X)
        if trainer.residualise:
            V_jt = _fwl_inspired_ridge_penalty(V_jt, trainer.t_ij_t, trainer.log_d_ij)
        V_jt = V_jt.detach()

    F_obs = trainer.observed_OD            # (T, N, N)
    t_ij_t = trainer.t_ij_t                # (T, N, N)
    log_d_ij = trainer.log_d_ij            # (N, N)
    alpha_buf = trainer.rum.alpha          # buffer 1.0
    N = V_jt.shape[1]
    p_dim = T + 1

    # ---- Total NLL — same as hessian_ci_rum ----------------------------
    def neg_log_lik(params: torch.Tensor) -> torch.Tensor:
        beta_t = params[:T]
        gamma = params[T]
        V_b = V_jt.unsqueeze(1).expand(T, N, N)
        d_b = log_d_ij.unsqueeze(0).expand(T, N, N)
        beta_b = beta_t.view(T, 1, 1).expand(T, N, N)
        logits = (alpha_buf * V_b + beta_b * t_ij_t + gamma * d_b)
        log_p = torch.log_softmax(logits, dim=-1)
        return -(F_obs * log_p).sum()

    params0 = torch.cat([beta_t_hat.flatten(), gamma_hat.unsqueeze(0)])

    # ---- Hessian H ----------------------------------------------------------
    H = torch.autograd.functional.hessian(neg_log_lik, params0)
    H_reg = H + ridge * torch.eye(p_dim, dtype=H.dtype, device=H.device)
    try:
        H_inv = torch.linalg.inv(H_reg)
    except RuntimeError:
        H_inv = torch.linalg.pinv(H_reg)

    # ---- Meat V: per-origin scores ----------------------------------------
    # Per-origin objective is L_i = -Σ_t Σ_j F[t,i,j] · log P(j|i,t).
    # Score_i = ∂L_i / ∂params (sign convention: same as ∂NLL/∂params).
    # V = Σ_i score_i · score_i^T (outer product, summed over origins).
    V_meat = torch.zeros((p_dim, p_dim), dtype=H.dtype, device=H.device)

    for i in range(N):
        params_i = params0.detach().clone().requires_grad_(True)
        beta_t = params_i[:T]
        gamma = params_i[T]
        # Build logits only for origin i (output: (T, N))
        # V_jt is destination-indexed (T, N); broadcast as destination row.
        logits_i = (alpha_buf * V_jt
                    + beta_t.view(T, 1) * t_ij_t[:, i, :]
                    + gamma * log_d_ij[i, :].unsqueeze(0).expand(T, N))   # (T, N)
        log_p_i = torch.log_softmax(logits_i, dim=-1)                       # (T, N)
        nll_i = -(F_obs[:, i, :] * log_p_i).sum()
        # Skip origins with zero observed flow — their score is 0 anyway
        # but autograd may produce NaN.
        if F_obs[:, i, :].sum() < 1e-12:
            continue
        score_i = torch.autograd.grad(nll_i, params_i, retain_graph=False)[0]
        V_meat = V_meat + torch.outer(score_i, score_i)

    # ---- Sandwich Cov ------------------------------------------------------
    cov_sandwich = H_inv @ V_meat @ H_inv

    diag_cov = torch.diagonal(cov_sandwich)
    cov_psd = bool((diag_cov >= 0).all().item())
    diag_clamped = diag_cov.clamp(min=1e-30)
    se_vec = torch.sqrt(diag_clamped)

    # ---- Z multiplier ------------------------------------------------------
    if abs(confidence - 0.95) < 1e-9:
        z = _Z_95
    else:
        try:
            from scipy.stats import norm
            z = float(norm.ppf((1 + confidence) / 2))
        except ImportError:
            alpha = 1 - confidence
            z = math.sqrt(2) * math.erfinv(1 - alpha)

    # ---- Sandwich correction ratio (diagnostic) ---------------------------
    H_inv_diag = torch.diagonal(H_inv).clamp(min=1e-30)
    vanilla_se = torch.sqrt(H_inv_diag)
    sandwich_correction = float((se_vec / vanilla_se).mean().item())

    # ---- Per-parameter results --------------------------------------------
    beta_t_se = se_vec[:T]
    gamma_se = se_vec[T]

    beta_block = cov_sandwich[:T, :T]
    beta_mean_var = beta_block.sum() / (T * T)
    beta_mean_var_clamped = beta_mean_var.clamp(min=1e-30)
    beta_mean_se = float(torch.sqrt(beta_mean_var_clamped).item())

    beta_t_mean = float(beta_t_hat.mean().item())
    gamma_val = float(gamma_hat.item())

    return {
        "beta_t_hat": beta_t_hat.cpu().numpy(),
        "beta_t_se": beta_t_se.cpu().numpy(),
        "beta_t_mean_hat": beta_t_mean,
        "beta_t_mean_se": beta_mean_se,
        "beta_t_mean_ci": (
            beta_t_mean - z * beta_mean_se,
            beta_t_mean + z * beta_mean_se,
        ),
        "gamma_hat": gamma_val,
        "gamma_se": float(gamma_se.item()),
        "gamma_ci": (
            gamma_val - z * float(gamma_se.item()),
            gamma_val + z * float(gamma_se.item()),
        ),
        "hessian": H.detach().cpu().numpy(),
        "meat": V_meat.detach().cpu().numpy(),
        "cov": cov_sandwich.detach().cpu().numpy(),
        "cov_psd": cov_psd,
        "sandwich_correction": sandwich_correction,
        "z": z,
        "confidence": confidence,
    }
