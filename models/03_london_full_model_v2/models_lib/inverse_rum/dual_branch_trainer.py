"""Inverse-RUM trainer with dual-branch encoder + RUM head.

Trains the joint (theta_static, theta_dyn, alpha=1, beta_t, beta_c, gamma) by
maximising the multinomial log-likelihood of observed hourly OD flows under

    U_ij(t) = alpha * V_jt + beta_t * t_ij + gamma * log_d_ij

with V_jt = DualBranchEncoder(X_static, X_dynamic, edge_index).

Differences from ``InverseRUMTrainer`` (single-branch):
- Constructor takes TWO feature inputs: ``X_static`` (N, F_s) and
  ``X_dynamic`` (T, N, F_d).
- Internal ``_forward_V`` calls the dual-branch encoder.
- Reuses ``_RUMHead`` from inverse_trainer (untouched) — RUM head is
  feature-agnostic, only needs V_jt of shape (T, N).
- Reuses loss / CPC / early-stopping logic from ``InverseRUMTrainer`` —
  copied locally to avoid coupling to that class's bool/dataclass plumbing.

Default loss path: full-softmax cross-entropy on observed hourly OD.
Top-K sampling is intentionally NOT supported here — it was broken in the
single-branch trainer (chosen alternative not forced into choice set; see
``inverse_trainer.py`` line 277-300) and re-implementing it correctly is
out of scope for this refactor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_branch_encoder import DualBranchEncoder
from .inverse_trainer import _RUMHead, TrainingLog


class DualBranchInverseTrainer:
    """Joint inverse-RUM estimator using dual-branch encoder.

    Canonical use::

        trainer = DualBranchInverseTrainer(
            X_static=X_s,                  # (N, F_s)
            X_dynamic=X_d,                  # (T, N, F_d)
            edge_index=edge_index,          # (2, E)
            observed_OD=F_ij_t,             # (T, N, N)
            t_ij_t=t_ij_t,                  # (T, N, N) or (N, N) free-flow minutes
            log_d_ij=log_d_ij,              # (N, N)
            train_mask=tm, val_mask=vm,
            hidden_dim=32, gru_hidden=32,
            epochs=100, patience=10, verbose=True,
        )
        encoder, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
        pred_F = trainer.predict_OD()
    """

    def __init__(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        observed_OD: torch.Tensor,
        t_ij_t: torch.Tensor,
        log_d_ij: torch.Tensor,
        encoder: Optional[DualBranchEncoder] = None,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
        seed: int = 42,
        hidden_dim: int = 32,
        gru_hidden: int = 32,
        n_sage_layers: int = 2,
        tcn_kernels: Tuple[int, ...] = (3, 5, 7),
        lr_theta: float = 1e-3,
        lr_rum: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = False,
        # Phase B / Q2 extensions: occupation match + observed income heterogeneity.
        # Both terms identifiable from aggregate OD because OccMatch[i,j] and
        # income_score(i) are observed (not latent).
        occ_match: Optional[torch.Tensor] = None,
        income_score_per_origin: Optional[torch.Tensor] = None,
    ):
        # ---- shape validation -------------------------------------------------
        if X_static.dim() != 2:
            raise ValueError(f"X_static must be (N, F_s); got {tuple(X_static.shape)}")
        N, F_s = X_static.shape
        if X_dynamic.dim() != 3 or X_dynamic.shape[1] != N:
            raise ValueError(
                f"X_dynamic must be (T, N, F_d) with N={N}; got {tuple(X_dynamic.shape)}"
            )
        T, _, F_d = X_dynamic.shape

        if observed_OD.dim() == 2:
            observed_OD = observed_OD.unsqueeze(0).expand(T, N, N)
        elif observed_OD.dim() != 3 or observed_OD.shape != (T, N, N):
            raise ValueError(
                f"observed_OD must be (T={T}, N={N}, N) or (N, N); "
                f"got {tuple(observed_OD.shape)}"
            )

        if t_ij_t.dim() == 2:
            if t_ij_t.shape != (N, N):
                raise ValueError(f"t_ij_t (N, N) must be ({N}, {N}); got {tuple(t_ij_t.shape)}")
            t_ij_t = t_ij_t.unsqueeze(0).expand(T, N, N)
        elif t_ij_t.dim() == 3:
            if t_ij_t.shape != (T, N, N):
                raise ValueError(
                    f"t_ij_t (T, N, N) must be ({T}, {N}, {N}); got {tuple(t_ij_t.shape)}"
                )
        else:
            raise ValueError(f"t_ij_t must be (N, N) or (T, N, N); got {tuple(t_ij_t.shape)}")

        if log_d_ij.dim() != 2 or log_d_ij.shape != (N, N):
            raise ValueError(f"log_d_ij must be ({N}, {N}); got {tuple(log_d_ij.shape)}")

        # ---- masks ------------------------------------------------------------
        if val_mask is None:
            val_mask = torch.zeros(N, dtype=torch.bool)
            val_mask[::5] = True
        if train_mask is None:
            train_mask = ~val_mask
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        overlap = (train_mask & val_mask).sum().item()
        assert overlap == 0, f"train/val mask overlap on {overlap} origins"

        # ---- store -----------------------------------------------------------
        self.N, self.T, self.F_s, self.F_d = N, T, F_s, F_d
        self.device = device
        self.seed = seed
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.lr_theta = lr_theta
        self.lr_rum = lr_rum
        self.weight_decay = weight_decay

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.X_static = X_static.to(device).float()
        self.X_dynamic = X_dynamic.to(device).float()
        self.edge_index = edge_index.to(device)
        self.observed_OD = observed_OD.to(device).float()
        self.t_ij_t = t_ij_t.to(device).float().contiguous()
        self.log_d_ij = log_d_ij.to(device).float()
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)

        # ---- model ----------------------------------------------------------
        if encoder is None:
            encoder = DualBranchEncoder(
                static_dim=F_s,
                dyn_dim=F_d,
                hidden_dim=hidden_dim,
                gru_hidden=gru_hidden,
                n_sage_layers=n_sage_layers,
                tcn_kernels=tcn_kernels,
            )
        self.encoder = encoder.to(device)
        # Q2 / Q1 optional augmentations.
        if occ_match is not None:
            occ_t = occ_match if isinstance(occ_match, torch.Tensor) else torch.tensor(occ_match)
            if occ_t.shape != (N, N):
                raise ValueError(f"occ_match must be ({N}, {N}); got {tuple(occ_t.shape)}")
            self.occ_match = occ_t.to(device).float()
            self._enable_occ_match = True
        else:
            self.occ_match = None
            self._enable_occ_match = False

        if income_score_per_origin is not None:
            isc = (income_score_per_origin if isinstance(income_score_per_origin, torch.Tensor)
                   else torch.tensor(income_score_per_origin))
            if isc.shape != (N,):
                raise ValueError(
                    f"income_score_per_origin must be ({N},); got {tuple(isc.shape)}"
                )
            isc = isc.float()
            isc = (isc - isc.mean()) / (isc.std() + 1e-8)
            self.income_score_per_origin = isc.to(device)
            self._enable_income_interaction = True
        else:
            self.income_score_per_origin = None
            self._enable_income_interaction = False

        self.rum = _RUMHead(n_hours=T, enable_occ_match=self._enable_occ_match).to(device)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": lr_theta,
                 "weight_decay": weight_decay},
                {"params": self.rum.parameters(), "lr": lr_rum,
                 "weight_decay": 0.0},
            ]
        )

    # -------------------------------------------------------------- forward
    def _forward_V(self, norm_stats=None) -> torch.Tensor:
        return self.encoder(self.X_static, self.X_dynamic, self.edge_index,
                            norm_stats=norm_stats)

    # -------------------------------------------------------------- log-prob
    def _per_hour_log_p(self, V_jt: torch.Tensor) -> torch.Tensor:
        """log P(j | i, t) for all (t, i, j) — full N×N softmax, one per hour.

        Logits per (t, i, j):
            α · V_{t,j}                           # GNN-learned attractiveness
            + β_eff(i) · t_{t,i,j}                # time disutility, optionally
                                                  #   scaled by origin income
            + γ · log_d_{i,j}                     # distance disutility
            + δ · OccMatch[i, j]                  # Q2: labour-market match
        β_eff(i) = β_t · (1 + φ · z_income(i))    # Q1: observed heterogeneity
        """
        T, N = V_jt.shape
        alpha = self.rum.alpha
        gamma = self.rum.gamma
        beta_t = self.rum.beta_t                                # (T, n_tiers=1)
        beta_per_hour = beta_t.squeeze(-1)                       # (T,)
        V_b = V_jt.view(T, 1, N).expand(T, N, N)                 # (T, N, N)
        d_b = self.log_d_ij.view(1, N, N)                         # (1, N, N)
        beta_b = beta_per_hour.view(T, 1, 1)                     # (T, 1, 1)
        # Q1: observed-heterogeneity scale on β
        if self._enable_income_interaction and self.income_score_per_origin is not None:
            phi = self.rum.phi
            score_o = self.income_score_per_origin                # (N,)
            scale_per_origin = 1.0 + phi * score_o.view(1, N, 1)  # (1, N, 1)
            beta_b = beta_b * scale_per_origin                     # (T, N, 1)
        logits = alpha * V_b + beta_b * self.t_ij_t + gamma * d_b
        # Q2: labour-market match additive utility term
        if self._enable_occ_match and self.occ_match is not None:
            delta = self.rum.delta
            logits = logits + delta * self.occ_match.view(1, N, N)
        return torch.log_softmax(logits, dim=-1)                 # (T, N, N)

    def _nll(self, log_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T, N, _ = log_p.shape
        mask_f = mask.view(1, N, 1).float()
        flow = self.observed_OD * mask_f
        denom = max(int((flow.sum(dim=2) > 0).sum()), 1)
        return -(flow * log_p).sum() / denom

    def _cpc(self, log_p: torch.Tensor, mask: torch.Tensor) -> float:
        with torch.no_grad():
            P = log_p.exp()
            row_sum = self.observed_OD.sum(dim=2, keepdim=True)
            pred = P * row_sum
            num = 2.0 * torch.minimum(pred[:, mask], self.observed_OD[:, mask]).sum()
            den = (pred[:, mask].sum() + self.observed_OD[:, mask].sum()).clamp(min=1.0)
            return float(num / den)

    # -------------------------------------------------------------- fit
    def fit(self) -> Tuple[DualBranchEncoder, float, torch.Tensor, float, float, TrainingLog]:
        log = TrainingLog()
        best_val = float("inf")
        no_improve = 0
        best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        for ep in range(self.epochs):
            self.encoder.train()
            self.rum.train()
            self.optimizer.zero_grad()

            V_jt = self._forward_V()
            log_p = self._per_hour_log_p(V_jt)
            loss = self._nll(log_p, self.train_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.rum.parameters()), 1.0
            )
            self.optimizer.step()

            self.encoder.eval()
            self.rum.eval()
            with torch.no_grad():
                V_e = self._forward_V()
                log_p_e = self._per_hour_log_p(V_e)
                val_nll = self._nll(log_p_e, self.val_mask).item()
                cpc_val = self._cpc(log_p_e, self.val_mask)

            log.epoch.append(ep)
            log.train_nll.append(float(loss.item()))
            log.val_nll.append(val_nll)
            log.cpc_val.append(cpc_val)
            log.alpha.append(float(self.rum.alpha.item()))
            log.beta_mean.append(float(self.rum.beta_t.mean().item()))
            log.beta_c.append(float(self.rum.beta_c.item()))
            log.gamma.append(float(self.rum.gamma.item()))

            improved = val_nll < best_val - 1e-4
            if improved:
                best_val = val_nll
                no_improve = 0
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                    "rum": {k: v.detach().clone() for k, v in self.rum.state_dict().items()},
                }
            else:
                no_improve += 1

            if self.verbose:
                print(
                    f"ep {ep:3d} | train_nll {loss.item():.4f} | val_nll {val_nll:.4f} | "
                    f"cpc {cpc_val:.3f} | beta_mean {self.rum.beta_t.mean().item():+.4f} | "
                    f"gamma {self.rum.gamma.item():.3f}"
                )

            if no_improve >= self.patience:
                if self.verbose:
                    print(f"early stop at epoch {ep}")
                break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.rum.load_state_dict(best_state["rum"])

        return (
            self.encoder,
            float(self.rum.alpha.item()),
            self.rum.beta_t.detach().clone().squeeze(-1),
            float(self.rum.beta_c.item()),
            float(self.rum.gamma.item()),
            log,
        )

    # -------------------------------------------------------------- predict
    def predict_OD(
        self,
        t_ij_override: Optional[torch.Tensor] = None,
        norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict (T, N, N) flow counts under the trained model, with optional
        cost override for temporal-holdout evaluation."""
        self.encoder.eval()
        self.rum.eval()
        with torch.no_grad():
            V_jt = self._forward_V(norm_stats=norm_stats)
            t_use = self.t_ij_t if t_ij_override is None else t_ij_override.to(self.device).float()
            if t_use.dim() == 2:
                t_use = t_use.unsqueeze(0).expand(self.T, self.N, self.N)
            elif t_use.dim() == 3 and t_use.shape[0] != self.T:
                t_use = t_use.mean(0, keepdim=True).expand(self.T, self.N, self.N)
            # Reuse log-prob computation but with overridden t
            T, N = V_jt.shape
            alpha = self.rum.alpha
            gamma = self.rum.gamma
            beta_per_hour = self.rum.beta_t.squeeze(-1)
            V_b = V_jt.view(T, 1, N).expand(T, N, N)
            beta_b = beta_per_hour.view(T, 1, 1)
            d_b = self.log_d_ij.view(1, N, N)
            if self._enable_income_interaction and self.income_score_per_origin is not None:
                phi = self.rum.phi
                score_o = self.income_score_per_origin
                beta_b = beta_b * (1.0 + phi * score_o.view(1, N, 1))
            logits = alpha * V_b + beta_b * t_use + gamma * d_b
            if self._enable_occ_match and self.occ_match is not None:
                delta = self.rum.delta
                logits = logits + delta * self.occ_match.view(1, N, N)
            log_p = torch.log_softmax(logits, dim=-1)
            P = log_p.exp()
            row_sum = self.observed_OD.sum(dim=2, keepdim=True)
            return (P * row_sum).cpu()
