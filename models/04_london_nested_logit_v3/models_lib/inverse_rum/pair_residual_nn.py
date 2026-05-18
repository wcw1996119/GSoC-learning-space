"""Pair-aware residual NN: outputs (T, N, N) instead of (T, N).

Motivation
----------
The default DualBranchEncoder outputs V_gnn_jt of shape (T, N) — one scalar
per (hour, destination grid). When fed into the RUM-mixture trainer it is
broadcast across origins:
    V_gnn[t, i, j] = V_gnn_jt[t, j]   for all i

This means NN cannot learn OD-pair-specific patterns. For 27-class
latent-class mixture, γ_M per class is starved of gradient (each class
mixture weight ~ 4 %), so the per-OD-pair gravity signal that γ_M would
normally express is lost. NN cannot pick it up because NN has no origin
awareness.

Pair-aware NN
-------------
We build two low-rank embeddings (origin- and destination-side) from raw
(X_static, X_dynamic) and bilinear them into V_pair[t, i, j]:

    e_origin[t, i, :] = MLP_origin(concat(X_static[i], X_dynamic[t, i]))   ∈ R^R
    e_dest  [t, j, :] = MLP_dest  (concat(X_static[j], X_dynamic[t, j]))   ∈ R^R
    V_pair  [t, i, j] = e_origin[t, i, :] · e_dest[t, j, :]               (rank R)

Memory: O(T·N·N) = 286 MB at 24·1725² — same as current broadcast.
Params: 2 × MLP. Tiny vs RUM head.

The output is z-normalised over (t, i, j) so the existing residual-scale
parameter w_NN in CerveroShenHead stays interpretable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairResidualNN(nn.Module):
    """OD-pair-aware residual NN.

    Parameters
    ----------
    static_dim : int      — number of static (per-grid) features
    dyn_dim    : int      — number of dynamic (per-(t, grid)) features
    hidden_dim : int      — MLP hidden width
    pair_rank  : int      — bilinear rank; total params ∝ 2 × hidden × pair_rank
    """

    def __init__(
        self,
        static_dim: int,
        dyn_dim: int,
        hidden_dim: int = 32,
        pair_rank: int = 8,
    ):
        super().__init__()
        in_dim = static_dim + dyn_dim
        self.origin_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pair_rank),
        )
        self.dest_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pair_rank),
        )
        self.pair_rank = pair_rank

    def forward(
        self,
        X_static: torch.Tensor,    # (N, F_s)
        X_dynamic: torch.Tensor,   # (T, N, F_d)
        edge_index: torch.Tensor,  # unused (no graph ops here); accept for API parity
        norm_stats=None,           # tuple (mean, std) for inference under intervention
    ) -> torch.Tensor:
        T, N, _ = X_dynamic.shape
        X_static_T = X_static.unsqueeze(0).expand(T, N, -1)        # (T, N, F_s)
        X_combined = torch.cat([X_static_T, X_dynamic], dim=-1)    # (T, N, F_s+F_d)
        e_origin = self.origin_mlp(X_combined)                     # (T, N, R)
        e_dest = self.dest_mlp(X_combined)                         # (T, N, R)
        # V_pair[t, i, j] = e_origin[t, i, :] · e_dest[t, j, :]
        V_pair = torch.einsum("tir,tjr->tij", e_origin, e_dest)    # (T, N, N)
        # Identifiability normalisation: zero-mean unit-std over (t, i, j)
        if norm_stats is None:
            mean_v = V_pair.mean()
            std_v = V_pair.std() + 1e-6
        else:
            mean_v, std_v = norm_stats
            std_v = std_v + 1e-6
        return (V_pair - mean_v) / std_v

    def compute_norm_stats(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        T, N, _ = X_dynamic.shape
        X_static_T = X_static.unsqueeze(0).expand(T, N, -1)
        X_combined = torch.cat([X_static_T, X_dynamic], dim=-1)
        e_origin = self.origin_mlp(X_combined)
        e_dest = self.dest_mlp(X_combined)
        V_pair = torch.einsum("tir,tjr->tij", e_origin, e_dest)
        return (V_pair.mean().detach(), V_pair.std().detach())
