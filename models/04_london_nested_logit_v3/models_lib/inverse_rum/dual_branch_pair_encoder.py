"""Full spatio-temporal encoder with OD-pair bilinear output.

Combines v2's neighbor-aware ST-GNN (GraphSAGE static + per-hour GraphSAGE +
GRU + Multi-Scale TCN + Gated Fusion) with pair-aware bilinear head.

Architecture
------------
Static branch:    GraphSAGE 2 layers       → h_static  (N, H)
Dynamic branch:   GraphSAGE per hour
                + GRU (along T axis)
                + Multi-Scale TCN (K=3,5,7) → h_dyn     (T, N, H)
Gated fusion:     gate · h_static + (1-gate) · h_dyn   → h_combined (T, N, H)
Origin head:      MLP (H → R)               → e_origin (T, N, R)
Dest head:        MLP (H → R)               → e_dest   (T, N, R)
Bilinear:         V_NN[t,i,j] = e_origin[t,i,:] · e_dest[t,j,:]
z-norm:           (V_NN - mean) / std

What it learns vs PairResidualNN
--------------------------------
PairResidualNN: only sees grid's OWN features → no spatial context, no temporal.
This encoder: neighbors (GraphSAGE) + history (GRU) + multi-window (TCN) +
              OD-pair (bilinear). All three signals retained.

Memory
------
V_NN (T, N, N) = 286 MB at 24 × 1725², same as PairResidualNN.
Intermediate h_combined (T, N, H) tiny (~5 MB).

Reuses StaticBranch, DynamicBranch from v2 dual_branch_encoder.py
(loaded via sys.path injection by train_cervero_shen.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# v2 namespace is on sys.path (trainer injects V2_ROOT). Reuse its components.
from models_lib.inverse_rum.dual_branch_encoder import StaticBranch, DynamicBranch
from models_lib.inverse_rum.structural_gnn import _build_neighbour_index


class DualBranchPairEncoder(nn.Module):
    """ST-GNN with OD-pair bilinear output (T, N, N)."""

    def __init__(
        self,
        static_dim: int,
        dyn_dim: int,
        hidden_dim: int = 32,
        gru_hidden: int = 32,
        n_sage_layers: int = 2,
        tcn_kernels: Tuple[int, ...] = (3, 5, 7),
        pair_rank: int = 32,
        pair_hidden: int = 64,
    ):
        super().__init__()
        self.static_branch = StaticBranch(static_dim, hidden_dim, n_layers=n_sage_layers)
        self.dyn_branch = DynamicBranch(
            dyn_dim, hidden_dim, n_sage_layers=n_sage_layers,
            gru_hidden=gru_hidden, tcn_kernels=tcn_kernels,
        )
        # Gate (no final scalar head — keep hidden_dim representation)
        self.gate_net = nn.Linear(hidden_dim * 2, 1)
        # Origin / Dest heads on the fused representation
        self.origin_head = nn.Sequential(
            nn.Linear(hidden_dim, pair_hidden),
            nn.ReLU(),
            nn.Linear(pair_hidden, pair_rank),
        )
        self.dest_head = nn.Sequential(
            nn.Linear(hidden_dim, pair_hidden),
            nn.ReLU(),
            nn.Linear(pair_hidden, pair_rank),
        )
        self.hidden_dim = hidden_dim
        self.pair_rank = pair_rank
        self._cached_M: Optional[torch.Tensor] = None
        self._cached_edge_key: Optional[tuple] = None

    def _get_M(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        try:
            edge_sum = int(edge_index.sum().item())
        except Exception:
            edge_sum = 0
        key = (num_nodes, tuple(edge_index.shape), edge_sum)
        if self._cached_edge_key != key or self._cached_M is None:
            self._cached_M = _build_neighbour_index(edge_index, num_nodes, None)
            self._cached_edge_key = key
        return self._cached_M

    def _compute_h_combined(
        self, X_static: torch.Tensor, X_dynamic: torch.Tensor, edge_index: torch.Tensor,
    ) -> torch.Tensor:
        N = X_static.shape[0]
        T = X_dynamic.shape[0]
        assert X_dynamic.shape[1] == N
        M = self._get_M(edge_index, N)
        h_s = self.static_branch(X_static, M)                          # (N, H)
        h_d = self.dyn_branch(X_dynamic, M)                            # (T, N, H)
        H = h_s.shape[-1]
        h_s_b = h_s.unsqueeze(0).expand(T, N, H)                       # (T, N, H)
        gate = torch.sigmoid(self.gate_net(torch.cat([h_s_b, h_d], dim=-1)))  # (T, N, 1)
        return gate * h_s_b + (1.0 - gate) * h_d                       # (T, N, H)

    def _compute_V_pair(self, h_combined: torch.Tensor) -> torch.Tensor:
        e_origin = self.origin_head(h_combined)                        # (T, N, R)
        e_dest = self.dest_head(h_combined)                            # (T, N, R)
        return torch.einsum("tir,tjr->tij", e_origin, e_dest)          # (T, N, N)

    def forward(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = self._compute_h_combined(X_static, X_dynamic, edge_index)
        V_pair = self._compute_V_pair(h)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._compute_h_combined(X_static, X_dynamic, edge_index)
        V_pair = self._compute_V_pair(h)
        return (V_pair.mean().detach(), V_pair.std().detach())
