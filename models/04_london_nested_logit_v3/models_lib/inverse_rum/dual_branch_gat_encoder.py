"""GAT-based dual branch encoder (Step 3 of GNN-side improvement).

Replaces _SAGELayer (mean-aggregation) in StaticBranch / DynamicBranch
with _GATLayer (Velickovic 2018 multi-head attention).

Reuses v2's _MultiScaleTCN and GatedFusion unchanged (they have no
graph dependency). v2 frozen — not modifying its files. _GATLayer
already exists in v2's structural_gnn.py; just import it.

Same forward signature as DualBranchEncoder so it drops in as a
swap target via CLI flag in the trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

V2_ROOT = Path(__file__).resolve().parents[3] / "03_london_full_model_v2"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from models_lib.inverse_rum.structural_gnn import _GATLayer  # noqa: E402
from models_lib.inverse_rum.dual_branch_encoder import (  # noqa: E402
    _MultiScaleTCN,
    GatedFusion,
)


class StaticBranchGAT(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 32,
                 n_layers: int = 2, gat_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.layers.append(_GATLayer(d_in, d_out, heads=gat_heads))

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        h = X
        for layer in self.layers:
            h = layer(h, edge_index, num_nodes)
        return h


class DynamicBranchGAT(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 32,
                 n_gat_layers: int = 2, gru_hidden: int = 32,
                 tcn_kernels: Tuple[int, ...] = (3, 5, 7),
                 gat_heads: int = 4):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * n_gat_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.gat_layers.append(_GATLayer(d_in, d_out, heads=gat_heads))
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=False)
        self.tcn = _MultiScaleTCN(hidden_dim, kernels=tcn_kernels)
        self.fuse = nn.Linear(gru_hidden + hidden_dim, hidden_dim)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        T, N, _ = X.shape
        h_list = []
        for t in range(T):
            h = X[t]
            for layer in self.gat_layers:
                h = layer(h, edge_index, num_nodes)
            h_list.append(h)
        h_stack = torch.stack(h_list, dim=0)
        gru_out, _ = self.gru(h_stack)
        tcn_out = self.tcn(h_stack)
        fused = self.fuse(torch.cat([gru_out, tcn_out], dim=-1))
        return F.relu(fused)


class DualBranchGATEncoder(nn.Module):
    """Drop-in replacement for DualBranchEncoder using GAT layers.

    Identical forward signature: forward(X_static, X_dynamic, edge_index,
    norm_stats=None) -> (T, N) z-normalised V.
    """

    def __init__(
        self,
        static_dim: int,
        dyn_dim: int,
        hidden_dim: int = 32,
        gru_hidden: int = 32,
        n_gat_layers: int = 2,
        gat_heads: int = 4,
        tcn_kernels: Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        self.static_branch = StaticBranchGAT(
            static_dim, hidden_dim, n_layers=n_gat_layers, gat_heads=gat_heads
        )
        self.dyn_branch = DynamicBranchGAT(
            dyn_dim, hidden_dim, n_gat_layers=n_gat_layers,
            gru_hidden=gru_hidden, tcn_kernels=tcn_kernels, gat_heads=gat_heads
        )
        self.fusion = GatedFusion(hidden_dim)

    def forward(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        N = X_static.shape[0]
        h_s = self.static_branch(X_static, edge_index, N)
        h_d = self.dyn_branch(X_dynamic, edge_index, N)
        V = self.fusion(h_s, h_d)
        if norm_stats is None:
            mean_v = V.mean()
            std_v = V.std() + 1e-6
        else:
            mean_v, std_v = norm_stats
            std_v = std_v + 1e-6
        return (V - mean_v) / std_v
