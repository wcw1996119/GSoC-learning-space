"""Spatial-Temporal GNN — T-GCN style (GAT spatial + GRU temporal).

Architecture per methodology/05_training_loss.md:
    Per hour t:
      x_t (N, F)  -->  GATv2(x_t, edge_attr) --> GATv2 --> h_t (N, hidden)
    Then over time:
      [h_0, h_1, ..., h_T-1]  -->  GRU  -->  z_t (N, hidden)
    Finally:
      V_j(t) = MLP(z_t)  scalar per (j, t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class V2_STGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,       # F_static + F_temporal
        edge_dim: int,        # 3 (d_log, t0_log, gauss)
        hidden_dim: int = 128,
        gat_heads: int = 4,
        gru_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat1 = GATv2Conv(
            node_dim, hidden_dim,
            edge_dim=edge_dim, heads=gat_heads, dropout=dropout
        )
        self.gat2 = GATv2Conv(
            hidden_dim * gat_heads, hidden_dim,
            edge_dim=edge_dim, heads=1, dropout=dropout
        )
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=False)
        self.out_head = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(self, x_seq, edge_index, edge_attr):
        """
        x_seq:      (T, N, node_dim)  hour-stacked node features
        edge_index: (2, E)
        edge_attr:  (E, edge_dim)

        Returns:
          V_jt: (T, N) — destination attractiveness per (hour, grid)
        """
        T, N, F_in = x_seq.shape
        h_seq = []
        for t in range(T):
            h = self.gat1(x_seq[t], edge_index, edge_attr)
            h = F.elu(h)
            h = self.gat2(h, edge_index, edge_attr)
            h = F.elu(h)
            h_seq.append(h)
        h_stack = torch.stack(h_seq, dim=0)        # (T, N, hidden_dim)

        # GRU expects (seq_len, batch, input_size)
        # Treat each grid as a separate "batch" so GRU runs N parallel sequences
        gru_in = h_stack                              # (T, N, hidden_dim)
        z_stack, _ = self.gru(gru_in)                # (T, N, gru_hidden)

        V_jt = self.out_head(z_stack).squeeze(-1)    # (T, N)
        return V_jt
