"""Dual-Branch ST-GNN encoder.

Computes V_jt (T, N) for the inverse-RUM trainer. Architecture:

    Static branch (time-invariant features X_static (N, F_s)):
        GraphSAGE × n_layers → A_static (N, hidden)

    Dynamic branch (per-hour features X_dyn (T, N, F_d)):
        For each hour t: GraphSAGE × n_layers → h_t (N, hidden)
        Stack to (T, N, hidden); pass through:
          - GRU along T axis  → gru_out (T, N, gru_hidden)
          - Multi-scale TCN (kernels 3, 5, 7 on T axis with circular padding)
            → tcn_out (T, N, hidden)
        Fuse via linear: dyn_out = Linear([gru_out | tcn_out]) → (T, N, hidden)

    Gated fusion:
        gate(t, j) = σ(Linear([A_static_j | dyn_out_{t,j}]))
        h_combined = gate * A_static_broadcast + (1 - gate) * dyn_out
        V_jt = Linear(h_combined) → (T, N)

    Identifiability normalisation (matches StructuralGNN):
        V_jt ← (V_jt - mean) / std

Why "dual-branch with multi-scale TCN":
- Static features (POI, employment, population) genuinely don't change hourly
  → wasteful to broadcast into a temporal model. Static branch handles them
  once and outputs a single embedding per destination.
- Dynamic features (congestion, hourly inflow, queue residual) need a
  temporal mechanism to capture day-night cycles + sequential dependence.
- GRU captures sequential ordering ("8am state remembers 7am").
- Multi-scale TCN captures different temporal windows (3h = peak,
  5h = morning, 7h = peak+shoulder).
- Gated fusion lets the model decide per-(hour, grid) whether the static
  or dynamic signal dominates.

Reuses ``_SAGELayer`` from structural_gnn.py — same neighbour aggregation
used elsewhere in the codebase.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structural_gnn import _SAGELayer, _build_neighbour_index


class StaticBranch(nn.Module):
    """GraphSAGE encoder for time-invariant features.

    Input  : X_static (N, F_s), edge_index (2, E)
    Output : h_static (N, hidden)
    """

    def __init__(self, node_dim: int, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        assert n_layers >= 1
        self.layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.layers.append(_SAGELayer(d_in, d_out))

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        h = X
        for layer in self.layers:
            h = layer(h, M)
        return h


class _MultiScaleTCN(nn.Module):
    """Three parallel 1D causal convs along the time axis, kernels (3, 5, 7).

    Operates on (T, N, C) → permutes to (N, C, T) for conv1d, then back.
    Uses circular padding so the 24-hour cycle wraps (8am features can see
    7am via causal padding).
    """

    def __init__(self, hidden_dim: int, kernels: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.kernels = kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2,
                      padding_mode="circular")
            for k in kernels
        ])
        self.proj = nn.Linear(hidden_dim * len(kernels), hidden_dim)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        # h_seq: (T, N, C) → (N, C, T) for conv → back
        T, N, C = h_seq.shape
        x = h_seq.permute(1, 2, 0)                    # (N, C, T)
        outs = []
        for conv in self.convs:
            y = F.relu(conv(x))                       # (N, C, T)
            outs.append(y)
        cat = torch.cat(outs, dim=1)                  # (N, C * len(kernels), T)
        cat = cat.permute(2, 0, 1)                    # (T, N, C * K)
        return self.proj(cat)                         # (T, N, C)


class DynamicBranch(nn.Module):
    """Per-hour GraphSAGE + GRU + multi-scale TCN.

    Input  : X_dyn (T, N, F_d), edge_index (2, E)
    Output : h_dyn (T, N, hidden)
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 32,
        n_sage_layers: int = 2,
        gru_hidden: int = 32,
        tcn_kernels: Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        self.sage_layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * n_sage_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.sage_layers.append(_SAGELayer(d_in, d_out))
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=False)
        self.tcn = _MultiScaleTCN(hidden_dim, kernels=tcn_kernels)
        # Fuse GRU + TCN outputs to a common hidden_dim
        self.fuse = nn.Linear(gru_hidden + hidden_dim, hidden_dim)

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        T, N, _ = X.shape
        # Per-hour SAGE
        h_list = []
        for t in range(T):
            h = X[t]
            for layer in self.sage_layers:
                h = layer(h, M)
            h_list.append(h)
        h_stack = torch.stack(h_list, dim=0)            # (T, N, hidden)
        # GRU: provides sequential memory ("9am sees 8am")
        gru_out, _ = self.gru(h_stack)                  # (T, N, gru_hidden)
        # Multi-scale TCN: provides multiple temporal-window views
        tcn_out = self.tcn(h_stack)                     # (T, N, hidden)
        fused = self.fuse(torch.cat([gru_out, tcn_out], dim=-1))   # (T, N, hidden)
        return F.relu(fused)


class GatedFusion(nn.Module):
    """Gated combination of static + dynamic branches → V_jt scalar.

    gate(t, j) = σ(Linear([h_s_j | h_d_{t,j}]))         scalar per (t, j)
    h_combined = gate * h_s_broadcast + (1 - gate) * h_d
    V_jt = Linear(h_combined)                          → (T, N)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_net = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h_static: torch.Tensor, h_dyn: torch.Tensor) -> torch.Tensor:
        # h_static: (N, hidden);  h_dyn: (T, N, hidden)
        T, N, H = h_dyn.shape
        h_s_b = h_static.unsqueeze(0).expand(T, N, H)                 # (T, N, hidden)
        gate = torch.sigmoid(self.gate_net(torch.cat([h_s_b, h_dyn], dim=-1)))   # (T, N, 1)
        h_combined = gate * h_s_b + (1.0 - gate) * h_dyn               # (T, N, hidden)
        v = self.head(h_combined).squeeze(-1)                           # (T, N)
        return v


class DualBranchEncoder(nn.Module):
    """Top-level encoder: combines StaticBranch + DynamicBranch via GatedFusion.

    Output V_jt is z-normalised (zero mean, unit std) over all (t, j) so that
    α (fixed at 1.0 in _RUMHead) does not need to absorb V's scale and the
    inverse-RUM identification is well-defined.

    Parameters
    ----------
    static_dim, dyn_dim : input feature dims
    hidden_dim          : shared hidden for both branches
    gru_hidden          : GRU state size (dynamic branch)
    n_sage_layers       : SAGE depth (same for both branches)
    """

    def __init__(
        self,
        static_dim: int,
        dyn_dim: int,
        hidden_dim: int = 32,
        gru_hidden: int = 32,
        n_sage_layers: int = 2,
        tcn_kernels: Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        self.static_branch = StaticBranch(static_dim, hidden_dim, n_layers=n_sage_layers)
        self.dyn_branch = DynamicBranch(
            dyn_dim, hidden_dim, n_sage_layers=n_sage_layers,
            gru_hidden=gru_hidden, tcn_kernels=tcn_kernels,
        )
        self.fusion = GatedFusion(hidden_dim)
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

    def forward(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Inputs
        ------
        X_static  : (N, F_s) — time-invariant per-grid features
        X_dynamic : (T, N, F_d) — per-hour per-grid features
        edge_index: (2, E) — fixed geographic adjacency
        norm_stats: optional (mean, std) tuple. Use baseline stats during
                    counterfactual forwards so interventions are not washed out.

        Returns
        -------
        V_jt : (T, N) — z-normalised structural utility per (hour, destination)
        """
        N = X_static.shape[0]
        T = X_dynamic.shape[0]
        assert X_dynamic.shape[1] == N, (
            f"X_dynamic shape {tuple(X_dynamic.shape)} grid count != X_static N={N}"
        )
        M = self._get_M(edge_index, N)
        h_s = self.static_branch(X_static, M)               # (N, hidden)
        h_d = self.dyn_branch(X_dynamic, M)                  # (T, N, hidden)
        V = self.fusion(h_s, h_d)                            # (T, N)
        # Identifiability normalisation (matches StructuralGNN convention)
        if norm_stats is None:
            mean_v = V.mean()
            std_v = V.std() + 1e-6
        else:
            mean_v, std_v = norm_stats
            std_v = std_v + 1e-6
        return (V - mean_v) / std_v

    def compute_norm_stats(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of un-normalised V at this input — for freezing
        the V scale at baseline before doing counterfactual forwards."""
        N = X_static.shape[0]
        M = self._get_M(edge_index, N)
        h_s = self.static_branch(X_static, M)
        h_d = self.dyn_branch(X_dynamic, M)
        V = self.fusion(h_s, h_d)
        return (V.mean().detach(), V.std().detach())

    def forward_under_intervention(
        self,
        X_static: torch.Tensor,
        X_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        intervention_mask_static: Optional[torch.Tensor] = None,
        intervention_value_static: Optional[torch.Tensor] = None,
        intervention_mask_dyn: Optional[torch.Tensor] = None,
        intervention_value_dyn: Optional[torch.Tensor] = None,
        norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward under do() interventions on either branch.

        Static intervention (e.g. Scenario A: OOC +65k jobs):
            mask + value shape (N, F_s); clamped on X_static.

        Dynamic intervention (e.g. Scenario B: peak×0.5, shoulder×1.5):
            mask + value shape (T, N, F_d); clamped on X_dynamic.

        ``norm_stats`` should be the baseline (mean, std) so the perturbation
        is not absorbed by re-centring.
        """
        Xs = X_static.clone()
        Xd = X_dynamic.clone()
        if intervention_mask_static is not None and intervention_value_static is not None:
            Xs = torch.where(intervention_mask_static, intervention_value_static, Xs)
        if intervention_mask_dyn is not None and intervention_value_dyn is not None:
            Xd = torch.where(intervention_mask_dyn, intervention_value_dyn, Xd)
        return self.forward(Xs, Xd, edge_index, norm_stats=norm_stats)
