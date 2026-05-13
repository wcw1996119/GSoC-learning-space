"""Structural GNN producing V_jt for inverse-RUM with do(X) interventions.

Paper-A treats V_jt = GNN_theta(X_j) as a *structural* function so that
counterfactual node-feature interventions ``do(X_S = x*)`` can be propagated
through the graph.  This module is intentionally:

  * pure-PyTorch (no torch_geometric — the host env doesn't have it),
  * GraphSAGE-style mean aggregation (cheaper / simpler than GAT for the
    inverse loop, and per spec "default GraphSAGE since simpler"),
  * temporally pooled with a small GRU head.

It exposes ``forward_under_intervention`` which clamps a subset of node
features to a target value before the forward pass — used by the
scenario-simulation pipeline.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_neighbour_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert (2, E) edge_index into a sparse neighbour-aggregation matrix.

    Returns a sparse COO tensor M of shape (N, N) so that M @ X computes
    the (weighted) neighbour-aggregate for every node. Self-loops are
    added so isolated nodes do not vanish.

    Parameters
    ----------
    edge_index   : (2, E) directed edges (src, dst).
    num_nodes    : N
    edge_weight  : optional (E,) per-edge weights (e.g. ``exp(-t_ij/tau)``).
                   If None, uses uniform weight=1 → mean aggregation
                   (original GraphSAGE behavior). When provided, weights are
                   row-normalised per destination so each row of M sums to 1.
                   Self-loop weight defaults to the mean of incoming edge
                   weights at each node.
    """
    src, dst = edge_index[0], edge_index[1]
    self_idx = torch.arange(num_nodes, device=edge_index.device)

    if edge_weight is None:
        # Uniform: mean over neighbours + self
        src_full = torch.cat([src, self_idx])
        dst_full = torch.cat([dst, self_idx])
        deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
        deg.index_add_(0, dst_full, torch.ones_like(dst_full, dtype=torch.float32))
        deg = deg.clamp(min=1.0)
        w = 1.0 / deg[dst_full]
    else:
        # Edge-weighted aggregation
        ew = edge_weight.to(torch.float32)
        # Self-loop weight = mean of incoming edge weights per node (or 1 if no incoming)
        sum_per_dst = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
        cnt_per_dst = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
        sum_per_dst.index_add_(0, dst, ew)
        cnt_per_dst.index_add_(0, dst, torch.ones_like(ew))
        self_w = (sum_per_dst / cnt_per_dst.clamp(min=1.0)).clamp(min=1e-6)
        # Concatenate edge weights + self-loop weights
        ew_full = torch.cat([ew, self_w])
        src_full = torch.cat([src, self_idx])
        dst_full = torch.cat([dst, self_idx])
        # Row-normalise per dst
        row_sum = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
        row_sum.index_add_(0, dst_full, ew_full)
        row_sum = row_sum.clamp(min=1e-6)
        w = ew_full / row_sum[dst_full]

    indices = torch.stack([dst_full, src_full], dim=0)
    M = torch.sparse_coo_tensor(indices, w, (num_nodes, num_nodes)).coalesce()
    return M


class _SAGELayer(nn.Module):
    """One GraphSAGE-mean layer: h_v = sigma(W_self h_v + W_neigh mean(h_u))."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim, bias=False)
        self.w_neigh = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim);  M: sparse (N, N)
        neigh = torch.sparse.mm(M, x)
        return F.relu(self.w_self(x) + self.w_neigh(neigh))


class _GATLayer(nn.Module):
    """One Graph Attention layer (Veličković et al. 2018), multi-head.

    For each destination node v and each neighbor u (including v itself):
        score_uv = LeakyReLU(a^T [W h_u || W h_v])      (per head)
        alpha_uv = softmax_v(score_uv)                   (per head)
        h'_v = concat_h(sigma(sum_u alpha_uv W_h h_u))    (concat heads)

    Parameters
    ----------
    in_dim, out_dim : feature dims (out is per-head; concat output is
                      out_dim * heads → projected back via final linear).
    heads           : number of attention heads (default 4).
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 negative_slope: float = 0.2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.w_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        # Attention coefficients: a_l (for source) and a_r (for target), both per head
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.out_proj = nn.Linear(out_dim * heads, out_dim, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """edge_index: (2, E) directed edges (src → dst). Self-loops added below."""
        device = x.device
        # Add self-loops
        self_idx = torch.arange(num_nodes, device=device)
        src = torch.cat([edge_index[0], self_idx])
        dst = torch.cat([edge_index[1], self_idx])

        # Project + reshape to multi-head
        h = self.w_proj(x).view(num_nodes, self.heads, self.out_dim)      # (N, H, D)
        # alpha contributions (one half from src, one half from dst, sum forms unnormalised score)
        alpha_src = (h * self.a_src).sum(dim=-1)                           # (N, H)
        alpha_dst = (h * self.a_dst).sum(dim=-1)                           # (N, H)

        # For each edge, score = alpha_src[src] + alpha_dst[dst]
        e_score = alpha_src[src] + alpha_dst[dst]                          # (E', H)
        e_score = F.leaky_relu(e_score, negative_slope=self.negative_slope)

        # Softmax over edges grouped by destination (per head)
        # subtract max-per-dst for numerical stability, then exp
        # we use scatter_max via a manual loop-free trick: subtract the running max
        e_score_safe = e_score - e_score.max()
        exp_e = e_score_safe.exp()                                          # (E', H)
        # row sum per dst per head
        denom = torch.zeros(num_nodes, self.heads, device=device, dtype=exp_e.dtype)
        denom.index_add_(0, dst, exp_e)
        denom = denom.clamp(min=1e-12)
        alpha = exp_e / denom[dst]                                          # (E', H)
        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Aggregate: for each dst, sum over its edges of alpha * h[src]
        # h[src] is (E', H, D). multiply by alpha (E', H, 1).
        msg = h[src] * alpha.unsqueeze(-1)                                  # (E', H, D)
        out = torch.zeros(num_nodes, self.heads, self.out_dim, device=device, dtype=msg.dtype)
        out.index_add_(0, dst, msg)                                         # (N, H, D)

        # Concat heads + project back
        out = out.reshape(num_nodes, self.heads * self.out_dim)
        out = self.out_proj(out)
        return F.elu(out)


class StructuralGNN(nn.Module):
    """GraphSAGE encoder + GRU temporal head producing a scalar V_jt.

    Parameters
    ----------
    node_dim    : input node feature dim F
    hidden_dim  : per-layer hidden dim
    n_layers    : 2 or 3 SAGE layers (default 2)
    gru_hidden  : hidden size for temporal GRU
    """

    def __init__(
        self,
        node_dim: Optional[int] = None,
        hidden_dim: int = 64,
        n_layers: int = 2,
        gru_hidden: int = 32,
        # Legacy / experiment-side aliases. Accepted so call sites in
        # experiments/paper_a/* (which use ``StructuralGNN(in_features=..., hidden=...,
        # out=1, depth=3)``) work without further edits.
        in_features: Optional[int] = None,
        hidden: Optional[int] = None,
        depth: Optional[int] = None,
        out: int = 1,
        layer_type: str = "sage",            # "sage" or "gat"
        gat_heads: int = 4,
    ):
        super().__init__()
        # Resolve aliases.
        if node_dim is None:
            node_dim = in_features
        if node_dim is None:
            raise TypeError("StructuralGNN requires node_dim (or in_features=)")
        if hidden is not None:
            hidden_dim = hidden
        if depth is not None:
            n_layers = depth
        self.layer_type = layer_type
        self.gat_heads = gat_heads
        # Allow depth>3 (experiments pass depth=3) but keep cheap default of 2.
        assert n_layers >= 1, "n_layers must be >= 1"
        assert layer_type in ("sage", "gat"), f"layer_type must be sage or gat; got {layer_type}"
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            if layer_type == "sage":
                self.layers.append(_SAGELayer(d_in, d_out))
            else:  # gat
                self.layers.append(_GATLayer(d_in, d_out, heads=gat_heads))
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=False)
        self.head = nn.Linear(gru_hidden, out)
        self._cached_M: Optional[torch.Tensor] = None
        self._cached_edge_key: Optional[tuple] = None
        # Optional default edge_weight applied automatically in forward when
        # caller does not pass one (e.g. when invoked from InverseRUMTrainer
        # which doesn't know about edge_weight). Set via ``set_default_edge_weight``.
        self._default_edge_weight: Optional[torch.Tensor] = None

    def set_default_edge_weight(self, edge_weight: Optional[torch.Tensor]) -> None:
        """Stash an edge_weight to be used by all subsequent forward() calls
        that don't explicitly pass one. Pass ``None`` to revert to uniform."""
        self._default_edge_weight = edge_weight
        # Invalidate cache so next forward rebuilds M with new weights
        self._cached_edge_key = None
        self._cached_M = None

    def _get_M(self, edge_index: torch.Tensor, num_nodes: int,
               edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            edge_sum = int(edge_index.sum().item())
        except Exception:
            edge_sum = 0
        ew_sig = 0.0 if edge_weight is None else float(edge_weight.sum().item())
        key = (num_nodes, tuple(edge_index.shape), edge_sum, ew_sig)
        if self._cached_edge_key != key or self._cached_M is None:
            self._cached_M = _build_neighbour_index(edge_index, num_nodes, edge_weight)
            self._cached_edge_key = key
        return self._cached_M

    def _encode(self, x: torch.Tensor, M: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                num_nodes: Optional[int] = None) -> torch.Tensor:
        h = x
        for layer in self.layers:
            if isinstance(layer, _GATLayer):
                h = layer(h, edge_index, num_nodes)
            else:
                h = layer(h, M)
        return h

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        hour_of_day: Optional[torch.Tensor] = None,
        norm_stats: Optional[tuple] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x_seq       : (T, N, F) or (N, F) node features. If 2D, treated as a
                      single time step (T=1) and the leading axis is squeezed
                      out of the return so callers expecting (N, out) get it.
        edge_index  : (2, E) directed edges.
        hour_of_day : optional (T,) — currently unused but kept in the signature
                      so callers can wire in a positional encoding without an API
                      change.
        norm_stats  : optional (mean, std) tuple of scalars. If supplied, used
                      to normalise the head output instead of recomputing on
                      the current forward batch — required for counterfactual
                      forwards (Scenario A/B) so that interventions are not
                      washed out by re-centring/re-scaling on the perturbed
                      output. Use ``compute_norm_stats(X_baseline, ...)`` to
                      obtain these from the baseline input.

        Returns
        -------
        V_jt : (T, N) if head out_dim == 1 else (T, N, out_dim); when called
               with 2D x_seq, returns (N,) or (N, out_dim) respectively.
        """
        squeeze_T = False
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
            squeeze_T = True
        T, N, _ = x_seq.shape
        # Resolve edge_weight: explicit arg > stored default > None (uniform)
        ew = edge_weight if edge_weight is not None else self._default_edge_weight
        M = self._get_M(edge_index, N, edge_weight=ew)
        h_list = [self._encode(x_seq[t], M, edge_index=edge_index, num_nodes=N) for t in range(T)]
        h_stack = torch.stack(h_list, dim=0)            # (T, N, hidden)
        z_stack, _ = self.gru(h_stack)                  # (T, N, gru_hidden)
        out = self.head(z_stack)                        # (T, N, out_dim)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)                       # (T, N)
        # Identifiability: pin V_jt scale by zero-mean / unit-std normalisation.
        # The product alpha * V_jt is identified only up to a positive scale
        # when V_jt is itself learnable; alpha is fixed at 1.0 (see _RUMHead)
        # and we additionally normalise V_jt so the GNN cannot trivially balloon
        # to absorb other parameters' variance. Done across all destinations
        # (and time steps when present) so V_jt has unit std overall.
        if norm_stats is None:
            mean_v = out.mean()
            std_v = out.std() + 1e-6
        else:
            mean_v, std_v = norm_stats
            std_v = std_v + 1e-6
        out = (out - mean_v) / std_v
        if squeeze_T:
            out = out.squeeze(0)
        return out

    def compute_norm_stats(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple:
        """Return (mean, std) of the un-normalised head output for ``x_seq``.

        Use this to freeze the V_jt scaling at the baseline scene so that
        counterfactual forwards with the same GNN report a structural ΔV
        rather than a re-centred-on-the-intervened-distribution ΔV.
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
        T, N, _ = x_seq.shape
        M = self._get_M(edge_index, N)
        h_list = [self._encode(x_seq[t], M, edge_index=edge_index, num_nodes=N) for t in range(T)]
        h_stack = torch.stack(h_list, dim=0)
        z_stack, _ = self.gru(h_stack)
        out = self.head(z_stack)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return (out.mean().detach(), out.std().detach())

    def forward_under_intervention(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        intervention_mask: torch.Tensor,
        intervention_value: torch.Tensor,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward under do(X_S = x*).

        Parameters
        ----------
        intervention_mask  : bool (N, F) — True where the feature is clamped.
        intervention_value : (N, F) target values to clamp to.

        Implementation: we copy x_seq, overwrite the masked entries at every
        time step with the intervention value, and run the standard forward.
        Intervention is therefore *static* across hours; pass an (T, N, F)
        intervention_value if you need a time-varying do().
        """
        squeeze_T = False
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
            squeeze_T = True
        T = x_seq.shape[0]
        x_do = x_seq.clone()
        if intervention_value.dim() == 2:
            iv = intervention_value.unsqueeze(0).expand(T, -1, -1)
        else:
            iv = intervention_value
        if intervention_mask.dim() == 2:
            mask = intervention_mask.unsqueeze(0).expand(T, -1, -1)
        else:
            mask = intervention_mask
        x_do = torch.where(mask, iv, x_do)
        out = self.forward(x_do, edge_index, hour_of_day=hour_of_day)
        if squeeze_T and out.dim() >= 2 and out.shape[0] == 1:
            # forward already squeezed; fall through.
            pass
        return out
