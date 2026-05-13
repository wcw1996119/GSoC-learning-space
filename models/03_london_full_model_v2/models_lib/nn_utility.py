"""NN-utility module — Step 2 of v2 4-step architecture.

V_ij^t = NN([V_j(t), OccMatch_ij, t_ij^t, log d_ij])

A small 2-layer MLP that combines:
  - V_j(t)         : "macro destination attractiveness" from STGNN encoder
  - OccMatch_ij    : agent's SOC × destination industry fit
  - t_ij^t         : congestion-aware travel time at hour t
  - log d_ij       : log distance prior (helps gravity-style decay)

Output is a scalar utility V_ij^t.
RUM closure (per origin × hour): P(j|i,t) = softmax_j(V_ij^t)
"""
import torch
import torch.nn as nn


class NNUtility(nn.Module):
    """
    Per (i, j, t) utility evaluation.

    Inputs (broadcast-ready):
      V_j_t       : (..., 1)  destination attractiveness (from STGNN)
      occ_match   : (..., 1)  OccMatch (cosine similarity)
      t_ij_t      : (..., 1)  travel time minutes
      log_d_ij    : (..., 1)  log distance km

    Output:
      V_ij_t      : (...)     scalar utility per OD-pair-time
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, V_j_t, occ_match, t_ij_t, log_d_ij):
        x = torch.stack([V_j_t, occ_match, t_ij_t, log_d_ij], dim=-1)
        return self.net(x).squeeze(-1)


def rum_with_nn_utility(
    V_jt: torch.Tensor,             # (T, N)
    occ_match_avg: torch.Tensor,    # (N_origin, N_dest)  — aggregate at training
    t_ij_t: torch.Tensor,           # (T, N, N)
    log_d_ij: torch.Tensor,         # (N, N)
    nn_util: NNUtility,
) -> torch.Tensor:
    """
    Compute log P(j|i, t) via NN-utility + softmax.

    Returns: (T, N_origin, N_dest) log probabilities.
    """
    T, N = V_jt.shape

    # Broadcast all to (T, N, N)
    Vj = V_jt[:, None, :].expand(T, N, N)              # V_j(t) at destination
    om = occ_match_avg[None, :, :].expand(T, N, N)      # (T, N_o, N_d)
    tt = t_ij_t                                          # (T, N, N)
    d  = log_d_ij[None, :, :].expand(T, N, N)            # (T, N, N)

    V_ij = nn_util(Vj, om, tt, d)                        # (T, N, N)
    log_p = torch.log_softmax(V_ij, dim=2)
    return log_p
