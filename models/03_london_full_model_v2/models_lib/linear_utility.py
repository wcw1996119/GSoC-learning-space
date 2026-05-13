"""Linear utility — Step 2 simplified for v2.1.

V_ij^t = α1 · V_j(t) + α2 · OccMatch_ij - β · t_ij(t) - γ · log d_ij + bias

Replaces full NN-utility for v2.1 baseline (NN version was OOM-prone on full 1725×1725 grid).
The linear form is sufficient because the GNN encoder already captures rich spatial structure;
the utility just needs to combine four scalar signals.

Calibration: α1, α2, β, γ are learned via NLL on aggregate F_ij^t.
At inference (ABM), each agent computes their personal V_ij with their own SOC's OccMatch.
"""
import torch
import torch.nn as nn


class LinearUtility(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize at sensible values
        self.alpha_V = nn.Parameter(torch.tensor(1.0))      # GNN prior weight
        self.alpha_occ = nn.Parameter(torch.tensor(0.5))    # OccMatch weight
        self.beta = nn.Parameter(torch.tensor(0.07))         # cost sensitivity
        self.gamma = nn.Parameter(torch.tensor(0.5))         # distance decay
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, V_j_t, occ_match, t_ij_t, log_d_ij):
        return (
            self.alpha_V * V_j_t
            + self.alpha_occ * occ_match
            - self.beta * t_ij_t
            - self.gamma * log_d_ij
            + self.bias
        )


def per_hour_log_p(
    util: LinearUtility,
    V_j_h: torch.Tensor,        # (N,) at hour h
    om: torch.Tensor,           # (N, N)
    t_ij_h: torch.Tensor,       # (N, N) at hour h
    log_d_ij: torch.Tensor,     # (N, N)
):
    """Returns log P(j|i, hour=h) — shape (N, N), softmax over dim=1."""
    N = V_j_h.shape[0]
    Vj_b = V_j_h[None, :].expand(N, N)
    V_ij = util(Vj_b, om, t_ij_h, log_d_ij)
    return torch.log_softmax(V_ij, dim=1)
