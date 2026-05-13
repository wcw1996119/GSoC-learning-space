"""RUM closure: V_j(t) - β·t_ij(t) → softmax → P(j|i, t).

Production-constrained: softmax over destinations (j) for each origin (i).
β fixed initially per methodology/05_training_loss.md (WebTAG VOT, ~0.07/min).
"""
import torch


def rum_closure(V_jt: torch.Tensor, t_ij_t: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute P(j | i, t) for all (i, j, t) under production-constrained MNL.

    V_jt:   (T, N)         — destination attractiveness per hour
    t_ij_t: (T, N, N) or (N, N) — travel time per hour (broadcast if 2D)
    beta:   float          — cost sensitivity per minute

    Returns log P(j|i,t):  (T, N, N) where dim=2 is destination.
    For each (t, i): logsumexp over j of (V_j(t) - beta * t_ij(t)) is the partition.
    """
    T, N = V_jt.shape
    if t_ij_t.dim() == 2:
        # broadcast (N,N) → (T,N,N)
        t_ij_t = t_ij_t.unsqueeze(0).expand(T, N, N)

    # logits[t, i, j] = V_j(t) - beta * t_ij(t)
    V_expand = V_jt.unsqueeze(1).expand(T, N, N)        # V_j(t) along j-dim
    logits = V_expand - beta * t_ij_t                    # (T, N, N)
    log_p = torch.log_softmax(logits, dim=2)             # softmax over j
    return log_p
