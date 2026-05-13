"""Production-constrained MNL log-likelihood.

L = -Σ_{(i,j,t) in batch} F_ij^t × log P(j|i, t)
"""
import torch


def production_constrained_mnl_nll(
    log_p: torch.Tensor,    # (T, N, N) log probabilities
    flow_target: torch.Tensor,  # (T, N, N) F_ij^t (sparse OK since multiplied with log_p)
    origin_mask: torch.Tensor | None = None,  # (N,) bool — 1 = origin in train set
) -> torch.Tensor:
    """
    Returns the average negative log-likelihood per origin × hour pair.

    flow_target should be (T, N, N) with F_ij^t. Origins outside origin_mask are ignored.
    """
    if origin_mask is not None:
        # zero out origins not in mask
        flow_target = flow_target * origin_mask.view(1, -1, 1).float()

    nll = -(flow_target * log_p).sum()  # scalar
    n_origins_active = (flow_target.sum(dim=2) > 0).sum().clamp(min=1)
    return nll / n_origins_active
