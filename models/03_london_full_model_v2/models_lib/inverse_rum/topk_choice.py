"""McFadden (1978) sampling-of-alternatives for top-K choice sets.

For each origin we sample K destinations from a *consideration prior*
proportional to ``softmax(V_jt)`` (the structural value), then evaluate the
likelihood inside that subset.  McFadden's correction
    log L_corrected = log P(j | C_i) - log q(C_i | i, j)
keeps the MLE consistent provided the sampling distribution q is positive
everywhere on the universe and known.

Per Paper-A spec we support K ∈ {20, 50, 100} so the choice-set size can be
ablated.

SAMPLING SCHEME (B4 fix)
------------------------
We use **with-replacement multinomial sampling** from softmax(prior_logits).
Concretely, each of the K alternatives is drawn IID with probability
``q_j = softmax(prior_logits)[j]`` and ``log q(j) = log softmax(...)[j]``
is recorded per slot. With-replacement draws make the per-alternative
log-q exact and therefore make the McFadden 1978 importance correction
unbiased without needing the Kool et al. 2019 Gumbel-top-K finite-sample
correction. The downside is the choice set may contain duplicates; the
trainer treats duplicates as multiple "draws" of the same alternative,
which is consistent with the IID-sampling derivation.

Reference: McFadden (1978) "Modelling the choice of residential location",
in Karlqvist et al. (eds) *Spatial Interaction Theory and Planning Models*.
Reference: Ben-Akiva & Lerman (1985) ch. 9 for sampling-of-alternatives.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class TopKChoiceSet:
    """Reusable helper.

    Parameters
    ----------
    K       : choice-set size (default 50).
    rng     : numpy Generator. If None a fresh default_rng() is used.
    include_chosen : if True, the observed chosen alternative is force-added
                     to the sample (standard practice in McFadden 1978).
    """

    def __init__(self, K: int = 50, rng: Optional[np.random.Generator] = None,
                 include_chosen: bool = True):
        if K not in (20, 50, 100):
            # not a hard error — but spec calls these out
            pass
        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.include_chosen = include_chosen

    def sample(
        self,
        origin_idx: int,
        prior_logits: torch.Tensor,
        chosen_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one choice set for one origin.

        Parameters
        ----------
        origin_idx   : origin id (used only to make logging easier).
        prior_logits : (N,) prior logits over destinations (e.g., V_jt for hour t).
        chosen_idx   : observed destination, force-added if include_chosen.

        Returns
        -------
        idx          : (K,) long tensor of sampled destination indices.
        log_q        : (K,) log-probability of each sampled element under the
                       sampling distribution q (used in the correction term).
        """
        N = prior_logits.shape[0]
        K = self.K
        # softmax(prior_logits) is the sampling distribution q.
        logits_np = prior_logits.detach().cpu().numpy().astype(np.float64)
        # numerical stability
        logits_np = logits_np - logits_np.max()
        log_q_full = logits_np - np.log(np.exp(logits_np).sum())
        q_full = np.exp(log_q_full)
        q_full = q_full / q_full.sum()  # exact normalisation against fp drift
        # B4 fix: with-replacement multinomial sampling. Each slot is an
        # independent draw from q_full, so log_q_per_slot is exact and the
        # McFadden 1978 importance correction is unbiased.
        if self.include_chosen and chosen_idx is not None:
            # First slot is forced to the chosen alternative; remaining K-1
            # slots are IID samples from q_full. We record the *natural* log_q
            # for the forced slot (which is what the McFadden correction
            # expects when chosen is "known to be in the set"; the orchestrator
            # is responsible for picking the chosen-slot likelihood, not for
            # weighting the forced slot's draw probability).
            n_random = max(K - 1, 0)
            random_draws = self.rng.choice(N, size=n_random, replace=True, p=q_full)
            topk = np.empty(K, dtype=np.int64)
            topk[0] = int(chosen_idx)
            if n_random > 0:
                topk[1:] = random_draws
        else:
            topk = self.rng.choice(N, size=K, replace=True, p=q_full).astype(np.int64)
        idx = torch.tensor(topk, dtype=torch.long, device=prior_logits.device)
        log_q = torch.tensor(log_q_full[topk], dtype=prior_logits.dtype,
                             device=prior_logits.device)
        return idx, log_q

    def correction_logp(
        self,
        log_p_in_set: torch.Tensor,
        log_q: torch.Tensor,
    ) -> torch.Tensor:
        """McFadden importance-correction:

            log L = log P(j | C_i) - log q(C_i | i, j)

        Implemented as ``log_p_in_set - log_q``. Caller selects the entry
        corresponding to the observed chosen alternative.
        """
        return log_p_in_set - log_q


def make_choice_set(
    origin_idx: int,
    V_jt: torch.Tensor,
    K: int = 50,
    rng: Optional[np.random.Generator] = None,
    chosen_idx: Optional[int] = None,
    include_chosen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional wrapper around ``TopKChoiceSet.sample``.

    Parameters
    ----------
    V_jt : (N,) prior logits for one hour.

    Returns
    -------
    idx, log_q : as in TopKChoiceSet.sample.
    """
    helper = TopKChoiceSet(K=K, rng=rng, include_chosen=include_chosen)
    return helper.sample(origin_idx, V_jt, chosen_idx=chosen_idx)
