"""inverse_rum — Paper A core module.

Public API for inverse Random-Utility-Maximisation estimation:

  ImplicitSoftmax        — autograd.Function with implicit-diff backward
  StructuralGNN          — GNN producing V_jt with do(X) intervention support
  TopKChoiceSet          — McFadden-1978 sampling-of-alternatives helper
  InverseRUMTrainer      — joint (theta, alpha, beta, gamma) recovery loop

This package does NOT modify the existing v2.x utility / GNN code in
``models_lib/{wage_utility,heterogeneous_utility,stgnn,rum}.py`` — it is
an additive Paper-A module. It depends only on torch (no torch_geometric,
no JAX) so it runs in the lightweight ``mesa-demo`` env.
"""
from .implicit_softmax import ImplicitSoftmax, implicit_softmax, nested_logit_logsum
from .structural_gnn import StructuralGNN
from .topk_choice import TopKChoiceSet, make_choice_set
from .inverse_trainer import InverseRUMTrainer
from .hessian_ci import hessian_ci_rum, sandwich_ci_rum
from .bpr_layer import bpr_multiplier, apply_bpr, solve_user_equilibrium, UEResult
from .accessibility import hansen_accessibility
from .inequality import gini, palma_ratio, atkinson, all_indices
from .agent_heterogeneity import (
    AgentBetaConfig, DEFAULT_COV_BY_TIER, WEBTAG_VOT_BY_TIER_GBP_PER_HOUR,
    assign_per_agent_epsilon, per_agent_beta,
    simulate_agent_choice_softmax, aggregate_per_agent_to_OD,
)

__all__ = [
    "ImplicitSoftmax",
    "implicit_softmax",
    "nested_logit_logsum",
    "StructuralGNN",
    "TopKChoiceSet",
    "make_choice_set",
    "InverseRUMTrainer",
    "hessian_ci_rum",
    "sandwich_ci_rum",
    "bpr_multiplier",
    "apply_bpr",
    "solve_user_equilibrium",
    "UEResult",
    "hansen_accessibility",
    "gini",
    "palma_ratio",
    "atkinson",
    "all_indices",
    "AgentBetaConfig",
    "DEFAULT_COV_BY_TIER",
    "WEBTAG_VOT_BY_TIER_GBP_PER_HOUR",
    "assign_per_agent_epsilon",
    "per_agent_beta",
    "simulate_agent_choice_softmax",
    "aggregate_per_agent_to_OD",
]
