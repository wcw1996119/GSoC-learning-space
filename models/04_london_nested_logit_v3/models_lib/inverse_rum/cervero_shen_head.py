"""CerveroShenHead — D→M nested logit with Cervero match + Shen competition.

v3 redesign of dm_nested_logit_head.py:

  Upper nest (destination):
      V_dest(i, j, t) = α · log(W_j)               ← wage attractiveness (NEW)
                      + γ · log(M_j)               ← jobs count, Hansen gravity (NEW form)
                      + ν · log(D_j)               ← Shen competition (NEW)
                      + δ · match_prob[i, j]       ← Cervero match (NEW)
                      + λ_b(j) · IV_mode(i, j, t)  ← inclusive value from lower nest
                      + (blend · V_GNN_jt)         ← optional Wang TB-ResNet

  Lower nest (mode | destination):
      V_mode(i, j, t, m) = ASC_m
                         + β_t_m · t_{ij,m}        ← linear time disutility per mode
                         + θ_inc_m · income_i      ← income×mode interaction (V_lower scope)

  Where:
      λ_b ∈ [ε_min, 1] via λ_b = ε_min + (1-ε_min) · sigmoid
      IV_mode(i,j,t) = log Σ_m exp(V_mode / λ_{b(j)})

Anchored to:
  - Cervero R, Rood T, Appleyard B (1999) — occupational match formula
  - Shen Q (1998) — competition-adjusted accessibility
  - Hansen WG (1959) — gravity model
  - McFadden D (1973) — conditional logit / nested logit

Changes vs v3 DMNestedLogitHead:
  REMOVED:
    γ · log_d_ij         (distance decay redundant with t·β captured in IV)
    β_slope · log_d      (per-mode log_d slope — redundant)
    δ_slope · log_d      (match × distance interaction — behaviorally unjustified)
    θ_per_mode · X_mode-level features (Hansen accessibility / mean_t / n_jobs_30min)

  ADDED:
    α · log(W_j)         (wage attractiveness)
    ν · log(D_j)         (Shen competition)
    δ · match_prob       (Cervero match — replaces old cosine OccMatch)
    θ_inc_m · income_i   (income × mode interaction in V_lower)

Parameter inventory (defaults M=3, B=33):
    raw_alpha_wage              scalar       wage coefficient (NEW, > 0)
    raw_gamma_M                 scalar       jobs coefficient (NEW, > 0)
    raw_nu_D                    scalar       competition coefficient (NEW, < 0)
    delta_match                 scalar       Cervero match coefficient (NEW)
    raw_beta_t_per_mode         (M,)         time disutility per mode
    asc_per_mode                (M,)         ASC per mode
    theta_inc_per_mode          (M,)         income × mode interaction
    raw_lambda_borough          (B,)         borough nest correlation
    raw_gnn_blend               scalar       (optional) Wang blend

Total free params (defaults): 1 + 1 + 1 + 1 + 3 + 3 + 3 + 33 + 1 = 47 RUM params
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    return math.log(math.exp(v) - 1.0)


def _inv_sigmoid(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return math.log(s / (1 - s))


class CerveroShenHead(nn.Module):
    """D→M nested logit head with Cervero match + Shen competition + Hansen attractiveness.

    Parameters
    ----------
    n_modes : int
        Number of modes (default 3 for car/transit/walk).
    n_boroughs : int
        Number of borough λ values to learn (default 33).
    alpha_wage_init : float
        Initial α (log W_j coefficient). Default 0.1, sign-free.
    gamma_M_init : float
        Initial γ (log M_j coefficient). Enforced > 0 via softplus. Default 0.5.
    nu_D_init : float
        Initial ν (log D_j coefficient). Enforced < 0 via -softplus. Default -0.1.
    delta_match_init : float
        Initial δ (match_prob coefficient). Sign-free. Default 1.0 (mild positive).
    beta_t_init : float
        Initial β_t,m (time disutility). Negative, enforced via -softplus. Default -0.07.
    theta_inc_init : float
        Initial θ_inc,m (income × mode interaction). Default 0.
    lambda_init : float
        Initial λ_b. Default 0.95 (close to flat MNL).
    lambda_eps_min : float
        Lower bound on λ_b. Default 0.05.
    use_gnn_blend : bool
        Enable Wang TB-ResNet blend.
    gnn_blend_init : float
        Initial blend. Default 0.5.
    blend_max : float
        Hard cap. Default 1.0.
    """

    def __init__(
        self,
        n_modes: int = 3,
        n_boroughs: int = 33,
        n_tiers: int = 3,
        alpha_wage_init: float = 0.1,
        gamma_M_init: float = 0.5,
        nu_D_init: float = -0.1,
        delta_match_init: float = 0.1,
        beta_t_init: float = -0.07,
        beta_t_slope_init: float = -0.01,
        theta_inc_init: float = 0.0,
        lambda_init: float = 0.95,
        lambda_eps_min: float = 0.05,
        use_gnn_blend: bool = False,
        gnn_blend_init: float = 0.5,
        blend_max: float = 1.0,
        gnn_mode: str = "convex",  # "convex" (v3b/c legacy) | "residual" (Wang TB-ResNet)
        gnn_residual_scale_init: float = 0.1,
        use_match_gate: bool = False,  # sigmoid gating on Cervero match (decision-tree style)
        gate_steepness_init: float = 1.0,
        gate_threshold_init: float = 0.15,
        use_tier_mixture: bool = False,  # income-tier latent class (3 discrete agent types)
        tier_init_scale: float = 0.0,    # 0 = symmetric init (collapse-prone);
                                          # >0 = tier k gets base × (1 + k·scale), breaks symmetry
                                          # towards lit-expected behavioral gradient
        use_push_pull: bool = False,     # origin labor surplus × dest M_j interaction (push-pull gravity)
        xi_push_pull_init: float = 0.0,  # free real init for push×pull coefficient
        use_self_loop_boost: bool = False,  # per-(origin borough, hour) intra-zone bias
        n_hours: int = 24,                  # T (used only if use_self_loop_boost)
        n_busy_dest: int = 0,               # K > 0: top-K busy destinations get
                                            # learnable per-hour boost (Option 2 — finer
                                            # than borough-level self-loop boost)
        n_origins: int = 0,                 # for moe mode: number of origin grids (=N)
        use_tier_threshold: bool = False,   # Bhat 1995 heterogeneous commute threshold:
                                            # per-tier T_k + β_kink_k for V_lower
        use_consideration_filter: bool = False,  # Lexicographic two-layer filter:
                                            # match_pass × cost_pass via sigmoid soft masks.
        use_stoll_match: bool = False,      # Stoll-Houston (2005) "effective jobs" coupling:
                                            # V_M = γ · log(M · match) = γ · (log_M + log_match)
                                            # Replaces δ_match · match · log_M (weak interaction)
                                            # with γ-weighted log_match (strong coupling).
                                            # Forces occupational match to have same elasticity
                                            # as gravity, matching common-sense intuition.
        match_thresh_floor: float = 0.0,    # Lower bound on match filter threshold.
                                            # Without this, model learns threshold ≈ 0 (no filtering).
                                            # Set to 0.30 to force "below 30% match = downweighted"
                                            # behaviour. Implemented via reparameterization:
                                            # thresh = floor + softplus(raw).
        k_match_init: float = 10.0,         # Initial sharpness of match filter sigmoid.
                                            # Larger = sharper cutoff. Default 10 (moderate).
                                            # Use ≥20 for near-hard cutoff behaviour.
    ):
        super().__init__()
        self.n_modes = n_modes
        self.n_boroughs = n_boroughs
        self.lambda_eps_min = float(lambda_eps_min)
        self.use_gnn_blend = use_gnn_blend
        self.blend_max = float(blend_max)
        self.gnn_mode = str(gnn_mode)
        assert self.gnn_mode in ("convex", "residual", "mult", "moe"), f"unknown gnn_mode={gnn_mode}"
        self.use_match_gate = bool(use_match_gate)
        self.use_stoll_match = bool(use_stoll_match)
        self.use_tier_mixture = bool(use_tier_mixture)
        self.n_income_tiers = int(n_tiers)
        self.use_push_pull = bool(use_push_pull)

        # V_upper destination-attractor params (α, γ, ν, δ_match)
        # If use_tier_mixture: each becomes (n_tiers,) tensor for tier-specific RUM
        # tier_init_scale > 0: tier k gets base × (1 + k·scale) to break symmetry &
        # encode lit-expected gradient (higher income → more selective)
        n_t = self.n_income_tiers if self.use_tier_mixture else 1

        def _tier_init(base_value: float, sign: int = 1) -> torch.Tensor:
            """Generate (n_t,) init values with lit-anchored tier gradient.
            sign=+1 for ≥0 params (α/γ/δ), -1 for ≤0 params (ν)."""
            target = max(abs(base_value), 1e-3)
            if n_t == 1:
                return torch.tensor(_inv_softplus(target))
            # tier 0 (low) gets base, tier k gets base · (1 + k · tier_init_scale)
            scaled = [target * (1.0 + k * tier_init_scale) for k in range(n_t)]
            raws = [_inv_softplus(max(v, 1e-3)) for v in scaled]
            return torch.tensor(raws, dtype=torch.float32)

        # α (wage) — enforced >= 0 via softplus
        self.raw_alpha_wage = nn.Parameter(_tier_init(alpha_wage_init, +1))

        # γ (M_j) — enforced > 0 via softplus
        self.raw_gamma_M = nn.Parameter(_tier_init(gamma_M_init, +1))

        # ν (D_j) — enforced < 0 via -softplus (use abs in init)
        self.raw_nu_D = nn.Parameter(_tier_init(-nu_D_init, -1))

        # δ (match_prob) — enforced >= 0 via softplus
        self.raw_delta_match = nn.Parameter(_tier_init(delta_match_init, +1))

        # Push-pull cross term: V += ξ · push_i · log_M_j
        #   push_i = log(workers reaching i) - log(jobs at i) = labor surplus indicator
        #   Lit anchor: gravity model push-pull (Wilson 1967, Schwanen 2003)
        #   ξ free real (sign learned from data); expected ξ > 0 if labor surplus
        #   amplifies preference for large-M_j destinations.
        if self.use_push_pull:
            self.xi_push_pull = nn.Parameter(torch.tensor(float(xi_push_pull_init)))
        else:
            self.xi_push_pull = None

        # Self-loop hour-of-day boost: per (origin borough, hour) bias for intra-zone
        # destination choice. Captures e.g. CBD lunch-time pulse where workers go to
        # destinations within their own borough — gravity model can't see this
        # because γ·log(M_j) treats self-loop like any other destination.
        # Residual diag finding: V_dest peak for CBD self-loops should be at midday
        # (lunch) not evening rush.
        self.use_self_loop_boost = bool(use_self_loop_boost)
        if self.use_self_loop_boost:
            self.self_loop_boost = nn.Parameter(torch.zeros(n_boroughs, n_hours))
        else:
            self.self_loop_boost = None

        # Top-K busy destination boost (Option 2)
        # Borough-level self_loop_boost can't resolve grid-level patterns within
        # large CBD boroughs (e.g. b=32 covers 816, 770, 726). This adds a
        # learnable per-(busy destination grid, hour) bias.
        # Implementation: dest_to_k_idx is a (N,) buffer mapping grid → busy_idx
        # (or -1 if not busy). busy_dest_boost has shape (K, T).
        # V_dest[t, i, j] += busy_dest_boost[dest_to_k[j], t]  (zero for non-busy j)
        self.n_busy_dest = int(n_busy_dest)
        if self.n_busy_dest > 0:
            self.busy_dest_boost = nn.Parameter(torch.zeros(self.n_busy_dest, n_hours))
            # Will be filled by trainer via set_busy_dest_index() after data load
            self.register_buffer("busy_dest_to_k_idx", torch.full((1,), -1, dtype=torch.long))
        else:
            self.busy_dest_boost = None
            self.busy_dest_to_k_idx = None

        # β_t,0 per mode (intercept) — enforced < 0 via -softplus
        target_b = -beta_t_init
        raw_b = _inv_softplus(target_b)
        self.raw_beta_t_per_mode = nn.Parameter(torch.full((n_modes,), raw_b))

        # β_t,1 per mode (slope on log_d) — enforced <= 0 via -softplus.
        # Captures distance-dependent VOT: at longer trips, each commute minute hurts MORE.
        # Lit anchor: Wachs et al. 1993, Levinson 2010 distance-dependent value of time.
        target_b1 = max(-beta_t_slope_init, 1e-3)
        raw_b1 = _inv_softplus(target_b1)
        self.raw_beta_t_slope_per_mode = nn.Parameter(torch.full((n_modes,), raw_b1))

        # Tier-specific commute-time threshold (Bhat 1995 / Hess 2007 heterogeneous VOT):
        #   V_lower += β_kink_k · max(0, t - T_k)  per income tier k
        # Captures "commute tolerance threshold": low-income commuters have a sharp
        # disutility acceleration at ~25 min, high-income tolerate up to ~55 min.
        # β_kink_k ≤ 0 (steeper disutility above threshold), T_k > 0 (in minutes).
        # Both per-tier learnable. Disabled by default (use_tier_threshold=False).
        self.use_tier_threshold = bool(use_tier_threshold)
        if self.use_tier_threshold:
            assert self.use_tier_mixture, "tier_threshold requires use_tier_mixture=True"
            # T_k init: low=25, mid=40, high=55 min — lit-anchored expectations
            T_init = [25.0, 40.0, 55.0][:self.n_income_tiers]
            if len(T_init) < self.n_income_tiers:
                T_init = T_init + [40.0] * (self.n_income_tiers - len(T_init))
            self.raw_T_threshold_per_tier = nn.Parameter(
                torch.tensor([_inv_softplus(t) for t in T_init])
            )
            # β_kink_k init: small negative (e.g. -0.05) — additional per-min penalty above T_k
            self.raw_beta_t_kink_per_tier = nn.Parameter(
                torch.full((self.n_income_tiers,), _inv_softplus(0.05))
            )
            # k_k (sharpness, minutes): how sharp the transition is around T_k.
            # k → 0 = step function (hard threshold); k = 5 = soft (transition over ~5 min).
            # Init k=5.0 min — gentle transition by default.
            self.raw_k_sharpness_per_tier = nn.Parameter(
                torch.full((self.n_income_tiers,), _inv_softplus(5.0))
            )
        else:
            self.raw_T_threshold_per_tier = None
            self.raw_beta_t_kink_per_tier = None
            self.raw_k_sharpness_per_tier = None

        # Lexicographic two-layer consideration filter (Swait 2001 / Cascetta 2001):
        # Layer 1 (occupation match): destination j passes if match(i,j) >= threshold.
        # Layer 2 (cost burden):      pass if monthly cost(i,j) / budget_tier <= ratio_tier.
        # Both are sigmoid soft masks; log(pass) is added to V_dest as log-mask.
        # Cost proxy: cost(i,j) = θ_t · t_min(i,j) + θ_d · log(distance(i,j))
        # All parameters learnable; ML-clean: filter strength learned from data.
        self.use_consideration_filter = bool(use_consideration_filter)
        if self.use_consideration_filter:
            assert self.use_tier_mixture, "consideration_filter requires use_tier_mixture=True"
            # Cost proxy coefficients (shared across tiers; cost is physical, tier in budget)
            self.raw_theta_t_cost = nn.Parameter(torch.tensor(_inv_softplus(0.05)))
            self.raw_theta_d_cost = nn.Parameter(torch.tensor(_inv_softplus(0.10)))
            # Monthly budget per tier (learnable, arbitrary scale — ratio matters)
            # Init: low<mid<high (low has tight budget)
            self.raw_cost_budget_per_tier = nn.Parameter(
                torch.tensor([_inv_softplus(v) for v in [3.0, 5.0, 9.0][:self.n_income_tiers]])
            )
            # Cost ratio threshold per tier (e.g., 10% / 12% / 18% — low tier strictest)
            self.raw_cost_thresh_per_tier = nn.Parameter(
                torch.tensor([_inv_softplus(v) for v in [0.10, 0.12, 0.18][:self.n_income_tiers]])
            )
            # Sharpness of cost filter sigmoid (smaller = sharper / steeper)
            self.raw_k_cost_filter = nn.Parameter(torch.tensor(_inv_softplus(3.0)))
            # Match filter (occupation match threshold + sharpness)
            # When match_thresh_floor > 0, actual threshold = floor + softplus(raw),
            # which forces the filter to actually cut destinations below the floor.
            self.match_thresh_floor = float(match_thresh_floor)
            # raw init so total ≈ floor + 0.05 (just above floor; lets model decide refinement)
            self.raw_match_filter_thresh = nn.Parameter(torch.tensor(_inv_softplus(0.05)))
            self.raw_k_match_filter = nn.Parameter(torch.tensor(_inv_softplus(float(k_match_init))))
        else:
            self.raw_theta_t_cost = None
            self.raw_theta_d_cost = None
            self.raw_cost_budget_per_tier = None
            self.raw_cost_thresh_per_tier = None
            self.raw_k_cost_filter = None
            self.raw_match_filter_thresh = None
            self.raw_k_match_filter = None

        # ASC per mode — free
        self.asc_per_mode = nn.Parameter(torch.zeros(n_modes))

        # θ_inc per mode — free (income × mode interaction)
        self.theta_inc_per_mode = nn.Parameter(torch.full((n_modes,), float(theta_inc_init)))

        # θ_kids per mode — free (household has dep children × mode interaction)
        self.theta_kids_per_mode = nn.Parameter(torch.zeros(n_modes))

        # θ_cars per mode — free (mean cars per household × mode interaction)
        self.theta_cars_per_mode = nn.Parameter(torch.zeros(n_modes))

        # λ_b per borough — bounded
        s = (lambda_init - lambda_eps_min) / (1.0 - lambda_eps_min)
        raw_l = _inv_sigmoid(s)
        self.raw_lambda_borough = nn.Parameter(torch.full((n_boroughs,), raw_l))

        # Wang blend (convex mode, legacy)
        if use_gnn_blend and self.gnn_mode == "convex":
            if self.blend_max <= 0.0:
                self.register_buffer("raw_gnn_blend", torch.tensor(-1e9))
                self._blend_is_buffer = True
            else:
                init = float(gnn_blend_init) / max(self.blend_max, 1e-6)
                init = min(max(init, 1e-4), 1 - 1e-4)
                raw_init = math.log(init / (1.0 - init))
                self.raw_gnn_blend = nn.Parameter(torch.tensor(raw_init))
                self._blend_is_buffer = False
        else:
            self.raw_gnn_blend = None
            self._blend_is_buffer = False

        # Wang TB-ResNet residual scale (additive mode):
        #   V_dest = V_RUM + w_NN · V_GNN_raw
        # w_NN ≥ 0 via softplus, learnable; small init (0.1) so NN starts as
        # minor correction and grows only if data needs it.
        if use_gnn_blend and self.gnn_mode == "residual":
            raw_w = _inv_softplus(max(float(gnn_residual_scale_init), 1e-4))
            self.raw_gnn_residual_scale = nn.Parameter(torch.tensor(raw_w))
        else:
            self.raw_gnn_residual_scale = None

        # Multiplicative gating (mult mode):
        #   V_dest = V_RUM · (1 + γ_mult · sigmoid(V_GNN_raw))
        # NN modulates RUM amplitude per (i, j, t) rather than adding residual.
        # γ_mult ≥ 0 via softplus (init 0.1 = up to 10 % amplitude swing).
        if use_gnn_blend and self.gnn_mode == "mult":
            raw_m = _inv_softplus(max(float(gnn_residual_scale_init), 1e-4))
            self.raw_gnn_mult_scale = nn.Parameter(torch.tensor(raw_m))
        else:
            self.raw_gnn_mult_scale = None

        # Per-origin Mixture-of-Experts gating (moe mode):
        #   g_i = sigmoid(raw_gate_per_origin[i]) ∈ [0, 1]
        #   V_dest[t, i, j] = (1 - g_i) · V_RUM + g_i · V_GNN
        # Each origin learns its own preference for RUM vs NN — directly interpretable
        # ("which origins are RUM-explainable vs NN-needed").
        # Init at raw=0 → g=0.5 (equal mix).
        if use_gnn_blend and self.gnn_mode == "moe":
            assert n_origins > 0, "moe mode requires n_origins > 0"
            self.raw_gate_per_origin = nn.Parameter(torch.zeros(n_origins))
        else:
            self.raw_gate_per_origin = None

        # Decision-tree-style sigmoid gate on Cervero match
        #   γ_effective(i, j) = γ + δ · sigmoid(k · (match_raw[i,j] - τ))
        # When k → 0: gate is flat 0.5, gamma_eff ≈ γ + 0.5·δ baseline.
        # When k large + match > τ: gate → 1, gamma_eff → γ + δ (boost).
        # When k large + match < τ: gate → 0, gamma_eff → γ (no boost / filtered).
        if self.use_match_gate:
            raw_k = _inv_softplus(max(float(gate_steepness_init), 1e-4))
            self.raw_gate_steepness = nn.Parameter(torch.tensor(raw_k))
            self.gate_threshold = nn.Parameter(torch.tensor(float(gate_threshold_init)))
        else:
            self.raw_gate_steepness = None
            self.gate_threshold = None

    def set_busy_dest_index(self, dest_to_k: torch.Tensor):
        """Called by trainer to register which grids are top-K busy destinations.
        dest_to_k: (N,) long tensor; entry = busy_idx in [0, K) or -1 if not busy.
        Re-allocates buffer to full N-size."""
        assert self.n_busy_dest > 0, "n_busy_dest=0; cannot set index"
        assert int((dest_to_k >= 0).sum()) == self.n_busy_dest, \
            f"dest_to_k has {int((dest_to_k>=0).sum())} busy entries; expected {self.n_busy_dest}"
        # Replace the (1,) placeholder buffer with the real (N,) one
        device = self.busy_dest_to_k_idx.device
        delattr(self, "busy_dest_to_k_idx")
        self.register_buffer("busy_dest_to_k_idx", dest_to_k.to(device).long())

    # =============================================================================
    # Transformed parameter properties
    # =============================================================================

    @property
    def alpha_wage(self) -> torch.Tensor:
        """α ≥ 0 (wage attracts, Cervero/Wang prior)."""
        return F.softplus(self.raw_alpha_wage)

    @property
    def gamma_M(self) -> torch.Tensor:
        """γ ≥ 0 (jobs attractiveness, must be non-negative behaviourally)."""
        return F.softplus(self.raw_gamma_M)

    @property
    def nu_D(self) -> torch.Tensor:
        """ν ≤ 0 (competition repels, must be non-positive behaviourally)."""
        return -F.softplus(self.raw_nu_D)

    @property
    def delta_match(self) -> torch.Tensor:
        """δ ≥ 0 — Cervero 1999 multiplicative match-gravity coefficient.

        Used in V_upper as: γ_effective(i, j) = γ + δ · match_prob_raw[i, j],
        i.e. high-match destinations get an amplified jobs-count attractor.
        This is Cervero's A_i = Σ E_j · match[i, j] / d^γ in log-utility form.
        Match modulates the gravity coefficient instead of being a separate
        additive term — the prior v3b spec ('+ δ·match_z') was lit-inconsistent.
        """
        return F.softplus(self.raw_delta_match)

    @property
    def beta_t_per_mode(self) -> torch.Tensor:
        """β_t,m (intercept) ≤ 0 (time disutility at log_d=0, per mode)."""
        return -F.softplus(self.raw_beta_t_per_mode)

    @property
    def T_threshold_per_tier(self) -> Optional[torch.Tensor]:
        """T_k > 0 (in minutes), per income tier — commute-tolerance threshold."""
        if self.raw_T_threshold_per_tier is None:
            return None
        return F.softplus(self.raw_T_threshold_per_tier)

    @property
    def beta_t_kink_per_tier(self) -> Optional[torch.Tensor]:
        """β_kink_k ≤ 0 (additional per-min disutility above T_k), per income tier."""
        if self.raw_beta_t_kink_per_tier is None:
            return None
        return -F.softplus(self.raw_beta_t_kink_per_tier)

    @property
    def k_sharpness_per_tier(self) -> Optional[torch.Tensor]:
        """k_k > 0 (minutes): sharpness of sigmoid transition around T_k.
        Small k → near-hard threshold; large k → gentle transition."""
        if self.raw_k_sharpness_per_tier is None:
            return None
        return F.softplus(self.raw_k_sharpness_per_tier).clamp(min=0.5)

    # ----- Lexicographic consideration filter properties -----
    @property
    def theta_t_cost(self) -> Optional[torch.Tensor]:
        if self.raw_theta_t_cost is None: return None
        return F.softplus(self.raw_theta_t_cost)
    @property
    def theta_d_cost(self) -> Optional[torch.Tensor]:
        if self.raw_theta_d_cost is None: return None
        return F.softplus(self.raw_theta_d_cost)
    @property
    def cost_budget_per_tier(self) -> Optional[torch.Tensor]:
        if self.raw_cost_budget_per_tier is None: return None
        return F.softplus(self.raw_cost_budget_per_tier)
    @property
    def cost_thresh_per_tier(self) -> Optional[torch.Tensor]:
        if self.raw_cost_thresh_per_tier is None: return None
        return F.softplus(self.raw_cost_thresh_per_tier)
    @property
    def k_cost_filter(self) -> Optional[torch.Tensor]:
        if self.raw_k_cost_filter is None: return None
        return F.softplus(self.raw_k_cost_filter).clamp(min=0.1)
    @property
    def match_filter_thresh(self) -> Optional[torch.Tensor]:
        if self.raw_match_filter_thresh is None: return None
        floor = getattr(self, "match_thresh_floor", 0.0)
        return floor + F.softplus(self.raw_match_filter_thresh)
    @property
    def k_match_filter(self) -> Optional[torch.Tensor]:
        if self.raw_k_match_filter is None: return None
        return F.softplus(self.raw_k_match_filter).clamp(min=0.1)

    @property
    def beta_t_slope_per_mode(self) -> torch.Tensor:
        """β_t,m,1 (slope on log_d) ≤ 0 — long-distance amplifies time disutility."""
        return -F.softplus(self.raw_beta_t_slope_per_mode)

    @property
    def lambda_per_borough(self) -> torch.Tensor:
        return self.lambda_eps_min + (1.0 - self.lambda_eps_min) * torch.sigmoid(
            self.raw_lambda_borough
        )

    @property
    def gnn_blend(self) -> Optional[torch.Tensor]:
        """Convex blend ∈ [0, blend_max] — legacy v3b/c mode only."""
        if self.raw_gnn_blend is None:
            return None
        return self.blend_max * torch.sigmoid(self.raw_gnn_blend)

    @property
    def gnn_mult_scale(self) -> Optional[torch.Tensor]:
        """γ_mult ≥ 0 (softplus) for multiplicative gating mode."""
        if self.raw_gnn_mult_scale is None:
            return None
        return F.softplus(self.raw_gnn_mult_scale)

    @property
    def gate_per_origin(self) -> Optional[torch.Tensor]:
        """g_i ∈ [0, 1] per origin for MoE mode (sigmoid of raw_gate_per_origin)."""
        if self.raw_gate_per_origin is None:
            return None
        return torch.sigmoid(self.raw_gate_per_origin)

    @property
    def gnn_residual_scale(self) -> Optional[torch.Tensor]:
        """w_NN ≥ 0 — Wang TB-ResNet additive residual scale (V_dest = V_RUM + w·V_NN)."""
        if self.raw_gnn_residual_scale is None:
            return None
        return F.softplus(self.raw_gnn_residual_scale)

    @property
    def gate_steepness(self) -> Optional[torch.Tensor]:
        """k ≥ 0 — sigmoid gate steepness on Cervero match.

        k → 0: gate is flat (no decision-tree gating, falls back to linear-ish).
        k large: gate becomes sharp threshold function — match below τ is
        filtered out (no γ boost), match above τ gets full δ boost.
        """
        if self.raw_gate_steepness is None:
            return None
        return F.softplus(self.raw_gate_steepness)

    def lambda_for_destination(self, grid_borough_idx: torch.Tensor) -> torch.Tensor:
        return self.lambda_per_borough[grid_borough_idx]

    # =============================================================================
    # Snapshot for json logging
    # =============================================================================

    def snapshot(self) -> dict:
        with torch.no_grad():
            # Handle tier mixture (tensor) vs single (scalar)
            def _to_py(t):
                return t.tolist() if t.dim() > 0 else float(t)
            return {
                "use_tier_mixture": self.use_tier_mixture,
                "n_income_tiers": self.n_income_tiers,
                "alpha_wage": _to_py(self.alpha_wage),
                "gamma_M": _to_py(self.gamma_M),
                "nu_D": _to_py(self.nu_D),
                "delta_match": _to_py(self.delta_match),
                "beta_t_per_mode": self.beta_t_per_mode.tolist(),
                "beta_t_slope_per_mode": self.beta_t_slope_per_mode.tolist(),
                "asc_per_mode": self.asc_per_mode.tolist(),
                "theta_inc_per_mode": self.theta_inc_per_mode.tolist(),
                "theta_kids_per_mode": self.theta_kids_per_mode.tolist(),
                "theta_cars_per_mode": self.theta_cars_per_mode.tolist(),
                "lambda_per_borough": self.lambda_per_borough.tolist(),
                "lambda_b_mean": float(self.lambda_per_borough.mean()),
                "lambda_b_std": float(self.lambda_per_borough.std()),
                "lambda_b_min": float(self.lambda_per_borough.min()),
                "lambda_b_max": float(self.lambda_per_borough.max()),
                "gnn_blend": float(self.gnn_blend) if self.gnn_blend is not None else None,
                "gnn_residual_scale": float(self.gnn_residual_scale) if self.gnn_residual_scale is not None else None,
                "gnn_mode": self.gnn_mode,
                "use_push_pull": self.use_push_pull,
                "xi_push_pull": float(self.xi_push_pull) if self.xi_push_pull is not None else None,
                "use_self_loop_boost": self.use_self_loop_boost,
                "self_loop_boost_mean_per_hour": (
                    self.self_loop_boost.mean(dim=0).tolist()
                    if self.self_loop_boost is not None else None
                ),
                "self_loop_boost_max_per_borough": (
                    self.self_loop_boost.max(dim=1).values.tolist()
                    if self.self_loop_boost is not None else None
                ),
                "n_busy_dest": self.n_busy_dest,
                "busy_dest_boost_mean_per_hour": (
                    self.busy_dest_boost.mean(dim=0).tolist()
                    if self.busy_dest_boost is not None else None
                ),
                "busy_dest_boost_per_dest_per_hour": (
                    self.busy_dest_boost.tolist()
                    if self.busy_dest_boost is not None else None
                ),
                "busy_dest_grid_idx": (
                    [int(i) for i in (self.busy_dest_to_k_idx >= 0).nonzero(as_tuple=True)[0].tolist()]
                    if self.busy_dest_to_k_idx is not None and self.busy_dest_to_k_idx.numel() > 1 else None
                ),
                "use_match_gate": self.use_match_gate,
                "gate_steepness": float(self.gate_steepness) if self.gate_steepness is not None else None,
                "gate_threshold": float(self.gate_threshold) if self.gate_threshold is not None else None,
            }
