"""Inverse-RUM joint estimator (theta, alpha, beta_t, beta_c, gamma).

Maximises the log-likelihood of observed OD flows F_ij^t under
    U_ij = alpha * V_jt + beta_t * t_ij + beta_c * c_ij + gamma * log d_ij
with V_jt produced by a structural utility net (default StructuralGNN, see
``structural_gnn.StructuralGNN``).  Sign reparametrisations:
    alpha    = 1.0 (fixed buffer; see _RUMHead docstring for justification)
    gamma    = -softplus(raw)            (negative; distance disutility)
    beta_t   = -softplus(raw_beta_t)     (negative; 1 per hour)
    beta_c   = -softplus(raw_beta_c)     (negative; cost numeraire)

Identifiability tricks (Paper A):
  * top-K=50 sampling-of-alternatives w/ McFadden correction (topk_choice.py).
    Sampling is with-replacement multinomial under softmax(V_jt) so log_q
    is exact (see ``topk_choice`` docstring for the B4 fix).
  * A ridge-style regularisation on V_jt's projection onto the column means
    of (t_ij, log d_ij). Originally labelled "FWL residualisation" but, as
    flagged in R1, this is NOT the OLS Frisch-Waugh-Lovell transformation;
    it is a column-mean partial-out used as a soft penalty on collinear
    GNN content. See ``_fwl_inspired_ridge_penalty`` below.
  * separate AdamW lr for theta (slow) vs (alpha, beta, gamma) (fast).

The implicit-softmax autograd.Function is used in lieu of plain
``log_softmax`` so the same code path supports nested-logit ablations.

Canonical API (matches experiments/paper_a/* call sites):
    trainer = InverseRUMTrainer(
        grid_features=X_static,        # (N, F)
        edge_index=edge_index,         # (2, E)
        observed_OD=F_ij_t,            # (T, N, N) or (N, N)
        t_ij_t=t_ij_t,                 # (T, N, N) or (N, N)
        log_d_ij=log_d_ij,             # (N, N)
        K=50,
        utility_net=None,              # if None build StructuralGNN internally
        choice_model="flat",           # "flat" or "nested"
        train_mask=...,                # (N,) bool — train origins
        val_mask=...,                  # (N,) bool — held-out origins
        device="cpu",
        seed=42,
    )
    theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log = trainer.fit()
    pred_F = trainer.predict_OD(t_ij_override=None)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .implicit_softmax import implicit_softmax
from .structural_gnn import StructuralGNN
from .topk_choice import make_choice_set


def _inv_softplus(v: float) -> float:
    return math.log(math.exp(v) - 1.0)


@dataclass
class TrainingLog:
    epoch: List[int] = field(default_factory=list)
    train_nll: List[float] = field(default_factory=list)
    val_nll: List[float] = field(default_factory=list)
    cpc_val: List[float] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)
    beta_mean: List[float] = field(default_factory=list)
    beta_c: List[float] = field(default_factory=list)
    gamma: List[float] = field(default_factory=list)
    topk_calls: int = 0


class _RUMHead(nn.Module):
    """Trainable (beta_t, beta_c, gamma) with sign constraints; alpha fixed at 1.0.

    alpha         : fixed buffer = 1.0 (NOT trainable). When V_jt = GNN_theta(X)
                    is itself a learnable function with arbitrary output scale,
                    the product alpha * V_jt is identified only up to a positive
                    scaling. We resolve by fixing alpha=1.0; the GNN absorbs all
                    scale variation. This is standard in deep discrete-choice
                    (Wang & Klabjan 2018) and aligns with conditional logit
                    identification (Train 2009 sec 3).
    gamma         : negative via -softplus (distance disutility coefficient is
                    structurally negative; without this constraint gamma can be
                    fit with the wrong sign by absorbing variance from V_jt).
    beta_t        : negative via -softplus (one parameter per hour; can be
                    averaged at report time)
    beta_c        : negative via -softplus (B6: cost numeraire, scalar)
    """

    def __init__(self, n_hours: int, n_income_tiers: int = 1,
                 alpha_init: float = 1.0,
                 beta_init_neg: float = -0.07,
                 beta_c_init_neg: float = -1.0,
                 gamma_init_neg: float = -0.5,
                 delta_init: float = 0.0,
                 enable_occ_match: bool = False):
        super().__init__()
        # alpha is a fixed buffer (NOT a trainable parameter). See docstring.
        # alpha_init is accepted for API parity but ignored — alpha is always 1.0.
        self.register_buffer("alpha", torch.tensor(1.0))
        # gamma = -softplus(raw_gamma); init so softplus(raw)=|gamma_init_neg|.
        target_gamma = -gamma_init_neg
        raw_gamma_init = _inv_softplus(target_gamma)
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma_init))
        # Phase B (2026-05-05): raw_beta_t shape (n_hours, n_income_tiers).
        # n_income_tiers=1 (default) is backward-compatible with paper A
        # minimal — beta_t reduces to a per-hour scalar.
        # n_income_tiers>1 enables mixture-logit RUM where each income tier
        # has its own β_t,k. Trainer multiplies by P(tier=k | origin i)
        # mixture weights at forward time.
        target = -beta_init_neg
        raw_init = _inv_softplus(target)
        self.n_income_tiers = n_income_tiers
        self.raw_beta_t = nn.Parameter(torch.full((n_hours, n_income_tiers), raw_init))
        # B6: beta_c = -softplus(raw_beta_c). Default init -1.0 (cost numeraire).
        target_c = -beta_c_init_neg
        raw_c_init = _inv_softplus(target_c)
        self.raw_beta_c = nn.Parameter(torch.tensor(raw_c_init))
        # Phase B (2026-05-05): δ for OccMatch attractiveness term.
        # δ is unconstrained (can be positive or negative; positive expected).
        # Disabled by default for backward compatibility — set enable_occ_match=True
        # in trainer.__init__ to opt in.
        self.enable_occ_match = enable_occ_match
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))
        # Phase B-2c (2026-05-05): φ for income-time INTERACTION term.
        # Replaces mixture-logit identification (which fails on aggregate OD;
        # see Train 2009 ch.6 §6.2). β_eff(i) = β_t · (1 + φ · z_income(i)),
        # where z_income(i) is observed origin-level income covariate (e.g.
        # IMD-derived). φ identifiable from aggregate OD because each origin's
        # income score is observed (not latent).
        # φ > 0: time-sensitivity scales up with income (WebTAG-direction)
        # φ < 0: time-sensitivity scales down with income (reverse)
        # Default 0 → no interaction (paper A minimal behavior).
        self.raw_phi = nn.Parameter(torch.tensor(0.0))
        # Phase B-2d (2026-05-05): ψ for destination-wage time INTERACTION.
        # β_eff(i, j) = β_t · (1 + φ · z_origin(i) + ψ · z_log_wage(j)).
        # Captures "commuters choosing high-wage destinations reveal higher
        # time-sensitivity (the WebTAG-direction)". ψ > 0 → high-wage j → larger
        # |β| → higher VOT, matching mainstream RP literature.
        self.raw_psi = nn.Parameter(torch.tensor(0.0))
        # Phase B-2e (2026-05-05): ξ for explicit wage attraction.
        # U += ξ · z_log_wage(j) (additive utility term, not β interaction).
        # Disentangles wage attraction from V_j(GNN) so β reflects time
        # disutility CONTROLLED for wage attraction. ξ > 0 expected.
        self.raw_xi = nn.Parameter(torch.tensor(0.0))

    @property
    def phi(self) -> torch.Tensor:
        """Income-time interaction coefficient (origin-side).
        If `_phi_signed_negative=True`, enforces φ ≤ 0 (mainstream-direction).
        """
        if getattr(self, "_phi_signed_negative", False):
            return -F.softplus(self.raw_phi)
        return self.raw_phi

    @property
    def psi(self) -> torch.Tensor:
        """Wage-time interaction coefficient (destination-side).
        If `_psi_signed_positive=True`, enforces ψ ≥ 0 (mainstream-direction).
        """
        if getattr(self, "_psi_signed_positive", False):
            return F.softplus(self.raw_psi)
        return self.raw_psi

    @property
    def xi(self) -> torch.Tensor:
        """Wage attraction coefficient (additive utility term)."""
        return self.raw_xi

    @property
    def gamma(self) -> torch.Tensor:
        return -F.softplus(self.raw_gamma)

    @property
    def beta_t(self) -> torch.Tensor:
        return -F.softplus(self.raw_beta_t)

    @property
    def beta_c(self) -> torch.Tensor:
        return -F.softplus(self.raw_beta_c)


def _fwl_inspired_ridge_penalty(
    V_jt: torch.Tensor,
    t_ij_t: torch.Tensor,
    log_d_ij: torch.Tensor,
) -> torch.Tensor:
    """Column-mean partial-out of V_jt against (mean t_ij, log d_ij).

    NOTE (R1): this is NOT the OLS Frisch-Waugh-Lovell transformation.
    It is a ridge-style regularisation on V_jt's projection onto the
    column-mean travel-time and log-distance directions; the residual
    we return is what the rest of the pipeline consumes in place of V.
    Full per-(i,j) FWL would require regressing every V column against
    its (i,j)-specific (t, log d) which loses the (T, N) structure.

    Inputs
    ------
    V_jt     : (T, N)
    t_ij_t   : (T, N, N) — averaged over origins to get per-(t, j) cost
    log_d_ij : (N, N)    — averaged over origins to get per-j distance

    Returns
    -------
    V_resid : (T, N) — residual after projecting V onto the (1, t_col, d_col)
              basis per hour. Algebraically a closed-form OLS in 3 columns.
    """
    T, N = V_jt.shape
    t_col_mean = t_ij_t.mean(dim=1)             # (T, N)
    d_col_mean = log_d_ij.mean(dim=0)           # (N,)
    d_col_mean = d_col_mean.unsqueeze(0).expand(T, -1)
    V_resid = torch.empty_like(V_jt)
    ones_N = torch.ones(N, device=V_jt.device, dtype=V_jt.dtype)
    for t in range(T):
        X = torch.stack([ones_N, t_col_mean[t], d_col_mean[t]], dim=1)   # (N, 3)
        y = V_jt[t]
        XtX = X.T @ X
        Xty = X.T @ y
        try:
            sol = torch.linalg.solve(XtX, Xty)
        except RuntimeError:
            sol = torch.linalg.lstsq(X, y).solution
        V_resid[t] = y - X @ sol
    return V_resid


class InverseRUMTrainer:
    """Joint inverse-RUM estimator with the canonical Paper-A API.

    See module docstring for usage. fit() returns:
        (theta_hat, alpha_hat, beta_t_hat, beta_c_hat, gamma_hat, log)
    where theta_hat is the trained utility net (StructuralGNN by default).
    """

    def __init__(
        self,
        grid_features: torch.Tensor,
        edge_index: torch.Tensor,
        observed_OD: torch.Tensor,
        t_ij_t: torch.Tensor,
        log_d_ij: torch.Tensor,
        K: int = 50,
        utility_net: Optional[nn.Module] = None,
        choice_model: str = "flat",
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
        seed: int = 42,
        lr_theta: float = 1e-3,
        lr_rum: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        patience: int = 8,
        residualise: bool = False,
        verbose: bool = False,
        use_topk_loss: bool = False,
        occ_match: Optional[torch.Tensor] = None,
        income_tier_props: Optional[torch.Tensor] = None,
        income_score_per_origin: Optional[torch.Tensor] = None,
        wage_score_per_dest: Optional[torch.Tensor] = None,
        enable_wage_attraction: bool = False,
        enforce_mainstream_direction: bool = False,
    ):
        # ---- KNOWN ISSUE: FWL ridge introduces systematic bias ---------------
        # _fwl_inspired_ridge_penalty subtracts V_jt's projection onto
        # (1, t_col_mean, log_d_col_mean). The teacher's data-generating V
        # has nonzero loading on these directions, so a residualised student
        # V cannot match the teacher V; the residual is absorbed into β/γ
        # estimates, producing a 1-3% systematic bias on β_t even at large N.
        # Default residualise=False (changed 2026-05-05). Pass residualise=True
        # only if you understand the implied bias and need it for some reason.
        # ---- KNOWN ISSUE: top-K loss is broken --------------------------------
        # _topk_log_likelihood samples K destinations from softmax(V_jt) without
        # forcing the observed chosen alternative into the choice set
        # (see _build_choice_sets: chosen_idx=None, include_chosen=False).
        # As V trains and softmax(V) peaks, sampled destinations stop overlapping
        # with positive-flow destinations -> F_at_idx -> 0 -> loss collapses to
        # a near-constant -> grad -> 0. Verified by W4 root-cause diagnostic
        # (see evaluation_outputs/paper_a/diag_trainer*.log, 2026-05-05).
        #
        # Until top-K is reimplemented per McFadden 1978 (chosen-inclusion +
        # per-(i,j) choice sets), default to full-softmax loss. Pass
        # use_topk_loss=True explicitly to opt back into the broken path
        # (only meaningful for reproducing the bug, not for production runs).
        self.use_topk_loss = use_topk_loss
        if use_topk_loss:
            import warnings
            warnings.warn(
                "InverseRUMTrainer: use_topk_loss=True is BROKEN (chosen not "
                "forced into choice set; see docstring). Loss will plateau "
                "and gradient will vanish after ~5 epochs. Use full-softmax "
                "loss (use_topk_loss=False) for valid recovery.",
                RuntimeWarning, stacklevel=2,
            )
        # --- normalise tensor shapes ---------------------------------------
        # grid_features may be (N, F) static or (T, N, F) hour-stacked.
        if grid_features.dim() == 2:
            self._X_static = True
            N, Fdim = grid_features.shape
        elif grid_features.dim() == 3:
            self._X_static = False
            T_x, N, Fdim = grid_features.shape
        else:
            raise ValueError(
                f"grid_features must be (N,F) or (T,N,F); got {tuple(grid_features.shape)}"
            )

        # observed_OD may be (N,N) or (T,N,N).
        if observed_OD.dim() == 2:
            T = 1
            observed_OD = observed_OD.unsqueeze(0)
        elif observed_OD.dim() == 3:
            T = observed_OD.shape[0]
        else:
            raise ValueError(
                f"observed_OD must be (N,N) or (T,N,N); got {tuple(observed_OD.shape)}"
            )
        if observed_OD.shape[1] != N or observed_OD.shape[2] != N:
            raise ValueError(
                f"observed_OD trailing shape {tuple(observed_OD.shape[1:])} != ({N},{N})"
            )

        # t_ij_t may be (N,N) or (T,N,N). Broadcast static cost across hours.
        if t_ij_t.dim() == 2:
            t_ij_t = t_ij_t.unsqueeze(0).expand(T, N, N)
        elif t_ij_t.dim() == 3:
            if t_ij_t.shape[0] != T:
                # rare: caller passed full (T_full, N, N) cost but aggregated F. Use mean.
                t_ij_t = t_ij_t.mean(0, keepdim=True).expand(T, N, N)
        else:
            raise ValueError(f"t_ij_t must be (N,N) or (T,N,N); got {tuple(t_ij_t.shape)}")

        if log_d_ij.dim() != 2 or log_d_ij.shape != (N, N):
            raise ValueError(f"log_d_ij must be (N,N); got {tuple(log_d_ij.shape)}")

        if grid_features.dim() == 2:
            X = grid_features.unsqueeze(0).expand(T, N, Fdim).contiguous()
        else:
            if grid_features.shape[0] != T:
                X = grid_features.mean(0, keepdim=True).expand(T, N, Fdim).contiguous()
            else:
                X = grid_features.contiguous()

        # --- masks ---------------------------------------------------------
        if val_mask is None:
            val_mask = torch.zeros(N, dtype=torch.bool)
            val_mask[::5] = True  # every 5th origin held out by default
        if train_mask is None:
            train_mask = ~val_mask
        if train_mask.dtype != torch.bool:
            train_mask = train_mask.bool()
        if val_mask.dtype != torch.bool:
            val_mask = val_mask.bool()
        # Mutual-exclusion guard (R1): both given but overlap is ambiguous.
        overlap = (train_mask & val_mask).sum().item()
        assert overlap == 0, (
            f"train_mask and val_mask overlap on {overlap} origins; "
            f"defaults guarantee mutual exclusion."
        )

        # --- store ---------------------------------------------------------
        self.N = N
        self.T = T
        self.Fdim = Fdim
        self.device = device
        self.seed = seed
        self.K = K
        self.choice_model = choice_model
        self.epochs = epochs
        self.patience = patience
        self.residualise = residualise
        self.verbose = verbose
        self.lr_theta = lr_theta
        self.lr_rum = lr_rum
        self.weight_decay = weight_decay

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.X = X.to(device)
        self.edge_index = edge_index.to(device)
        self.observed_OD = observed_OD.to(device).float()
        self.t_ij_t = t_ij_t.to(device).float()
        self.log_d_ij = log_d_ij.to(device).float()
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        # Phase B: OccMatch matrix (N, N) — see paperA_v23_aux.npz.
        # If None, OccMatch term is disabled (paper A baseline behavior).
        if occ_match is not None:
            occ_t = occ_match if isinstance(occ_match, torch.Tensor) else torch.tensor(occ_match)
            if occ_t.shape != (N, N):
                raise ValueError(
                    f"occ_match must be (N, N) = ({N}, {N}); got {tuple(occ_t.shape)}"
                )
            self.occ_match = occ_t.to(device).float()
            self._enable_occ_match = True
        else:
            self.occ_match = None
            self._enable_occ_match = False

        # Phase B (B-2b): income_tier_props (N_origins, n_tiers) — origin-grid
        # mixture weights P(income tier = k | origin i). If None, n_tiers=1 and
        # the mixture-logit path collapses to paper A baseline (single β per
        # hour). If provided, RUM head learns tier-specific β_t,k and forward
        # uses Σ_k π_k|i · softmax(U_ij^k).
        # NOTE 2026-05-05: mixture path is empirically not identifiable on
        # aggregate OD (Train 2009 §6.2). Prefer the income_score_per_origin
        # interaction path (Phase B-2c) for paper A real-data experiments.
        if income_tier_props is not None:
            itp = (income_tier_props if isinstance(income_tier_props, torch.Tensor)
                   else torch.tensor(income_tier_props))
            if itp.dim() != 2 or itp.shape[0] != N:
                raise ValueError(
                    f"income_tier_props must be (N, n_tiers) with N={N}; "
                    f"got {tuple(itp.shape)}"
                )
            self.income_tier_props = itp.to(device).float()
            self.n_income_tiers = int(itp.shape[1])
            self._enable_mixture = self.n_income_tiers > 1
        else:
            self.income_tier_props = None
            self.n_income_tiers = 1
            self._enable_mixture = False

        # Phase B-2c: income_score_per_origin (N,) — observed origin-level
        # income covariate (z-scored). β_eff(i) = β_t · (1 + φ · score(i)).
        # Identifiable from aggregate OD because score(i) is observed.
        if income_score_per_origin is not None:
            isc = (income_score_per_origin if isinstance(income_score_per_origin, torch.Tensor)
                   else torch.tensor(income_score_per_origin))
            if isc.shape != (N,):
                raise ValueError(
                    f"income_score_per_origin must be (N,) with N={N}; "
                    f"got {tuple(isc.shape)}"
                )
            # z-score normalize so φ is interpretable as "per-stddev shift"
            isc = isc.float()
            isc = (isc - isc.mean()) / (isc.std() + 1e-8)
            self.income_score_per_origin = isc.to(device)
            self._enable_income_interaction = True
        else:
            self.income_score_per_origin = None
            self._enable_income_interaction = False

        # Phase B-2d: wage_score_per_dest (N,) — observed destination wage
        # covariate (z-scored log wage). β_eff(i, j) augmented with ψ · z_w(j).
        # Captures "commuters to high-wage destinations reveal higher time
        # sensitivity" — interpretable as occupational selection effect (rich
        # workers commute to high-wage CBD jobs and value time more).
        if wage_score_per_dest is not None:
            wsc = (wage_score_per_dest if isinstance(wage_score_per_dest, torch.Tensor)
                   else torch.tensor(wage_score_per_dest))
            if wsc.shape != (N,):
                raise ValueError(
                    f"wage_score_per_dest must be (N,) with N={N}; "
                    f"got {tuple(wsc.shape)}"
                )
            wsc = wsc.float()
            wsc = (wsc - wsc.mean()) / (wsc.std() + 1e-8)
            self.wage_score_per_dest = wsc.to(device)
            self._enable_dest_wage_interaction = True
        else:
            self.wage_score_per_dest = None
            self._enable_dest_wage_interaction = False

        # Phase B-2e: explicit wage attraction. Reuses wage_score_per_dest
        # tensor as additive utility term (not β interaction).
        self._enable_wage_attraction = bool(enable_wage_attraction) and self.wage_score_per_dest is not None

        # --- utility net ---------------------------------------------------
        if utility_net is None:
            utility_net = StructuralGNN(node_dim=Fdim, hidden_dim=32,
                                        n_layers=2, gru_hidden=16)
        self.gnn = utility_net.to(device)  # name kept for back-compat with experiments
        self.rum = _RUMHead(n_hours=T,
                            n_income_tiers=self.n_income_tiers,
                            enable_occ_match=self._enable_occ_match).to(device)
        # Variant 6 (2026-05-05): identification check — constrain φ ≤ 0 and
        # ψ ≥ 0 (mainstream-direction). If with this constraint CPC ≈ unconstrained
        # CPC, both directions are statistically equivalent local optima.
        if enforce_mainstream_direction:
            self.rum._phi_signed_negative = True
            self.rum._psi_signed_positive = True

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.gnn.parameters(), "lr": lr_theta,
                 "weight_decay": weight_decay},
                {"params": self.rum.parameters(), "lr": lr_rum,
                 "weight_decay": 0.0},
            ]
        )

        # rng for top-K sampling. Logged so tests can probe topk_calls.
        self._rng = np.random.default_rng(seed)
        self._topk_calls = 0

    # ------------------------------------------------------------------ utility-net forward
    def _forward_V(self, X: torch.Tensor, norm_stats=None) -> torch.Tensor:
        """Run the utility net to obtain V_jt of shape (T, N).

        StructuralGNN expects (T, N, F) and returns (T, N).
        Linear / MLP utility nets used in gnn_ablation.py expect (N, F)
        and return (N, 1). We adapt by looping over the T axis when the
        net rejects 3D input.

        ``norm_stats`` (optional (mean, std)) is forwarded to StructuralGNN
        so counterfactual forwards can freeze the V scaling at baseline
        (see Scenario A/B harnesses). Ignored for non-GNN utility nets.
        """
        try:
            if norm_stats is not None:
                V = self.gnn(X, self.edge_index, norm_stats=norm_stats)
            else:
                V = self.gnn(X, self.edge_index)
        except (TypeError, RuntimeError, AttributeError):
            # Non-GNN utility net: assume callable with single static input.
            V_list = []
            for t in range(X.shape[0]):
                v_t = self.gnn(X[t])
                if v_t.dim() == 2 and v_t.shape[-1] == 1:
                    v_t = v_t.squeeze(-1)
                V_list.append(v_t)
            V = torch.stack(V_list, dim=0)
        # squeeze trailing singleton if present
        if V.dim() == 3 and V.shape[-1] == 1:
            V = V.squeeze(-1)
        return V

    # ------------------------------------------------------------------ likelihood
    def _build_choice_sets(
        self,
        V_jt: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build top-K choice sets per (t, origin) via TopKChoiceSet.sample.

        Short-circuits to None when N <= K (full-N softmax fallback for
        tiny test cases).

        Returns dict with:
          idx   : (T, N, K) sampled destination indices
          log_q : (T, N, K) sampling log-probabilities
        """
        T, N = V_jt.shape
        K = self.K
        if N <= K:
            # Full-N fallback: equivalent to softmax over all destinations.
            return None
        idx_all = torch.empty(T, N, K, dtype=torch.long, device=V_jt.device)
        log_q_all = torch.empty(T, N, K, dtype=V_jt.dtype, device=V_jt.device)
        for t in range(T):
            V_t = V_jt[t].detach()  # sampling is non-differentiable
            for i in range(N):
                idx, log_q = make_choice_set(
                    origin_idx=i,
                    V_jt=V_t,
                    K=K,
                    rng=self._rng,
                    chosen_idx=None,           # we sample without forcing
                    include_chosen=False,
                )
                idx_all[t, i] = idx
                log_q_all[t, i] = log_q
                self._topk_calls += 1
        return {"idx": idx_all, "log_q": log_q_all}

    def _per_hour_log_p_full(
        self,
        V_jt: torch.Tensor,
        t_ij_t: torch.Tensor,
        log_d_ij: torch.Tensor,
    ) -> torch.Tensor:
        """log P(j | i, t) for all (t, i, j) — full N×N softmax.

        Phase B (2026-05-05): supports income-tier mixture logit when
        ``self._enable_mixture`` is True (n_income_tiers > 1):

            P(j | i, t) = Σ_k π_{k|i} · softmax_j( α V_jt + β_{t,k} t_ij + γ log_d + δ occ )

        Reduces to paper A minimal when n_income_tiers = 1.
        """
        T, N = V_jt.shape
        alpha, gamma = self.rum.alpha, self.rum.gamma
        beta_t = self.rum.beta_t                                  # (T, n_tiers)
        K = self.n_income_tiers
        beta_c = self.rum.beta_c
        _ = beta_c  # referenced for clarity; not in logits — see B6 note.

        # Broadcast all terms to shape (T, K, N, N) — destination axis last.
        V_b = V_jt.view(T, 1, 1, N).expand(T, K, N, N)            # repeat along K, i
        t_b = t_ij_t.view(T, 1, N, N).expand(T, K, N, N)          # repeat along K
        beta_b = beta_t.view(T, K, 1, 1).expand(T, K, N, N)
        d_b = log_d_ij.view(1, 1, N, N).expand(T, K, N, N)
        # Phase B-2c/2d: income-time and wage-time interactions.
        #   β_eff(i, j) = β_t · (1 + φ·z_origin_income(i) + ψ·z_dest_wage(j))
        # Both covariates observed → identifiable from aggregate OD.
        scale_per_origin = None
        if self._enable_income_interaction and self.income_score_per_origin is not None:
            phi = self.rum.phi
            score_o = self.income_score_per_origin                # (N,)
            scale_per_origin = phi * score_o.view(1, 1, N, 1)     # (1,1,N,1)
        scale_per_dest = None
        if self._enable_dest_wage_interaction and self.wage_score_per_dest is not None:
            psi = self.rum.psi
            score_d = self.wage_score_per_dest                    # (N,)
            scale_per_dest = psi * score_d.view(1, 1, 1, N)       # (1,1,1,N)
        if scale_per_origin is not None or scale_per_dest is not None:
            scale = 1.0
            if scale_per_origin is not None:
                scale = scale + scale_per_origin
            if scale_per_dest is not None:
                scale = scale + scale_per_dest
            beta_b = beta_b * scale                                # (T, K, N, N)
        logits = alpha * V_b + beta_b * t_b + gamma * d_b         # (T, K, N, N)

        # Phase B-2a: add δ · OccMatch[i, j] term if enabled.
        if self._enable_occ_match and self.occ_match is not None:
            delta = self.rum.delta
            occ_b = self.occ_match.view(1, 1, N, N).expand(T, K, N, N)
            logits = logits + delta * occ_b

        # Phase B-2e: add explicit ξ · z_log_wage(j) wage attraction.
        # Disentangles wage attraction from V_j(GNN) so β reflects time
        # disutility controlled for wage attraction.
        if self._enable_wage_attraction and self.wage_score_per_dest is not None:
            xi = self.rum.xi
            wage_term = xi * self.wage_score_per_dest.view(1, 1, 1, N)  # (1,1,1,N)
            logits = logits + wage_term.expand(T, K, N, N)

        # log_softmax over destinations j (last axis), per (t, k, i)
        log_p_per_tier = torch.log_softmax(logits, dim=-1)        # (T, K, N, N)

        if self._enable_mixture and self.income_tier_props is not None:
            # Mixture: log P(j|i, t) = logsumexp_k( log π_{k|i} + log_p_per_tier[t, k, i, j] )
            log_pi = torch.log(self.income_tier_props.clamp(min=1e-30))   # (N, K)
            log_pi_b = log_pi.t().view(1, K, N, 1)                          # (1, K, N, 1)
            log_p_combined = log_p_per_tier + log_pi_b                       # (T, K, N, N)
            log_p = torch.logsumexp(log_p_combined, dim=1)                   # (T, N, N)
        else:
            # n_tiers = 1: squeeze the tier axis. Mathematically identical to
            # paper A baseline.
            log_p = log_p_per_tier.squeeze(1)                                # (T, N, N)

        return log_p

    def _topk_log_likelihood(
        self,
        V_jt: torch.Tensor,
        t_ij_t: torch.Tensor,
        log_d_ij: torch.Tensor,
        F_ij_t: torch.Tensor,
        cs: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sampled-alternatives multinomial NLL with McFadden correction.

        For each (t, i), evaluate log P(j | C_i) over the sampled K
        destinations, apply the importance correction, and weight by the
        observed flow F_ij_t. Slots are with-replacement IID samples so
        log_q is exact.

        Returns
        -------
        nll : scalar — negative log-likelihood averaged over masked origins.
        """
        T, N = V_jt.shape
        K = self.K
        idx = cs["idx"]                       # (T, N, K)
        log_q = cs["log_q"]                   # (T, N, K)
        alpha = self.rum.alpha
        gamma = self.rum.gamma
        beta_t = self.rum.beta_t              # (T,)
        beta_c = self.rum.beta_c              # scalar

        # Gather V at sampled indices: V_jt[t, idx[t, i, k]]
        # (T, N, K) <- index V_jt: shape (T, N) along dim=1
        V_at_idx = torch.gather(
            V_jt.unsqueeze(1).expand(T, N, N),  # (T, N, N) — V is destination-indexed; replicate over i
            dim=2,
            index=idx,
        )                                      # (T, N, K)

        # Gather t_ij at sampled indices.
        t_at_idx = torch.gather(t_ij_t, dim=2, index=idx)        # (T, N, K)
        d_at_idx = torch.gather(
            log_d_ij.unsqueeze(0).expand(T, N, N),
            dim=2, index=idx,
        )                                                          # (T, N, K)

        beta_b = beta_t.view(T, 1, 1)
        # See _per_hour_log_p_full: beta_c is the cost numeraire, not a
        # logit coefficient on time. Reference it so the optimizer keeps
        # the parameter alive (regularised at init by AdamW weight decay
        # of 0 on RUM head).
        _ = beta_c
        logits = (alpha * V_at_idx
                  + beta_b * t_at_idx
                  + gamma * d_at_idx)                              # (T, N, K)

        # McFadden correction: subtract log_q before softmax over set.
        log_p_set = torch.log_softmax(logits - log_q, dim=-1)      # (T, N, K)

        # Observed flow gathered at sampled indices.
        F_at_idx = torch.gather(F_ij_t, dim=2, index=idx)          # (T, N, K)

        # Multinomial NLL: -sum_{t,i,k in C_i} F[t,i,j(k)] * log_p_set[t,i,k]
        # masked over origins.
        mask_f = mask.view(1, N, 1).to(log_p_set.dtype)
        weighted = -(F_at_idx * log_p_set * mask_f)
        # Denominator: number of (t, i) with positive observed flow inside C_i.
        active = (F_at_idx.sum(dim=2) > 0) & (mask.view(1, N).bool())
        denom = max(int(active.sum().item()), 1)
        return weighted.sum() / denom

    def _nll(
        self,
        log_p: torch.Tensor,
        F_ij_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multinomial NLL on full (T,N,N) log-prob tensor (fallback path)."""
        T, N, _ = F_ij_t.shape
        if mask is None:
            flow = F_ij_t
            denom = max(int((F_ij_t.sum(dim=2) > 0).sum()), 1)
        else:
            mask_f = mask.view(1, N, 1).float()
            flow = F_ij_t * mask_f
            denom = max(int((flow.sum(dim=2) > 0).sum()), 1)
        nll = -(flow * log_p).sum() / denom
        return nll

    def _cpc(
        self,
        log_p: torch.Tensor,
        F_ij_t: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Standard CPC on full (T, N, N) log-prob tensor.

        CPC = 2 * sum(min(pred, F)) / (sum(pred) + sum(F))
        (Lenormand et al. 2012). The previous Sørensen-style numerator-only
        variant only matched in the special case sum(pred) == sum(F);
        switching to the standard form per R1.
        """
        with torch.no_grad():
            P = log_p.exp()
            row_sum = F_ij_t.sum(dim=2, keepdim=True)
            pred = P * row_sum
            num = 2.0 * torch.minimum(pred[:, mask], F_ij_t[:, mask]).sum()
            den = (pred[:, mask].sum() + F_ij_t[:, mask].sum()).clamp(min=1.0)
            return float(num / den)

    # ------------------------------------------------------------------ fit
    def fit(self) -> Tuple[nn.Module, float, torch.Tensor, float, float, TrainingLog]:
        """Run the inverse loop using the constructor-supplied data.

        Returns
        -------
        theta_hat   : trained utility net (the GNN object itself)
        alpha_hat   : float
        beta_t_hat  : (T,) tensor (per-hour) — already negative
        beta_c_hat  : float — already negative
        gamma_hat   : float
        log         : TrainingLog
        """
        log = TrainingLog()
        best_val = float("inf")
        no_improve = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for ep in range(self.epochs):
            self.gnn.train(); self.rum.train()
            self.optimizer.zero_grad()
            V_jt = self._forward_V(self.X)                          # (T, N)
            if self.residualise:
                V_jt = _fwl_inspired_ridge_penalty(V_jt, self.t_ij_t, self.log_d_ij)

            # Top-K loss path is gated behind use_topk_loss flag because it is
            # currently broken (see __init__ docstring). Default path is
            # full-softmax NLL, which is also faster than top-K at N <= 5000
            # because top-K's _build_choice_sets uses a Python double loop.
            use_topk = self.use_topk_loss and self.K < self.N
            cs = self._build_choice_sets(V_jt) if use_topk else None
            if cs is not None:
                loss = self._topk_log_likelihood(
                    V_jt, self.t_ij_t, self.log_d_ij, self.observed_OD, cs,
                    mask=self.train_mask,
                )
            else:
                log_p = self._per_hour_log_p_full(V_jt, self.t_ij_t, self.log_d_ij)
                loss = self._nll(log_p, self.observed_OD, mask=self.train_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn.parameters()) + list(self.rum.parameters()), 1.0
            )
            self.optimizer.step()

            # eval
            self.gnn.eval(); self.rum.eval()
            with torch.no_grad():
                V_jt_e = self._forward_V(self.X)
                if self.residualise:
                    V_jt_e = _fwl_inspired_ridge_penalty(V_jt_e, self.t_ij_t, self.log_d_ij)
                # eval always uses full-softmax for stable metrics
                log_p_e = self._per_hour_log_p_full(V_jt_e, self.t_ij_t, self.log_d_ij)
                val_nll = self._nll(log_p_e, self.observed_OD, mask=self.val_mask).item()
                cpc_val = self._cpc(log_p_e, self.observed_OD, self.val_mask)

            log.epoch.append(ep)
            log.train_nll.append(float(loss.item()))
            log.val_nll.append(val_nll)
            log.cpc_val.append(cpc_val)
            log.alpha.append(float(self.rum.alpha.item()))
            log.beta_mean.append(float(self.rum.beta_t.mean().item()))
            log.beta_c.append(float(self.rum.beta_c.item()))
            log.gamma.append(float(self.rum.gamma.item()))
            log.topk_calls = self._topk_calls

            improved = val_nll < best_val - 1e-4
            if improved:
                best_val = val_nll
                no_improve = 0
                best_state = {
                    "gnn": {k: v.detach().clone() for k, v in self.gnn.state_dict().items()},
                    "rum": {k: v.detach().clone() for k, v in self.rum.state_dict().items()},
                }
            else:
                no_improve += 1

            if self.verbose:
                print(f"ep {ep:3d} | train_nll {loss.item():.4f} | "
                      f"val_nll {val_nll:.4f} | cpc {cpc_val:.3f} | "
                      f"alpha {self.rum.alpha.item():.3f} "
                      f"beta_mean {self.rum.beta_t.mean().item():+.4f} "
                      f"beta_c {self.rum.beta_c.item():+.4f} "
                      f"gamma {self.rum.gamma.item():.3f}")

            if no_improve >= self.patience:
                if self.verbose:
                    print(f"early stop at epoch {ep} (patience {self.patience})")
                break

        if best_state is not None:
            self.gnn.load_state_dict(best_state["gnn"])
            self.rum.load_state_dict(best_state["rum"])

        return (
            self.gnn,                                          # theta_hat
            float(self.rum.alpha.item()),                       # alpha_hat
            self.rum.beta_t.detach().clone(),                   # beta_t_hat (T,)
            float(self.rum.beta_c.item()),                      # beta_c_hat
            float(self.rum.gamma.item()),                       # gamma_hat
            log,                                                # TrainingLog
        )

    # ------------------------------------------------------------------ predict_OD
    def predict_OD(self, t_ij_override: Optional[torch.Tensor] = None,
                   norm_stats: Optional[tuple] = None) -> torch.Tensor:
        """Predict (T, N, N) flow probabilities under the trained model.

        Parameters
        ----------
        t_ij_override : optional (T,N,N) or (N,N) cost. If supplied, used in
                        place of self.t_ij_t (for temporal-holdout test eval).
        norm_stats    : optional (mean, std) for V_jt — when forwarding a
                        counterfactual ``self.X`` (Scenario A/B), freeze the
                        scaling at baseline so the intervention is not washed
                        out by re-centring on the perturbed output.

        Returns
        -------
        F_pred : (T, N, N) — predicted flow counts. Each row's mass equals
                 the corresponding row's mass in observed_OD (production-
                 constrained), so callers can compare directly against
                 observed flows or sum to row totals.
        """
        self.gnn.eval(); self.rum.eval()
        with torch.no_grad():
            V_jt = self._forward_V(self.X, norm_stats=norm_stats)
            if self.residualise:
                V_jt = _fwl_inspired_ridge_penalty(V_jt, self.t_ij_t, self.log_d_ij)
            t_use = self.t_ij_t if t_ij_override is None else t_ij_override.to(self.device).float()
            if t_use.dim() == 2:
                t_use = t_use.unsqueeze(0).expand(self.T, self.N, self.N)
            elif t_use.dim() == 3 and t_use.shape[0] != self.T:
                t_use = t_use.mean(0, keepdim=True).expand(self.T, self.N, self.N)
            log_p = self._per_hour_log_p_full(V_jt, t_use, self.log_d_ij)
            P = log_p.exp()                              # (T, N, N)
            row_sum = self.observed_OD.sum(dim=2, keepdim=True)
            return (P * row_sum).cpu()
