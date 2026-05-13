"""LondonV2Model — Mesa model orchestrating the 4-step v2 architecture.

Per step (one equilibrium iteration):
  1. Each agent samples workplace using their personal V_ij^t (from NN-utility + GNN encoder)
  2. Aggregate agent flows per (origin grid, hour) and (destination grid, hour)
  3. Recompute congestion + travel time t_ij^t using BPR with agent demand
  4. Optionally re-sample (if iter<max_iters)

The model expects pre-computed: STGNN V_j(t), per-SOC OccMatch matrix, NN-utility model.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mesa

from agents import CommuterAgent
from models_lib.occupation_match import occupation_match_per_soc, grid_industry_vector
from models_lib.linear_utility import LinearUtility


class LondonV2Model(mesa.Model):
    def __init__(
        self,
        agents_df: pd.DataFrame,            # synthesized population
        grid_features_df: pd.DataFrame,
        V_jt_baseline: np.ndarray,          # (T=24, N) STGNN baseline V_j
        t_ij_t_initial: np.ndarray,         # (T, N, N) initial travel time (free-flow + TomTom or BPR baseline)
        log_d_ij: np.ndarray,               # (N, N) log distance km
        linear_utility_model: LinearUtility,
        norm_stats: dict,                   # mean/std for V_jt, t_log, d_log, om
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.grid_features_df = grid_features_df.reset_index(drop=True)
        self.N = len(grid_features_df)
        self.T = V_jt_baseline.shape[0]

        # Preserve baseline tensors (RAW values, normalized at inference)
        self.V_jt_raw = torch.tensor(V_jt_baseline, dtype=torch.float32)
        self.t_ij_t_raw = torch.tensor(t_ij_t_initial, dtype=torch.float32).clone()
        self.log_d_ij_raw = torch.tensor(log_d_ij, dtype=torch.float32)

        # Normalization stats (from training)
        self.V_jt_mean = float(norm_stats["V_jt_mean"])
        self.V_jt_std = float(norm_stats["V_jt_std"])
        self.t_log_mean = float(norm_stats["t_log_mean"])
        self.t_log_std = float(norm_stats["t_log_std"])
        self.d_log_mean = float(norm_stats["d_log_mean"])
        self.d_log_std = float(norm_stats["d_log_std"])
        self.om_mean = float(norm_stats["om_mean"])
        self.om_std = float(norm_stats["om_std"])

        # Per-SOC OccMatch (9, N) — aligned to grid order
        gx = grid_industry_vector(grid_features_df)
        self.om_per_soc = occupation_match_per_soc(gx)  # (9, N)
        # Normalize using training stats so model sees same scale as in training
        self.om_per_soc_norm = (self.om_per_soc - self.om_mean) / (self.om_std + 1e-8)
        self.om_per_soc_norm_t = torch.tensor(self.om_per_soc_norm, dtype=torch.float32)

        # Pre-normalize V_jt and log_d_ij (these are static across iterations)
        self.V_jt_norm = (self.V_jt_raw - self.V_jt_mean) / (self.V_jt_std + 1e-8)
        self.log_d_ij_norm = (self.log_d_ij_raw - self.d_log_mean) / (self.d_log_std + 1e-8)

        # t_ij_t is mutable across iterations (re-normalized each step)
        self._update_t_norm()

        self.utility = linear_utility_model
        self.utility.eval()

        # Build agents
        self._spawn_agents(agents_df)

        # State
        self.iter_count = 0
        self.flow_per_grid_hour = None  # filled during step()

    def _spawn_agents(self, agents_df: pd.DataFrame):
        soc_to_idx = {f"soc{i}": i - 1 for i in range(1, 10)}
        for _, row in agents_df.iterrows():
            agent = CommuterAgent(
                unique_id=int(row["agent_id"]),
                model=self,
                home_grid_idx=int(row["home_grid_idx"]),
                soc=row["soc"],
                soc_idx=soc_to_idx.get(row["soc"], 0),
                income_tier=int(row["income_tier"]),
                age_band=row["age_band"],
                mode=row["mode_initial"],
                departure_hour=int(row.get("departure_hour", 8)),  # default 8am
            )

    def _update_t_norm(self):
        """Recompute normalized t_ij_t after congestion update."""
        t_log = torch.log1p(self.t_ij_t_raw)
        self.t_ij_t_norm = (t_log - self.t_log_mean) / (self.t_log_std + 1e-8)

    def compute_log_p_for(self, origin_idx: int, hour: int, soc_idx: int,
                           income_tier: int = None, mode_idx: int = None) -> np.ndarray:
        """
        Compute log P(j | origin, hour, soc) for one agent.
        Dispatches based on utility type:
          - HeterogeneousUtility: uses utility_personal with agent's (income_tier, mode_idx)
          - LinearUtility: simple call with normalized features
        """
        Vj = self.V_jt_norm[hour]
        om = self.om_per_soc_norm_t[soc_idx]
        tt = self.t_ij_t_norm[hour, origin_idx]
        d  = self.log_d_ij_norm[origin_idx]

        with torch.no_grad():
            if hasattr(self.utility, "utility_personal") and income_tier is not None and mode_idx is not None:
                # v2.2 heterogeneous: scale β to normalized t_ij
                # β_personal × t_ij_minutes ≈ β_personal × t_log_std × t_norm (log-scaling approximation)
                t_log_std = float(self.t_log_std)
                # Trick: pre-scale β when calling utility_aggregate (we override here)
                # Compute personal V_ij = α_V·V_norm + α_occ·om − β_personal·t_log_std·t_norm − γ·d_norm + bias
                util = self.utility
                beta = util.beta(income_tier, mode_idx) * t_log_std
                V_ij = (util.alpha_V * Vj + util.alpha_occ * om - beta * tt - util.gamma * d + util.bias)
            else:
                V_ij = self.utility(Vj, om, tt, d)
            log_p = torch.log_softmax(V_ij, dim=0).numpy()
        return log_p

    def step(self, recompute_congestion: bool = True):
        """
        One equilibrium iteration:
          - Each agent samples workplace based on current t_ij_t
          - Aggregate flows
          - Update congestion (if recompute_congestion=True)
        """
        # Track flow per (hour, origin_grid, dest_grid)
        flow = np.zeros((self.T, self.N, self.N), dtype=np.float32)
        agent_changes = 0
        total = 0

        # Mode index lookup
        from models_lib.heterogeneous_utility import MODE_TO_IDX
        for agent in self.agents:
            mode_idx = MODE_TO_IDX.get(agent.mode, 0)
            log_p = self.compute_log_p_for(
                agent.home_grid_idx, agent.departure_hour, agent.soc_idx,
                income_tier=agent.income_tier, mode_idx=mode_idx,
            )
            new_work = int(agent.sample_workplace(log_p, self.rng))
            if agent.work_grid_idx is not None and agent.work_grid_idx != new_work:
                agent_changes += 1
            agent.work_grid_idx = new_work
            agent.commute_time = float(self.t_ij_t_raw[agent.departure_hour, agent.home_grid_idx, new_work])

            # Add to flow tensor
            flow[agent.departure_hour, agent.home_grid_idx, new_work] += 1.0
            total += 1

        self.flow_per_grid_hour = flow

        # Update congestion if requested
        if recompute_congestion:
            self._update_congestion(flow)

        self.iter_count += 1
        self.last_step_info = {
            "iter": self.iter_count,
            "n_agents": total,
            "agent_changes": agent_changes,
            "convergence_rate": agent_changes / max(total, 1),
        }
        return self.last_step_info

    def _update_congestion(self, flow: np.ndarray):
        """
        BPR-style update of t_ij_t using agent flow as demand.
        Uses 95th percentile of nonzero inflow as capacity proxy → ratio ~1 in typical grid.
        Multiplier capped at 2.5x to avoid runaway with sparse agents.
        """
        inflow_per_grid_hour = flow.sum(axis=1)  # (T, N)
        nonzero = inflow_per_grid_hour[inflow_per_grid_hour > 0]
        if len(nonzero) > 0:
            capacity = float(np.percentile(nonzero, 95))
        else:
            capacity = 1.0
        capacity = max(capacity, 1.0)

        alpha = 0.15
        beta = 2.0
        ratio = inflow_per_grid_hour / capacity
        bpr_mult = 1.0 + alpha * (ratio ** beta)
        bpr_mult = np.clip(bpr_mult, 1.0, 2.5)   # cap multiplier
        bpr_t = torch.tensor(bpr_mult, dtype=torch.float32)

        if not hasattr(self, "_t_ij_baseline_raw"):
            self._t_ij_baseline_raw = self.t_ij_t_raw.clone()
        self.t_ij_t_raw = self._t_ij_baseline_raw * bpr_t[:, None, :]
        self._update_t_norm()

    def collect_metrics(self) -> dict:
        """Aggregate metrics for reporting."""
        flow = self.flow_per_grid_hour
        if flow is None:
            return {}
        total_trips = float(flow.sum())
        commute_times = [a.commute_time for a in self.agents if a.commute_time is not None]
        # Inflow per grid (24h)
        inflow_24h = flow.sum(axis=(0, 1))  # (N,)
        # Outflow per grid
        outflow_24h = flow.sum(axis=(0, 2))
        # Mean commute time
        mean_ct = float(np.mean(commute_times)) if commute_times else 0.0

        # Peak hour ratio
        peak_hour_flow = flow.sum(axis=(1, 2))  # (T,)
        peak_hour = int(peak_hour_flow.argmax())

        return {
            "total_trips": total_trips,
            "mean_commute_time": mean_ct,
            "peak_hour": peak_hour,
            "peak_hour_flow": float(peak_hour_flow[peak_hour]),
            "inflow_24h": inflow_24h,
            "outflow_24h": outflow_24h,
            "flow_per_hour": peak_hour_flow,
        }
