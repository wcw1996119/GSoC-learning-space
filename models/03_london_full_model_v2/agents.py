"""CommuterAgent — Mesa agent for v2 4-step architecture.

Each agent:
  - has a fixed home_grid_idx, soc, income_tier, age_band, mode_initial, departure_hour
  - on each model.step(), samples workplace using personal P(j | home, hour, soc)
"""
import numpy as np
import mesa


class CommuterAgent(mesa.Agent):
    def __init__(
        self,
        unique_id: int,
        model,
        home_grid_idx: int,
        soc: str,
        soc_idx: int,         # 0..8 for soc1..soc9
        income_tier: int,     # 1/2/3 (low/mid/high)
        age_band: str,
        mode: str,            # "car" | "pt" | "active"
        departure_hour: int,  # 6-21
    ):
        super().__init__(model)
        self.unique_id_ = unique_id  # mesa 3+ exposes self.unique_id auto, keep ours
        self.home_grid_idx = home_grid_idx
        self.soc = soc
        self.soc_idx = soc_idx
        self.income_tier = income_tier
        self.age_band = age_band
        self.mode = mode
        self.departure_hour = departure_hour
        self.work_grid_idx = None   # set by step()
        self.commute_time = None    # filled after sampling

    def sample_workplace(self, log_p_per_origin_hour: np.ndarray, rng):
        """
        log_p_per_origin_hour: (N,) log P(j | home, departure_hour, this agent's SOC)
        rng: np.random.Generator

        Returns chosen j (int).
        """
        p = np.exp(log_p_per_origin_hour - log_p_per_origin_hour.max())
        p = p / p.sum()
        return rng.choice(len(p), p=p)
