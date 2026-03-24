import mesa_geo as mg


class MSOAAgent(mg.GeoAgent):
    """Represents an MSOA zone in London."""

    def __init__(self, model, geometry, crs):
        super().__init__(model, geometry, crs)
        self.msoa_code = None
        self.msoa_name = None
        self.lat = None
        self.lon = None
        self.employment_attraction = 0
        self.accessibility = 0.0

    def step(self):
        pass


class CommuterAgent(mg.GeoAgent):
    """
    Represents a commuter with a fixed workplace assigned at initialisation
    based on real OD flow probabilities.
    """

    def __init__(self, model, geometry, crs):
        super().__init__(model, geometry, crs)
        self.home_msoa = None
        self.chosen_work_msoa = None
        self.commute_mode = None  # 'car', 'pt', 'active'
        self.commute_time_minutes = 0.0
        self.occupation = None  # 'soc1' to 'soc9'

    def step(self):
        pass  # Workplace is fixed; no re-selection each step