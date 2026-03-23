import numpy as np
import mesa
import solara
from matplotlib.figure import Figure

from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.utils import update_counter

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B


class MoneyAgent(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.wealth = 1
        self.cell = cell

    def move(self):
        neighbors = list(self.cell.connections.values())
        new_cell = self.random.choice(neighbors)
        self.cell = new_cell

    def give_money(self):
        cellmates = [a for a in self.cell.agents if a is not self]
        if cellmates and self.wealth > 0:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

    def step(self):
        self.move()
        self.give_money()


class MoneyModel(mesa.Model):
    def __init__(self, *, n=50, width=10, height=10, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = OrthogonalMooreGrid((width, height), random=self.random)

        # Create a PropertyLayer to represent land value
        self.grid.create_property_layer("land_value", default_value=0.0)
        print("land_value data:", self.grid.land_value.data)
        # Random initialization，land price higher in central areas
        cx, cy = width // 2, height // 2
        for x in range(width):
            for y in range(height):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                self.grid.land_value.data[x, y] = max(0, 100 - dist * 10)

        MoneyAgent.create_agents(
            self, n,
            self.random.choices(self.grid.all_cells.cells, k=n),
        )

        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"},
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        # decrease overall
        self.grid.land_value.data *= 0.95
        # Land price arises where there are agents
        for agent in self.agents:
            x, y = agent.cell.coordinate
            self.grid.land_value.data[x, y] += 5
        self.datacollector.collect(self)


# Agent portrayal：blue with wealth, red without
def agent_portrayal(agent):
    return AgentPortrayalStyle(
        color="tab:blue" if agent.wealth > 0 else "tab:red",
        size=100 if agent.wealth > 0 else 20,
    )


# PropertyLayer portrayal：land value
def propertylayer_portrayal(layer):
    if layer.name == "land_value":
        return PropertyLayerStyle(
            colormap="YlOrRd",
            vmin=0,
            vmax=100,
            alpha=0.4,
            colorbar=True,
        )


model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}

money_model = MoneyModel(n=50, width=10, height=10)

renderer = SpaceRenderer(model=money_model, backend="matplotlib")
renderer.draw_structure(lw=1, ls="solid", color="black", alpha=0.2)
renderer.setup_propertylayer(propertylayer_portrayal)
renderer.draw_propertylayer()
renderer.setup_agents(agent_portrayal)
renderer.draw_agents()


@solara.component
def WealthHistogram(model):
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    wealth_vals = [agent.wealth for agent in model.agents]
    ax.hist(wealth_vals, bins=10, color="tab:blue", edgecolor="white")
    ax.set_title("Wealth Distribution")
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Number of Agents")
    solara.FigureMatplotlib(fig)

GiniPlot = make_plot_component("Gini")

Page = SolaraViz(
    money_model,
    renderer,
    components=[GiniPlot, WealthHistogram],
    model_params=model_params,
    name="Money Model with Land Value",
)