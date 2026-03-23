# Money Model with Land Value

## What this model does

A Boltzmann Wealth Model extended with a PropertyLayer representing land value.
Agents move randomly on a 10×10 grid, give money to cellmates, and land value 
rises where agents congregate and decays elsewhere.

Mesa features used:
- CellAgent and OrthogonalMooreGrid for discrete space
- PropertyLayer for environment properties
- DataCollector for Gini coefficient tracking
- SolaraViz with SpaceRenderer for interactive visualization
- Custom Solara component (WealthHistogram)

## What I learned

This was my first Mesa model. Key things I understood through building it:

- The relationship between Agent and Model: Agent handles individual behavior 
  in step(), Model handles global logic in its own step(). I initially put 
  Model-level logic inside Agent.step() which caused an AttributeError 
  (agents don't have model.agents directly accessible that way).
- AgentSet is automatically maintained — no need to manually add/remove agents.
- PropertyLayer sits on the grid as a numpy array, updated each step in Model.step().
- SolaraViz requires all Model __init__ parameters to be keyword-only (use * in signature).

## What was hard

At first I didn't understand why a Model class was needed at all. I thought just defining agents and their rules would be enough. Then I assumed Model was simply a container for agents, but gradually realized it holds everything: agents, space, data collection, and time progression. This misconception also caused confusion about the order of class definitions. I initially thought MoneyAgent had to be defined after MoneyModel because Model "contains" agents, but learned that class definitions are just declarations, and the actual agent instances are only created inside MoneyModel.__init__() when create_agents() is called.

## What I'd do differently

- Remove the debug print statement left in __init__
- Use dynamic vmax in propertylayer_portrayal instead of fixed 100

## How to run
```bash
conda activate mesa-demo
solara run model.py
```