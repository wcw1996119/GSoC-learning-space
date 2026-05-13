# What this model does — plain-language summary

## 1. The high-level picture

We divide Greater London into 1 km × 1 km grid cells. Each cell is a "place".
The model answers: **for every hour of the day, how many people commute from where to where?**

We can ask a "what if" question — for example, *what if we add 50,000 jobs to grid X?* — and the model
predicts the resulting commute pattern: which origins send more commuters, which corridors get busier, and how
average commute time shifts. This is what the literature calls a **counterfactual** prediction.

## 2. The two parts of the model

The model has two pieces working together:

- **GNN (graph neural network)** — learns how attractive each grid is as a workplace. It looks at each grid's
  features (jobs, sector mix, POIs, population, subway stations, …) plus the features of nearby grids
  (via message passing on the spatial graph). It outputs an attractiveness score `V_j(t)` — one value per
  grid per hour.

- **RUM closure (random utility model)** — combines the attractiveness `V_j` with the commute cost
  `β · t_ij(t)` and applies softmax to get `P(j | i, t)`, the probability that a commuter living in `i`
  chooses workplace `j` at hour `t`. This is the standard discrete-choice model from McFadden 1974
  (50+ years of behavioural economics).

In one line: **GNN learns "where is attractive", RUM combines attractiveness and travel cost into choice
probabilities.**

## 3. How counterfactuals work

Three steps:

1. **Change the input** — e.g. raise the "jobs" feature of grid X from 20,000 to 70,000.
2. **GNN re-runs forward** — it produces a new `V_j(t)` for *all* grids (because of message passing,
   neighbours' attractiveness shifts too).
3. **RUM re-runs** — new `V_j` plugged into the choice formula gives a new commute distribution.

No retraining is needed. This is forward inference only.

## 4. The three layers of credibility evidence

A model can output any number — but should you believe it? We use three checks:

- **Plausibility (how far is the scenario from training data?)** — k-NN distance from the perturbed
  input to the training distribution. If it's far away (out-of-distribution / OOD), only trust the
  *direction* of the prediction, not the magnitude.

- **Magnitude scan (does the model behave consistently?)** — sweep the perturbation magnitude
  (0, +5k, +10k, +20k, +50k, +100k) and check whether the predicted response is monotonic and smooth.
  Kinks or jumps mean the model is breaking down.

- **Multi-model cross-check (do simpler baselines agree on the direction?)** — run the same scenario
  through (a) plain gravity, (b) RUM with observed-prior, (c) the v2 STGNN+RUM. If all three agree on
  the direction, that's strong evidence; if they disagree, the result is fragile.

## 5. What you can and cannot trust right now

✅ **Can trust:**
- Direction of effects (more jobs at X → more commuters going to X — yes)
- Spatial spillover patterns (which neighbour grids absorb the change)
- In-distribution scenarios (small perturbations to typical residential / mixed grids)

❌ **Cannot trust:**
- Exact magnitudes when plausibility is in the "far OOD" tier
- Single-grid extreme deltas (e.g. +200k jobs in one cell — no training example resembles that)
- Anything that needs true hourly OD data to validate (we used synthesized hourly OD)

⚠️ **Important framing:**
- **London is a development sandbox**. The methodology will not appear in the final paper.
- The real research target is **Beijing** (data ethics review pending). Once that clears, the same
  framework will be applied to Beijing with new providers (Amap routing, Beijing yearbook, Amap POIs).
- Model code is data-source-agnostic — no rewrite needed when migrating.

## 6. Relation to v1

v1 (`02_london_commuting_model/`) is a simple ABM deployed on Hugging Face. It uses fixed mode shares,
no GNN, and BPR-only travel times. It exists for public demonstration.

v2 (this repo) is the actual research framework — GNN + RUM + congestion equilibrium + three-evidence
counterfactual evaluation. Designed to be portable to Beijing.

## 7. Files to read for depth

- `methodology/03_node_features.md` — 22 static + 4 temporal feature design
- `methodology/04_edge_features.md` — kNN graph + edge attributes
- `methodology/05_training_loss.md` — loss spec, training procedure, ablations
- `../02_london_commuting_model/methodology/02_counterfactual_evaluation.md` —
  three-evidence framework
