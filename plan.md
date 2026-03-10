# March Madness MOE: Architecture & Project Plan

## Context

Build a March Madness prediction system targeting the **2026 tournament** (Selection Sunday: March 15). The system uses a **Mixture of Experts (MOE)** architecture that mirrors a production advertising allocation system: three expert models with **multi-task ranking losses** (each expert evaluated on a different strategic dimension via Spearman correlation), a gating network that learns context-dependent expert weights, and a Monte Carlo simulation engine that consumes blended probabilities to generate bracket predictions with confidence intervals.

### Production System Mapping

| Production Component | March Madness Analog | What It Demonstrates |
|---------------------|---------------------|---------------------|
| Spend:sales hill curve models (deltas) | XGBoost on KenPom feature deltas | Tree models on relative difference features |
| Scale expert (efficiency at volume) | Seed-baseline expert | Reliable baseline predictions |
| Growth expert (NTB-weighted growth) | Efficiency-delta expert | Finding hidden value beyond surface metrics |
| Competitive expert (rank improvement/spend) | Uncertainty-calibration expert | Identifying where marginal attention matters |
| `MultiTaskHistoricalRankLoss` | Per-expert Spearman correlation loss | Expert specialization via evaluation criteria |
| MC simulation engine | MC bracket simulation | Simulating outcomes under uncertainty |
| MOE opportunity scoring | Gating network blending | Context-dependent expert weighting |

---

## Architecture Overview

```
  KenPom Data (Kaggle 2002-2025 + kenpompy 2026)
  + Tournament Results (Kaggle March ML Mania)
                    |
                    v
          Feature Engineering
     (KenPom deltas + context features)
                    |
     +--------------+--------------+
     |              |              |
     v              v              v
  Expert 1      Expert 2      Expert 3
  Seed-baseline Eff-delta     Uncertainty
  Spearman vs   Spearman vs   Spearman vs
  seed-implied  AdjEM delta   certainty score
     |              |              |
     +--------------+--------------+
                    |
                    v
          Gating Network (PyTorch)
     (seed_diff, round, adjEM_delta,
      luck_delta) -> expert weights
                    |
                    v
          Blended P(higher seed wins)
          w1*P1 + w2*P2 + w3*P3
                    |
                    v
          MC Simulation Engine
          10K bracket simulations
                    |
                    v
          Bracket Predictions
          + Per-team advancement probs
          + Confidence Intervals
```

**Key architectural principle**: MC simulation is an **infrastructure layer** downstream of the MOE, not a peer expert. In the production system, MC consumed MOE opportunity scores to simulate allocation outcomes — it didn't feed into the MOE. Same pattern here: the MOE produces game-level blended probabilities; MC consumes them for bracket-level simulation.

---

## Project Structure

```
march-madness-moe/
├── pyproject.toml
├── .env                          # KenPom credentials (gitignored)
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── kenpom/               # Kaggle historical + 2026 scraped
│   │   ├── tournament/           # Bracket results, seeds, matchups
│   ├── processed/
│   │   ├── features/             # Per-year feature matrices
│   │   └── targets/              # Per-year outcomes
│   └── cache/                    # Scraping cache
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_single_tree_baseline.ipynb
│   ├── 04_multi_expert_dev.ipynb
│   ├── 05_gating_network_dev.ipynb
│   ├── 06_backtesting_analysis.ipynb
│   ├── 07_mc_simulation_dev.ipynb
│   └── 08_bracket_output.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, hyperparams, constants
│   ├── data/
│   │   ├── scraper.py            # kenpompy wrapper (2026 only)
│   │   ├── kaggle_loader.py      # Historical KenPom data (2002-2025)
│   │   ├── tournament_data.py    # Bracket results, seeds, matchups
│   │   └── merge.py              # Join KenPom stats with matchups
│   ├── features/
│   │   ├── kenpom_deltas.py      # Team A - Team B metric deltas
│   │   ├── context_features.py   # Seed diff, round, conference flags
│   │   ├── ranking_criteria.py   # Per-expert ranking target computation
│   │   └── pipeline.py           # Orchestrate full feature pipeline
│   ├── models/
│   │   ├── tree_expert.py        # XGBoost expert wrapper (sample-weighted)
│   │   ├── multi_task_experts.py # PyTorch multi-head with torchsort Spearman
│   │   ├── gating_network.py     # PyTorch MOE gating MLP
│   │   ├── moe_ensemble.py       # Orchestrates experts + gating
│   │   └── ranking_loss.py       # MultiTaskHistoricalRankLoss implementation
│   ├── evaluation/
│   │   ├── backtester.py         # Nested LOYO CV (experts + gating)
│   │   ├── metrics.py            # Log-loss, accuracy, Brier, bracket score
│   │   └── calibration.py        # Reliability diagrams, per-expert calibration
│   ├── simulation/
│   │   ├── mc_engine.py          # Monte Carlo bracket simulation
│   │   └── log5.py               # Log5 baseline win probability
│   ├── bracket/
│   │   ├── structure.py          # Bracket data model (regions, slots)
│   │   ├── predictor.py          # Full bracket generation from MOE + MC
│   │   └── visualizer.py         # Bracket rendering
│   └── utils/
│       ├── team_names.py         # Name normalization across sources
│       └── constants.py          # Seeds, regions, rounds
├── scripts/
│   ├── scrape_kenpom.py          # CLI: scrape 2026 KenPom data
│   ├── build_features.py         # CLI: run feature pipeline
│   ├── train_baseline.py         # CLI: train single-tree baseline
│   ├── train_experts.py          # CLI: train multi-expert MOE
│   ├── run_backtest.py           # CLI: full nested LOYO backtesting
│   └── predict_bracket.py        # CLI: generate 2026 bracket
├── models/                       # Saved model artifacts
└── tests/
```

---

## Module Details

### Data Layer

**`src/data/kaggle_loader.py`** — Load Pilafas Kaggle dataset (2002-2025 KenPom metrics for all D1 teams). Normalize column names to standard schema. Returns DataFrame keyed by `(year, team)`.

**`src/data/scraper.py`** — Wraps `kenpompy` to scrape 2026 season only (efficiency + four factors). Caches to `data/cache/` to avoid re-scraping. Requires KenPom subscription ($20/year).

**`src/data/tournament_data.py`** — Loads historical tournament matchups from Kaggle March ML Mania data (`NCAATourneyCompactResults.csv`, `NCAATourneySeeds.csv`). Provides `(year, round, seed_A, seed_B, team_A, team_B, score_A, score_B, winner)`.

**`src/data/merge.py`** — Joins KenPom team-level stats with tournament matchup records. Uses `team_names.py` for cross-source name normalization. **This is the most fragile module** — KenPom, Kaggle, and NCAA all use different team names (~20-30 need manual mapping).

### Feature Engineering

**`src/features/kenpom_deltas.py`** — Core features. For each matchup, compute higher-seed minus lower-seed deltas:
- AdjEM, AdjO, AdjD, AdjT (efficiency metrics)
- Luck, SOS, SOR, NCSOS (strength/schedule)
- Four factors: eFG%, TO%, OR%, FTRate (offense + defense = 8 features)
- ~25-30 total delta features

Convention: Always `higher_seed - lower_seed`. For same-seed matchups, use KenPom ranking as tiebreaker.

**`src/features/context_features.py`** — Game context beyond raw KenPom:
- Seed differential (integer)
- Round number (1-6)
- Same-conference flag
- Historical upset rate for this seed matchup

**`src/features/ranking_criteria.py`** — Computes per-expert ranking targets from input features. These are the Spearman correlation targets each expert is evaluated against:

```python
def seed_implied_probability(seed_diff):
    """Expert 1 target: historical win rate for this seed matchup.
    Monotonic function of |seed_diff|. 1v16 -> 0.99, 8v9 -> 0.51."""

def efficiency_delta_rank(adjEM_delta):
    """Expert 2 target: normalized AdjEM delta magnitude.
    Captures where team quality diverges from seed expectations."""

def game_certainty_score(seed_diff, adjEM_delta, luck_delta, ncsos_delta):
    """Expert 3 target: composite game predictability score.
    certainty = 0.4 * |seed_diff|/15
              + 0.3 * |adjEM_delta|/max_delta
              + 0.2 * (1 - |luck_delta|/max_luck)
              + 0.1 * (1 - |ncsos_delta|/max_ncsos)
    Expert 3 is trained so |P - 0.5| tracks this score:
    high certainty games should get confident predictions,
    low certainty games should get hedged predictions."""
```

### Models

**`src/models/tree_expert.py`** — XGBoost binary classifier wrapper with support for **sample weighting** (Tier 1 implementation). Input: KenPom delta features. Output: P(higher seed wins). Default params: `max_depth=4, lr=0.05, n_estimators=300, objective=binary:logistic`. Early stopping on validation set.

Three expert variants via different sample weights:
- **Seed-baseline expert**: Uniform weights (standard training)
- **Efficiency-delta expert**: Weights proportional to `|adjEM_delta|` — focuses learning on games where KenPom quality diverges from seed expectations
- **Uncertainty-calibration expert**: Weights inversely proportional to `|seed_diff|` — focuses learning on close-seed matchups where outcomes are most uncertain

**`src/models/multi_task_experts.py`** — (Tier 2 / stretch) PyTorch multi-head network with `torchsort` differentiable Spearman loss:

```
Shared backbone:
  Input(~30 KenPom deltas) -> Linear(32) -> ReLU -> Dropout(0.3)
  -> Linear(24) -> ReLU

Three output heads:
  Head 1 (Seed-baseline):   Linear(24) -> Linear(1) -> Sigmoid
  Head 2 (Efficiency-delta): Linear(24) -> Linear(1) -> Sigmoid
  Head 3 (Uncertainty-cal):  Linear(24) -> Linear(1) -> Sigmoid
```

Each head trained with `MultiTaskHistoricalRankLoss` — the shared backbone learns general KenPom representations while each head specializes via its ranking criterion.

**`src/models/ranking_loss.py`** — Implements `MultiTaskHistoricalRankLoss`:

```python
class MultiTaskHistoricalRankLoss:
    """Each expert is evaluated on Spearman correlation against
    its own strategic ranking criterion.

    Uses torchsort.soft_rank for differentiable ranking.

    Loss = BCE(P, y) + lambda * (1 - SpearmanR(P, ranking_target))

    where ranking_target differs per expert:
      - Expert 1: seed_implied_probability
      - Expert 2: adjEM_delta (normalized)
      - Expert 3: game_certainty_score
    """
```

For Expert 3 specifically, the Spearman is computed between `|P - 0.5|` (model confidence) and `game_certainty_score`, so the expert learns to calibrate its uncertainty.

**`src/models/gating_network.py`** — PyTorch MLP:
```
Input(~4-6 context features) -> Linear(16) -> ReLU -> Dropout(0.2)
-> Linear(8) -> ReLU -> Linear(3) -> Softmax
```
Intentionally tiny — only ~1,260 training games exist. Outputs `[w_seed, w_eff, w_unc]` expert weights.

Context features: `(seed_diff, round, adjEM_delta, luck_delta)`. Minimal and interpretable — the gating network is barely more than a learned lookup table by game context.

**Trained via nested CV** (see Backtesting Protocol) — never sees the same data the experts trained on.

**`src/models/moe_ensemble.py`** — Orchestrator:
1. Get `p_seed = seed_baseline_expert.predict_proba(matchup)`
2. Get `p_eff = efficiency_delta_expert.predict_proba(matchup)`
3. Get `p_unc = uncertainty_calibration_expert.predict_proba(matchup)`
4. Get `[w_seed, w_eff, w_unc] = gating_network(context_features)`
5. Return `w_seed * p_seed + w_eff * p_eff + w_unc * p_unc`
6. For bracket generation: feed blended probabilities to MC engine

### Simulation

**`src/simulation/mc_engine.py`** — Monte Carlo tournament simulator. Takes bracket structure + `win_prob_fn` (the MOE's blended probabilities), runs 10K simulations. Each sim: draw Bernoulli(P(A wins)) at each game, propagate winners through all 6 rounds. Output: per-team probability of reaching each round, most likely bracket, confidence intervals.

**`src/simulation/log5.py`** — Log5 baseline win probability function. Uses team win percentages to compute game-level probabilities. Used as a sanity-check comparator against the MOE.

### Evaluation

**`src/evaluation/backtester.py`** — **Nested Leave-One-Year-Out CV** (see full protocol below).

**`src/evaluation/metrics.py`** — Log-loss (primary), accuracy, Brier score, ESPN bracket score (10/20/40/80/160/320 pts by round), round-by-round breakdown.

**`src/evaluation/calibration.py`** — Reliability diagrams for each expert individually and the blended MOE. Per-expert calibration reveals whether each expert is well-calibrated within its domain.

---

## Backtesting Protocol: Nested LOYO CV

The backtesting protocol uses **nested cross-validation** to prevent information leakage between expert training and gating training. The gating network never sees data the experts were trained on.

### Protocol

For each held-out test year T in {2003-2025, skip 2020} — 22 folds:

```
Step 1: Set aside year T as TEST SET (~60 games)
        Remaining years = TRAIN_POOL (~1,200 games)

Step 2: INNER LOOP — Generate held-out expert predictions for gating training
        For each year V in TRAIN_POOL:
          - Train 3 experts on TRAIN_POOL minus V (~1,140 games)
          - Predict year V games with each expert
          - Record: (year_V_games, P_seed, P_eff, P_unc)
        Result: held-out expert predictions for ALL years in TRAIN_POOL

Step 3: TRAIN GATING on inner-loop held-out predictions
        Input: (context_features, P_seed, P_eff, P_unc) for all TRAIN_POOL games
        Target: actual game outcomes
        The gating network learns expert weights from expert predictions
        it has NEVER seen the experts train on

Step 4: TRAIN FINAL EXPERTS on full TRAIN_POOL
        3 experts trained on all ~1,200 non-test games
        (these are the experts that will predict the test year)

Step 5: PREDICT TEST YEAR T
        - Each final expert predicts year T games
        - Gating network (from Step 3) produces expert weights
        - Blended probability = weighted sum
        - Record full decomposition per game

Step 6: SCORE
        Record per game:
        (year, round, seed_A, seed_B,
         P_seed, P_eff, P_unc,              # expert predictions
         w_seed, w_eff, w_unc,              # gating weights
         P_blend,                            # blended prediction
         outcome)                            # actual result (0/1)
```

### Why Nested CV

Simple LOYO (train experts + gating together on same data) has a subtle information leak: the gating network learns to compensate for the experts' training-set-specific biases rather than their genuine predictive strengths. Nested CV ensures the gating sees expert predictions on data the experts haven't trained on — the same regime it will face at test time.

This mirrors the production system: experts were trained on historical data, and the MOE scoring engine was trained on a recent holdout window of expert performance — never on the same data the experts learned from.

### Validation Artifacts

The ~1,260 held-out predictions (22 test years × ~60 games) produce these portfolio artifacts:

| Artifact | What It Proves |
|----------|---------------|
| Log-loss: MOE vs single-tree vs seed-baseline | Does the architecture improve predictions? |
| Expert weights by round (heatmap) | Does gating learn round-dependent trust? |
| Expert weights by seed_diff (line plot) | Does gating shift strategy for close vs blowout matchups? |
| Per-expert accuracy when dominant (w > 0.5) | Is each expert good at its claimed specialty? |
| Calibration plot: per-expert + blended | Are seed-baseline predictions well-calibrated? Is uncertainty expert appropriately hedged? |
| Year-over-year log-loss (line chart) | Is performance stable across eras? |
| Expert weight evolution (faceted heatmap) | **Portfolio killer feature**: heat map of `[w_seed, w_eff, w_unc]` by `(round, seed_diff)` showing when the system trusts which strategic lens |

### Expected Baselines

| Model | Expected Accuracy | Expected Log-Loss |
|-------|------------------|--------------------|
| Seed baseline (always pick higher seed) | ~67% | ~0.60 |
| Single XGBoost on KenPom deltas | ~70-72% | ~0.52-0.55 |
| Three-expert MOE (Tier 1) | ~70-73% | ~0.51-0.54 |
| Three-expert MOE (Tier 2, stretch) | ~71-74% | ~0.50-0.53 |

**If the MOE doesn't beat single-tree**: The portfolio narrative becomes "I implemented the production architecture, backtested rigorously, and determined that ~1,260 tournament games doesn't provide enough data for multi-expert specialization to emerge — here's what I'd need." This is MORE impressive than cherry-picked results because it demonstrates analytical maturity.

---

## Data Sources

| Source | Provides | Years | Access |
|--------|----------|-------|--------|
| Kaggle: Pilafas March Madness Historical Dataset | All KenPom metrics per team | 2002-2025 | Free download |
| Kaggle: March ML Mania 2025 competition data | Tournament results, seeds, team ID mappings | 1985-2025 | Free (join competition) |
| KenPom.com via `kenpompy` | Current season stats | 2026 | $20/year subscription |

---

## Implementation Levels & Timeline

The project builds in four levels, each adding distinct portfolio value. Each level is independently useful — even partial completion tells the portfolio story.

### Level 1 — Baseline (Days 1-2)
**Demonstrates**: Data engineering, tree models, backtesting rigor

- Project scaffolding, `pyproject.toml`, install deps
- Download Kaggle datasets manually
- Build `kaggle_loader.py`, `tournament_data.py`, `team_names.py`, `merge.py`
- Build `kenpom_deltas.py` + `context_features.py`
- Build `tree_expert.py` (single XGBoost, no sample weighting)
- Build `backtester.py` (simple LOYO, no nesting yet), `metrics.py`
- Run LOYO backtesting, establish baseline performance
- **Deliverable**: Single-tree baseline with backtested accuracy ~70-72%, log-loss ~0.52-0.55

### Level 2 — Multi-Expert MOE (Days 3-4)
**Demonstrates**: MOE architecture, expert specialization, multi-task concepts

- Build `ranking_criteria.py` — per-expert ranking target computation
- Build three XGBoost expert variants with different sample weights (Tier 1)
- Build `gating_network.py` — small PyTorch MLP
- Upgrade `backtester.py` to nested LOYO CV
- Build `moe_ensemble.py` — orchestrator
- Run nested LOYO backtesting, compare to Level 1
- Build expert weight visualizations (the portfolio killer feature)
- Build `calibration.py` for per-expert reliability diagrams
- **Deliverable**: Three-expert MOE with nested CV backtesting, expert weight heatmaps

### Level 3 — Full Multi-Task Loss (Day 4-5, stretch)
**Demonstrates**: Custom loss functions, differentiable ranking, PyTorch multi-head architecture

- Build `ranking_loss.py` — `MultiTaskHistoricalRankLoss` with `torchsort`
- Build `multi_task_experts.py` — shared-backbone multi-head PyTorch network
- Backtest Tier 2 vs Tier 1 comparison
- **Deliverable**: PyTorch multi-head experts with differentiable Spearman loss, comparison to XGBoost Tier 1

### Level 4 — Bracket Generation (Days 5-6)
**Demonstrates**: Monte Carlo simulation, uncertainty quantification

- Build `simulation/mc_engine.py`, `simulation/log5.py`
- Build `bracket/structure.py`, `bracket/predictor.py`, `bracket/visualizer.py`
- Build `scraper.py`, scrape 2026 KenPom data
- Wait for Selection Sunday bracket (Mar 15)
- Generate final bracket + per-team advancement probabilities + confidence intervals
- **Deliverable**: Complete 2026 bracket prediction with uncertainty quantification

---

## Dependencies to Install

```bash
pip install xgboost kenpompy torchsort seaborn jupyterlab optuna shap tqdm python-dotenv
```

Already available: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

## Key Design Decisions

1. **Kaggle for history, kenpompy for 2026 only** — Avoids scraping 24 years from KenPom. Saves a full day.
2. **XGBoost over LightGBM** — Interchangeable at this data scale; XGBoost has slightly better regularization controls for small datasets.
3. **Three strategic faces, not two generic experts** — The original 2-expert (tree + MC) architecture reads as a simple ensemble. Three experts with per-expert ranking losses directly mirror the production system's multi-task target architecture.
4. **MC as infrastructure, not peer expert** — MC simulation operates at the bracket level, consuming game-level MOE predictions. This matches the production system where MC consumed MOE scores rather than feeding into the MOE.
5. **Expert specialization via ranking criteria, not data partitioning** — All experts see the same ~1,260 games. Specialization comes from different Spearman correlation targets per expert, mirroring `MultiTaskHistoricalRankLoss` from the production system.
6. **Nested LOYO CV** — Gating network trains on held-out expert predictions to prevent information leakage. Mirrors production practice of training the scoring engine on recent expert performance, not historical training data.
7. **Two-tier expert implementation** — Tier 1 (XGBoost sample-weighted) is practical and buildable in the timeline. Tier 2 (PyTorch `torchsort` Spearman) is the full production-faithful architecture as a stretch goal. Documenting both + backtested comparison is itself portfolio value.
8. **Higher-seed-minus-lower-seed convention** — Removes ordering ambiguity in features. Target = "did higher seed win?"
9. **Skip 2020** — No tournament due to COVID.
10. **Tiny gating network** — Only ~1,260 training games. Complexity = overfitting. The gating network is barely more than a learned lookup table by (seed_diff, round).

---

## Expert Decorrelation Analysis

A concern: Expert 1 (seed-baseline) and Expert 3 (uncertainty-calibration) both use `seed_diff`. But Expert 3 incorporates 60% non-seed signal:

| Feature | Expert 1 Weight | Expert 3 Weight |
|---------|----------------|----------------|
| seed_diff | 100% (sole criterion) | 40% |
| adjEM_delta | 0% | 30% |
| luck_delta | 0% | 20% |
| NCSOS_delta | 0% | 10% |

**Concrete example of differentiation:**
- Game A: 5v12, adjEM_delta=2 (teams are actually close), luck_delta=0.06 (12-seed had lucky season)
  - Expert 1: moderately confident (seed says ~65%)
  - Expert 3: LOW confidence (small efficiency gap + high luck → uncertain game)
- Game B: 5v12, adjEM_delta=15 (5-seed is clearly superior), luck_delta=0.01
  - Expert 1: same confidence (same seed matchup)
  - Expert 3: HIGH confidence (efficiency gap confirms seed signal)

Expert 1 cannot distinguish these games. Expert 3 can. The 60% non-seed signal creates genuine specialization.

---

## Verification Plan

1. **Data validation**: Spot-check merged data against known results (e.g., 2023 UConn championship run, 2018 UMBC over Virginia)
2. **Single-tree sanity**: Verify accuracy beats naive seed baseline (~67%). Check feature importance makes sense (AdjEM delta should dominate).
3. **Expert specialization check**: Verify the three experts produce different probability rankings for the same games. Compute pairwise Spearman correlation between expert outputs — should be positive but < 0.9.
4. **Gating network sanity**: Verify expert weights vary by context. If gating assigns ~uniform weights (0.33, 0.33, 0.33) everywhere, the MOE hasn't learned anything.
5. **Nested CV integrity**: Verify no data leakage — gating network's training data predictions must come from experts that did NOT train on that data.
6. **MOE backtest**: Compare MOE log-loss vs single-tree across all 22 held-out years.
7. **Calibration**: Reliability diagrams per expert + blended. Predicted 70% should win ~70% of the time.
8. **MC simulation check**: Verify 1-seeds advance past R64 at ~99% rate, 5v12 upsets at ~35% rate — matches historical frequencies.
9. **End-to-end**: Run `predict_bracket.py` on a historical year (e.g., 2023) and verify output format and sanity of picks.

---

## Falsification Triggers

The architecture thesis breaks if:
1. **`torchsort` Spearman can't train on binary classification outputs** — fall back to Tier 1 (XGBoost sample-weighted). The portfolio still demonstrates the concept.
2. **All three experts converge to identical probability rankings** — would mean the ranking criteria are too correlated or the data is too small for specialization. Document why and what data scale would be needed.
3. **Gating network assigns uniform weights everywhere** — would mean the context features don't provide enough signal for expert selection. Analyze which context features correlate most with expert accuracy to understand what's missing.
