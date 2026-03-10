# March Madness MOE

A Mixture of Experts prediction system for the NCAA March Madness tournament. Three specialized XGBoost experts — each trained on different feature subsets with distinct sample weighting strategies — are dynamically blended by a PyTorch gating network that routes predictions based on game context (seed differential, round, efficiency gap, luck differential).

## Architecture

```
                         Game Features (39 columns)
                                   |
                    +--------------+--------------+
                    |              |              |
              Seed Baseline   Efficiency    Uncertainty
              (5 features)    (27 features)  (13 features)
              uniform wt.     |AdjEM| wt.   1/|seed_diff| wt.
                    |              |              |
                    +------+-------+------+------+
                           |              |
                     Gating Network    Expert Preds
                     (PyTorch MLP)     (3 probabilities)
                           |              |
                           +----- * ------+
                                  |
                          Blended Prediction
                                  |
                       Isotonic Calibration
                                  |
                         P(higher seed wins)
```

**Experts** see non-overlapping slices of the feature space, forcing genuine specialization:

| Expert | Theory | Features | Sample Weighting |
|--------|--------|----------|-----------------|
| Seed Baseline | Seed matchup history drives outcomes | Seed metrics, round, conference, historical upset rates | Uniform |
| Efficiency Delta | KenPom efficiency gaps reveal hidden value | Offensive/defensive four-factors, net rating, tempo | Proportional to \|net_rating_delta\| |
| Uncertainty Calibration | Luck, schedule strength, and roster volatility create uncertainty | Luck, SOS, experience, height, bench depth | Inverse seed diff (close games weighted higher) |

**Gating Network** learns context-dependent routing:
- One-hot encoded round (6 dims) for round-specific expert weighting
- Seed diff, AdjEM delta, luck delta as continuous context signals
- Min-weight floor (0.15) + entropy bonus prevent collapse to a single expert
- Post-hoc isotonic calibration corrects blend-induced miscalibration

## Results

Evaluated via nested Leave-One-Year-Out cross-validation (2003-2025, 22 test folds):

| Metric | MOE | Single XGBoost Baseline |
|--------|-----|------------------------|
| ESPN Bracket Score | **33,280** | 33,240 |
| Accuracy | 82.5% | 83.0% |
| Log-Loss | 0.446 | 0.405 |
| Expert Diversity (mean Spearman) | 0.57 | N/A |

The MOE wins on bracket scoring (what matters for pools) while maintaining competitive accuracy. Expert predictions show meaningful diversity (Spearman correlations 0.41-0.84 across pairs), confirming the feature subsetting strategy works.

## Quick Start

### Prerequisites

- Python >= 3.10
- [KenPom](https://kenpom.com/) subscription (for live data scraping)
- Kaggle datasets: [Pilafas KenPom](https://www.kaggle.com/datasets/andrewpilafas/kenpom-data), [March ML Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)

### Installation

```bash
git clone https://github.com/AustinSalter/march-madness-moe.git
cd march-madness-moe
pip install -e ".[dev]"
```

### Data Setup

1. Download KenPom CSVs from Kaggle into `data/raw/kenpom/`
2. Download tournament CSVs into `data/raw/tournament/`
3. (Optional) For live KenPom data, create `.env`:
   ```
   KENPOM_EMAIL=your_email@example.com
   KENPOM_PASSWORD=your_password
   ```
   Then run: `python scripts/scrape_kenpom.py`

### Generate a Bracket

```bash
# Train MOE and predict the 2025 bracket
python scripts/predict_bracket.py --season 2025

# With Monte Carlo simulation (10k tournament simulations)
python scripts/predict_bracket.py --season 2025 --n-sims 10000
```

### Run the Full Backtest

```bash
python scripts/run_backtest.py
```

This runs the nested LOYO protocol (~1,400 XGBoost fits, ~15 min) and compares MOE vs. baseline.

## Project Structure

```
march-madness-moe/
├── src/
│   ├── config.py                 # Hyperparameters, paths, feature subsets
│   ├── data/                     # Loading & merging (KenPom + Kaggle)
│   ├── features/                 # Delta features, context features, ranking targets
│   ├── models/
│   │   ├── tree_expert.py        # XGBoost expert with sample weighting + calibration
│   │   ├── gating_network.py     # PyTorch MLP gating with entropy bonus
│   │   └── moe_ensemble.py       # MOE orchestrator (nested CV, blend, calibration)
│   ├── evaluation/               # Metrics, calibration, nested LOYO backtester
│   ├── simulation/               # Monte Carlo bracket simulation, Log5
│   └── bracket/                  # Bracket structure, predictor, visualizer
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_single_tree_baseline.ipynb
│   ├── 04_multi_expert_dev.ipynb
│   ├── 05_gating_network_dev.ipynb
│   ├── 06_backtesting_analysis.ipynb
│   ├── 07_mc_simulation_dev.ipynb
│   └── 08_bracket_output.ipynb
├── scripts/
│   ├── scrape_kenpom.py          # Fetch live KenPom ratings
│   ├── build_features.py         # Run feature pipeline
│   ├── train_baseline.py         # Train single XGBoost
│   ├── train_experts.py          # Train full MOE system
│   ├── run_backtest.py           # Nested LOYO evaluation
│   └── predict_bracket.py        # Generate bracket predictions
└── pyproject.toml
```

## Notebooks

The notebooks follow the development progression and serve as both documentation and reproducible analysis:

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Exploration | Raw data inspection, distributions, missing values |
| 02 | Feature Engineering | Build 39 delta features, verify ranking targets |
| 03 | Single Tree Baseline | XGBoost baseline, feature importance, overfit checks |
| 04 | Multi-Expert Dev | 3 expert archetypes, sample weights, per-expert evaluation |
| 05 | Gating Network Dev | Architecture search, LR sweep, weight routing analysis |
| 06 | Backtesting Analysis | Full nested LOYO, MOE vs. baseline, per-year breakdown |
| 07 | MC Simulation | Tournament paths, champion probabilities, confidence intervals |
| 08 | Bracket Output | Final bracket generation and visualization |

## Key Design Decisions

**Why Mixture of Experts?** A single model can't simultaneously optimize for seed-driven chalk outcomes and efficiency-driven upset detection. MOE lets each expert specialize, then routes based on context — seed expert for R1 chalk, efficiency expert when talent gaps are large, uncertainty expert for volatile matchups.

**Why feature subsetting?** Early experiments showed all 3 experts converging to nearly identical predictions (Spearman > 0.95) despite different sample weights. Restricting each expert to its own feature slice forces genuine specialization and drops correlations to 0.41-0.84.

**Why nested LOYO?** The gating network needs held-out expert predictions to avoid overfitting. The inner loop generates these via leave-one-year-out within the training pool. The outer loop ensures no test-year leakage.

**Why isotonic calibration?** Blending well-calibrated experts with a poorly-calibrated one (via forced minimum weights) degrades the blend's calibration. A lightweight isotonic layer on the final output corrects this without changing the gating's routing behavior.

## Tech Stack

- **XGBoost** — Gradient-boosted tree experts
- **PyTorch** — Gating network MLP
- **scikit-learn** — Isotonic calibration, stratified splits
- **Optuna** — Hyperparameter tuning (50 trials per expert)
- **kenpompy** — Live KenPom data scraping
- **pandas / numpy** — Feature engineering
- **matplotlib / seaborn / SHAP** — Visualization and interpretability

## License

MIT
