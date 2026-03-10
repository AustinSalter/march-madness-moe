# Model Layer Implementation

## Implemented Files

### Core Models (Tier 1)

1. **`src/config.py`** — Updated `XGBOOST_PARAMS` with aggressive regularization + added `EARLY_STOPPING_ROUNDS`, `EXPERT_TYPES`, sample weight constants, `OPTUNA_*` settings, `CALIBRATION_N_BINS`
2. **`src/models/tree_expert.py`** (~240 lines) — `TreeExpert` class with per-type sample weighting, early stopping, isotonic calibration, feature importance, save/load, plus `tune_expert_hyperparams()` Optuna function
3. **`src/models/gating_network.py`** (~190 lines) — `GatingMLP` (Linear→ReLU→Dropout→Linear→ReLU→Linear→Softmax) + `GatingNetwork` training wrapper with BCE(blended, y) loss, early stopping, save/load
4. **`src/models/moe_ensemble.py`** (~300 lines) — `MOEEnsemble` orchestrator: trains 3 experts, gating network, `generate_inner_cv_predictions()` for nested LOYO, `predict_decomposed()` for full weight decomposition
5. **`src/evaluation/metrics.py`** (~120 lines) — logloss, accuracy, Brier score, ESPN bracket score, round breakdown, expert agreement (pairwise Spearman)
6. **`src/evaluation/calibration.py`** (~90 lines) — ECE, calibration curves, per-expert calibration, calibration report
7. **`src/evaluation/backtester.py`** (~180 lines) — `BacktestResult` dataclass + `NestedLOYOBacktester` with `run()` (full nested LOYO) and `run_baseline()` (simple LOYO comparison)

### Scripts

8. **`scripts/train_baseline.py`** — Train single XGBoost, evaluate, save
9. **`scripts/train_experts.py`** — Full MOE nested training + diagnostics
10. **`scripts/run_backtest.py`** — Run both baseline and MOE backtests, compare, save results

### Tier 2 Stretch

11. **`src/models/ranking_loss.py`** (~65 lines) — BCE + λ(1 - SpearmanR) with torchsort soft_rank
12. **`src/models/multi_task_experts.py`** (~200 lines) — SharedBackbone + 3 ExpertHeads with MultiTaskRankingLoss, drop-in replacement for TreeExperts

---

## Architecture

### Mixture of Experts (MOE)

```
Input Features (~39)
        │
        ├──► TreeExpert 1 (seed_baseline)      ──► P1
        ├──► TreeExpert 2 (efficiency_delta)    ──► P2
        ├──► TreeExpert 3 (uncertainty_calibration) ──► P3
        │
Gating Features (4) ──► GatingMLP ──► [w1, w2, w3]
        │
        └──► Blended P = w1·P1 + w2·P2 + w3·P3
```

### Expert Sample Weighting

| Expert | Weighting Strategy |
|---|---|
| `seed_baseline` | Uniform weights |
| `efficiency_delta` | `\|adjem_delta\|` with floor 0.1, normalized |
| `uncertainty_calibration` | `1 / (\|seed_diff\| + 1)`, same-seed gets 5.0, normalized |

### Gating Network

- **Input**: 4 context features — `seed_diff`, `round`, `adjem_delta`, `luck_delta`
- **Architecture**: Linear(4→16) → ReLU → Dropout(0.2) → Linear(16→8) → ReLU → Dropout(0.2) → Linear(8→3) → Softmax
- **Loss**: BCE(blended_prob, y) where blended_prob = Σ(weights × expert_preds)
- Expert predictions are fixed inputs — no gradient flows through them

### Nested LOYO Backtesting

```
Outer loop: for each test_year in 2003-2025 (skip 2020):
    train_seasons = all seasons - {test_year}

    Inner loop: for each val_year in train_seasons:
        inner_train = train_seasons - {val_year}
        Train 3 experts on inner_train
        Predict val_year → accumulate (context, expert_preds, y)

    Train gating on accumulated inner-loop predictions
    Train final experts on full train_seasons
    Predict test_year → record decomposition
```

- 22 outer folds × 21 inner folds × 3 experts = **1,386 XGBoost fits** (~10-20 min)

---

## XGBoost Configuration

```python
XGBOOST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 50,
    "gamma": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
}
EARLY_STOPPING_ROUNDS = 50
```

**Rationale**: Level-wise growth is safer for ~1,260 samples. Aggressive regularization (`reg_lambda=50`, `gamma=1`, `min_child_weight=10`) prevents overfitting on small data.

---

## Optuna Strategy

**Pre-tune, then backtest** (avoids 22×21×50 = 23,100 trial explosion):

1. Hold out most recent season as validation
2. Train on remaining seasons (skip 2020)
3. Run 50 trials per expert type (150 total, ~10min)
4. Save best params per expert type
5. Run full nested backtest with fixed tuned params

---

## Calibration Strategy

- XGBoost `binary:logistic` produces reasonably calibrated probabilities
- Post-hoc isotonic regression via `CalibratedClassifierCV(cv=5)` on held-out data
- Isotonic preferred over Platt: XGBoost under sample weighting can have non-sigmoid miscalibration
- Blended MOE output gets a final isotonic pass using inner-CV predictions
- Verified with reliability diagrams (ECE metric) in backtesting results

---

## Verification Checklist

- [ ] **Smoke test**: Train MOE on 3 seasons, predict 1 season — verify outputs
- [ ] **Expert decorrelation**: Pairwise Spearman between experts < 0.9
- [ ] **Gating non-uniformity**: std(weights) across games > 0 meaningfully
- [ ] **Sanity checks**: 1v16 → P > 0.90, 8v9 → P ≈ 0.50-0.55
- [ ] **Calibration**: Reliability diagrams show predicted 70% wins ~70%
- [ ] **Overfitting**: train vs test log-loss gap < 0.10 per fold
- [ ] **Baseline comparison**: MOE log-loss < single-tree log-loss
- [ ] **Full backtest**: 22-fold nested LOYO produces ~1,260 held-out predictions

---

## Usage

```bash
# Install dependencies
pip install -e .

# Train baseline
python scripts/train_baseline.py

# Train full MOE
python scripts/train_experts.py

# Run full backtest (baseline + MOE comparison)
python scripts/run_backtest.py
```
