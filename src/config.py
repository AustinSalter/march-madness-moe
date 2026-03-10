"""Paths, hyperparameters, and constants for the March Madness MOE project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KENPOM_DIR = RAW_DIR / "kenpom"
TOURNAMENT_DIR = RAW_DIR / "tournament"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
TARGETS_DIR = PROCESSED_DIR / "targets"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Tournament constants ───────────────────────────────────────────────────
FIRST_YEAR = 2003  # First year with KenPom data + tournament results
LAST_YEAR = 2025
SKIP_YEARS = {2020}  # COVID — no tournament

# ── XGBoost expert defaults ───────────────────────────────────────────────
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

# ── Expert sample weighting ──────────────────────────────────────────────
EXPERT_TYPES = ["seed_baseline", "efficiency_delta", "uncertainty_calibration"]
EFFICIENCY_WEIGHT_FLOOR = 0.1
UNCERTAINTY_SAME_SEED_WEIGHT = 5.0

# ── Optuna hyperparameter tuning ─────────────────────────────────────────
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 600
OPTUNA_SEARCH_SPACE = {
    "max_depth": (3, 7),
    "learning_rate": (0.01, 0.15),
    "min_child_weight": (5, 30),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (10, 100),
    "gamma": (0, 5),
}

# ── Calibration ──────────────────────────────────────────────────────────
CALIBRATION_N_BINS = 10

# ── Gating network defaults ──────────────────────────────────────────────
GATING_HIDDEN_SIZES = [16, 8]
GATING_DROPOUT = 0.2
GATING_LR = 1e-3
GATING_EPOCHS = 200
GATING_BATCH_SIZE = 64

# ── Multi-task expert defaults (Tier 2 / stretch) ────────────────────────
BACKBONE_HIDDEN_SIZES = [32, 24]
BACKBONE_DROPOUT = 0.3
RANKING_LOSS_LAMBDA = 0.3

# ── Monte Carlo simulation ───────────────────────────────────────────────
MC_SIMULATIONS = 10_000
MC_RANDOM_SEED = 42

# ── Bracket scoring (ESPN standard) ──────────────────────────────────────
ESPN_ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
