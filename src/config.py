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

EXPERT_FEATURE_SUBSETS: dict[str, list[str] | None] = {
    "seed_baseline": [
        "seed_diff", "hist_upset_rate", "round", "same_conf",
        "adjem_delta", "net_rating_delta", "expected_margin",
        "higher_seed", "seed_sum", "log_seed_ratio", "chalk_round",
        "seed_implied_prob",           # ranking target as feature
    ],
    "efficiency_delta": [
        "adjem_delta", "adjoe_delta", "adjde_delta", "adjtempo_delta",
        "net_rating_delta", "expected_margin",
        "off_efgpct_delta", "off_topct_delta", "off_orpct_delta", "off_ftrate_delta",
        "off_fg2pct_delta", "off_fg3pct_delta", "off_ftpct_delta",
        "off_blockpct_delta", "off_arate_delta", "off_fg3rate_delta",
        "off_stlrate_delta", "off_nstrate_delta",
        "def_fg2pct_delta", "def_fg3pct_delta", "def_ftpct_delta",
        "def_blockpct_delta", "def_arate_delta", "def_fg3rate_delta",
        "def_stlrate_delta", "def_nstrate_delta",
        "efficiency_delta_rank",       # Change 4: ranking target as feature
    ],
    "uncertainty_calibration": [
        "luck_delta", "sos_adjem_delta", "sos_opp_o_delta", "sos_opp_d_delta",
        "ncsos_adjem_delta", "experience_delta", "avg_height_delta",
        "eff_height_delta", "bench_delta", "seed_diff", "round", "adjtempo_delta",
        "game_certainty_score",        # Change 4: ranking target as feature
    ],
}

EXPERT_RANKING_TARGET: dict[str, str] = {
    "seed_baseline": "seed_implied_prob",
    "efficiency_delta": "efficiency_delta_rank",
    "uncertainty_calibration": "game_certainty_score",
}

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
GATING_MIN_WEIGHT = 0.15  # Floor per expert — prevents gating collapse
GATING_ENTROPY_LAMBDA = 0.30  # Reward for spreading weight across experts

# ── Multi-task expert defaults (Tier 2 / stretch) ────────────────────────
BACKBONE_HIDDEN_SIZES = [32, 24]
BACKBONE_DROPOUT = 0.3
RANKING_LOSS_LAMBDA = 0.3

# ── Monte Carlo simulation ───────────────────────────────────────────────
MC_SIMULATIONS = 10_000
MC_RANDOM_SEED = 42

# ── Bracket scoring (ESPN standard) ──────────────────────────────────────
ESPN_ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
