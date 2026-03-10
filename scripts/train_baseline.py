"""CLI: Train single-tree XGBoost baseline."""

import logging
import sys

from src.config import FIRST_YEAR, LAST_YEAR, MODELS_DIR, SKIP_YEARS
from src.data.merge import merge_kenpom_with_matchups
from src.evaluation.metrics import compute_all_metrics
from src.features.pipeline import build_features_for_split
from src.models.tree_expert import TreeExpert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train a single XGBoost baseline and evaluate."""
    logger.info("Loading data...")
    merged_df = merge_kenpom_with_matchups()
    logger.info("Loaded %d games", len(merged_df))

    # Hold out most recent season for evaluation
    all_seasons = [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in SKIP_YEARS]
    val_season = all_seasons[-1]
    train_seasons = all_seasons[:-1]

    logger.info("Train seasons: %d-%d, Val season: %d", train_seasons[0], train_seasons[-1], val_season)

    train_fs, val_fs = build_features_for_split(train_seasons, [val_season], merged_df=merged_df)
    logger.info("Train: %d games, Val: %d games", len(train_fs.y), len(val_fs.y))

    # Train baseline
    expert = TreeExpert(expert_type="seed_baseline")
    expert.fit(train_fs, val_fs=val_fs)

    # Evaluate
    train_preds = expert.predict_proba(train_fs.X)
    val_preds = expert.predict_proba(val_fs.X)

    train_metrics = compute_all_metrics(train_fs.y.values, train_preds)
    val_metrics = compute_all_metrics(
        val_fs.y.values, val_preds,
        rounds=val_fs.meta["round"].values if "round" in val_fs.meta.columns else None,
    )

    logger.info("=== Train Metrics ===")
    for k, v in train_metrics.items():
        logger.info("  %s: %.4f", k, v)

    logger.info("=== Val Metrics (%d) ===", val_season)
    for k, v in val_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # Overfit check
    gap = val_metrics["logloss"] - train_metrics["logloss"]
    logger.info("Train-val logloss gap: %.4f", gap)

    # Feature importance
    importance = expert.get_feature_importance()
    logger.info("=== Top 10 Features (gain) ===")
    for feat, gain in importance.head(10).items():
        logger.info("  %s: %.2f", feat, gain)

    # Save
    save_path = MODELS_DIR / "baseline_expert.pkl"
    expert.save(save_path)
    logger.info("Saved baseline to %s", save_path)


if __name__ == "__main__":
    main()
