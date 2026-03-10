"""CLI: Train multi-expert MOE system."""

import logging
import sys

from src.config import FIRST_YEAR, LAST_YEAR, MODELS_DIR, SKIP_YEARS
from src.data.merge import merge_kenpom_with_matchups
from src.evaluation.metrics import compute_all_metrics
from src.features.pipeline import build_features_for_split
from src.models.moe_ensemble import MOEEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train full MOE system with nested CV for gating."""
    logger.info("Loading data...")
    merged_df = merge_kenpom_with_matchups()
    logger.info("Loaded %d games", len(merged_df))

    # Hold out most recent season for final evaluation
    all_seasons = [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in SKIP_YEARS]
    val_season = all_seasons[-1]
    train_seasons = all_seasons[:-1]

    logger.info("Train seasons: %d-%d, Val season: %d", train_seasons[0], train_seasons[-1], val_season)

    train_fs, val_fs = build_features_for_split(train_seasons, [val_season], merged_df=merged_df)
    logger.info("Train: %d games, Val: %d games", len(train_fs.y), len(val_fs.y))

    # Train MOE with full nested protocol
    moe = MOEEnsemble()
    moe.train_full_nested(train_fs, train_seasons, merged_df)

    # Evaluate on held-out season
    decomposed = moe.predict_decomposed(val_fs)
    val_metrics = compute_all_metrics(
        val_fs.y.values, decomposed["p_blend"].values,
        rounds=val_fs.meta["round"].values if "round" in val_fs.meta.columns else None,
    )

    logger.info("=== Val Metrics (%d) ===", val_season)
    for k, v in val_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # Expert weight distribution
    logger.info("=== Expert Weight Distribution ===")
    for col in ["w_seed", "w_eff", "w_unc"]:
        logger.info(
            "  %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            col, decomposed[col].mean(), decomposed[col].std(),
            decomposed[col].min(), decomposed[col].max(),
        )

    # Expert prediction correlations
    from src.evaluation.metrics import compute_expert_agreement
    expert_preds = decomposed[["p_seed", "p_eff", "p_unc"]].values
    agreement = compute_expert_agreement(expert_preds)
    logger.info("=== Expert Agreement (Spearman) ===")
    logger.info("\n%s", agreement.to_string())

    # Save
    save_dir = MODELS_DIR / "moe_ensemble"
    moe.save(save_dir)
    logger.info("Saved MOE ensemble to %s", save_dir)


if __name__ == "__main__":
    main()
