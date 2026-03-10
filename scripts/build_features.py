"""CLI: Run the full feature engineering pipeline."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from src.features.pipeline import build_features, save_features

    logger.info("Building features...")
    fs = build_features()

    logger.info("Feature matrix: %s", fs.X.shape)
    logger.info("Feature columns: %s", fs.feature_names)
    logger.info("Gating features: %s", fs.gating_features)
    logger.info("Target distribution: %s", fs.y.value_counts().to_dict())

    # Sanity checks
    score_cols = [c for c in fs.X.columns if "score" in c.lower()]
    if score_cols:
        logger.error("LEAKAGE: score columns found in X: %s", score_cols)
        sys.exit(1)

    ab_cols = [c for c in fs.X.columns if c.endswith("_a") or c.endswith("_b")]
    if ab_cols:
        logger.error("Raw _a/_b columns found in X: %s", ab_cols)
        sys.exit(1)

    logger.info("Saving features...")
    save_features(fs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
