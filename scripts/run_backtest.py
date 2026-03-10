"""CLI: Run full nested LOYO backtesting."""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR
from src.data.merge import merge_kenpom_with_matchups
from src.evaluation.backtester import NestedLOYOBacktester
from src.evaluation.metrics import compute_round_breakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = MODELS_DIR / "backtest_results"


def main():
    """Run full nested LOYO backtest and baseline comparison."""
    logger.info("Loading data...")
    merged_df = merge_kenpom_with_matchups()
    logger.info("Loaded %d games", len(merged_df))

    backtester = NestedLOYOBacktester()

    # Run baseline first
    logger.info("=" * 60)
    logger.info("Running baseline backtest...")
    logger.info("=" * 60)
    baseline_result = backtester.run_baseline(merged_df=merged_df)
    logger.info("\n%s", baseline_result.summary())

    # Run full MOE backtest
    logger.info("=" * 60)
    logger.info("Running full MOE backtest...")
    logger.info("=" * 60)
    moe_result = backtester.run(merged_df=merged_df)
    logger.info("\n%s", moe_result.summary())

    # Comparison
    logger.info("=" * 60)
    logger.info("=== Baseline vs MOE Comparison ===")
    logger.info("=" * 60)
    for metric in ["logloss", "accuracy", "brier_score"]:
        b = baseline_result.metrics_overall[metric]
        m = moe_result.metrics_overall[metric]
        better = "MOE" if (m < b if metric != "accuracy" else m > b) else "Baseline"
        logger.info("  %s: Baseline=%.4f, MOE=%.4f (%s wins)", metric, b, m, better)

    # Round breakdown for MOE
    if "round" in moe_result.predictions.columns:
        rounds = moe_result.predictions["round"].values
        rd_breakdown = compute_round_breakdown(
            moe_result.predictions["y_true"].values,
            moe_result.predictions["p_blend"].values,
            rounds,
        )
        logger.info("\n=== MOE Round Breakdown ===")
        logger.info("\n%s", rd_breakdown.to_string(index=False))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    moe_result.predictions.to_parquet(RESULTS_DIR / "moe_predictions.parquet")
    moe_result.metrics_by_year.to_csv(RESULTS_DIR / "moe_metrics_by_year.csv", index=False)
    baseline_result.predictions.to_parquet(RESULTS_DIR / "baseline_predictions.parquet")
    baseline_result.metrics_by_year.to_csv(RESULTS_DIR / "baseline_metrics_by_year.csv", index=False)
    logger.info("Saved backtest results to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
