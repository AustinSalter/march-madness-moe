"""Nested Leave-One-Year-Out cross-validation for experts + gating."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import EXPERT_TYPES, FIRST_YEAR, LAST_YEAR, SKIP_YEARS
from src.data.merge import merge_kenpom_with_matchups
from src.evaluation.calibration import compute_calibration_report
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_expert_agreement,
    compute_round_breakdown,
)
from src.features.pipeline import FeatureSet, build_features_for_split
from src.models.moe_ensemble import MOEEnsemble
from src.models.tree_expert import TreeExpert

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    predictions: pd.DataFrame  # Per-game predictions across all folds
    metrics_by_year: pd.DataFrame  # Per-year metrics
    metrics_overall: dict  # Aggregate metrics across all folds
    expert_agreement: pd.DataFrame  # Pairwise expert Spearman correlations
    calibration: dict  # Calibration report

    def summary(self) -> str:
        """Pretty-print summary of backtest results."""
        lines = ["=== Backtest Summary ==="]
        for k, v in self.metrics_overall.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append(f"\n  Total predictions: {len(self.predictions)}")
        lines.append(f"  Years tested: {self.metrics_by_year['season'].nunique()}")
        lines.append(f"\n  Expert agreement (Spearman):")
        lines.append(self.expert_agreement.to_string())
        lines.append(f"\n  Blend ECE: {self.calibration.get('blend_ece', 'N/A'):.4f}")
        return "\n".join(lines)


def _get_all_seasons() -> list[int]:
    """Return list of valid seasons."""
    return [y for y in range(FIRST_YEAR, LAST_YEAR + 1) if y not in SKIP_YEARS]


class NestedLOYOBacktester:
    """Nested Leave-One-Year-Out backtester for the MOE system.

    Outer loop: leave one year out for testing.
    Inner loop (within MOEEnsemble): leave one year out of training for gating.
    """

    def __init__(self, expert_params: dict | None = None):
        self.expert_params = expert_params or {}

    def run(self, merged_df: pd.DataFrame | None = None) -> BacktestResult:
        """Run full nested LOYO backtest.

        Args:
            merged_df: Full merged DataFrame. Loaded if None.

        Returns:
            BacktestResult with predictions, metrics, and diagnostics.
        """
        if merged_df is None:
            merged_df = merge_kenpom_with_matchups()

        seasons = _get_all_seasons()
        all_fold_dfs = []

        for test_year in seasons:
            train_seasons = [s for s in seasons if s != test_year]
            logger.info(
                "Outer fold: test_year=%d, train_years=%d seasons",
                test_year, len(train_seasons),
            )

            train_fs, test_fs = build_features_for_split(
                train_seasons, [test_year], merged_df=merged_df,
            )

            if len(test_fs.y) == 0:
                logger.warning("No test data for %d — skipping", test_year)
                continue

            moe = MOEEnsemble(expert_params=self.expert_params)

            # Inner CV: generate held-out expert preds for gating training
            context_X, expert_preds_inner, y_inner = moe.generate_inner_cv_predictions(
                train_seasons, merged_df,
            )

            # Train gating on inner-loop predictions
            moe.train_gating(context_X, expert_preds_inner, y_inner)

            # Train final experts on full training pool
            moe.train_experts(train_fs)

            # Predict + record decomposition
            decomposed = moe.predict_decomposed(test_fs)
            fold_df = test_fs.meta.copy()
            fold_df = fold_df.join(decomposed)
            fold_df["y_true"] = test_fs.y.values
            fold_df["round"] = test_fs.meta["round"].values if "round" in test_fs.meta.columns else np.nan
            all_fold_dfs.append(fold_df)

            # Log fold metrics
            fold_metrics = compute_all_metrics(
                test_fs.y.values, decomposed["p_blend"].values,
            )
            logger.info(
                "Fold %d: logloss=%.4f, accuracy=%.4f, n=%d",
                test_year, fold_metrics["logloss"], fold_metrics["accuracy"],
                len(test_fs.y),
            )

        return self._aggregate_results(all_fold_dfs)

    def run_baseline(self, merged_df: pd.DataFrame | None = None) -> BacktestResult:
        """Run simple LOYO with a single XGBoost for comparison.

        Args:
            merged_df: Full merged DataFrame. Loaded if None.

        Returns:
            BacktestResult for baseline comparison.
        """
        if merged_df is None:
            merged_df = merge_kenpom_with_matchups()

        seasons = _get_all_seasons()
        all_fold_dfs = []

        for test_year in seasons:
            train_seasons = [s for s in seasons if s != test_year]
            logger.info("Baseline fold: test_year=%d", test_year)

            train_fs, test_fs = build_features_for_split(
                train_seasons, [test_year], merged_df=merged_df,
            )

            if len(test_fs.y) == 0:
                logger.warning("No test data for %d — skipping", test_year)
                continue

            expert = TreeExpert(expert_type="seed_baseline")
            expert.fit(train_fs)
            preds = expert.predict_proba(test_fs.X)

            fold_df = test_fs.meta.copy()
            fold_df["p_blend"] = preds
            # For baseline, all experts produce same prediction
            fold_df["p_seed"] = preds
            fold_df["p_eff"] = preds
            fold_df["p_unc"] = preds
            fold_df["w_seed"] = 1.0 / 3
            fold_df["w_eff"] = 1.0 / 3
            fold_df["w_unc"] = 1.0 / 3
            fold_df["y_true"] = test_fs.y.values
            all_fold_dfs.append(fold_df)

        return self._aggregate_results(all_fold_dfs)

    def _aggregate_results(self, all_fold_dfs: list[pd.DataFrame]) -> BacktestResult:
        """Aggregate fold-level results into BacktestResult.

        Args:
            all_fold_dfs: List of per-fold DataFrames.

        Returns:
            BacktestResult with all diagnostics.
        """
        predictions = pd.concat(all_fold_dfs, ignore_index=True)

        y_true = predictions["y_true"].values
        y_pred = predictions["p_blend"].values
        rounds = predictions["round"].values if "round" in predictions.columns else None

        # Overall metrics
        metrics_overall = compute_all_metrics(y_true, y_pred, rounds)

        # Per-year metrics
        year_rows = []
        for season, group in predictions.groupby("season"):
            yt = group["y_true"].values
            yp = group["p_blend"].values
            rd = group["round"].values if "round" in group.columns else None
            m = compute_all_metrics(yt, yp, rd)
            m["season"] = season
            m["n_games"] = len(group)
            year_rows.append(m)
        metrics_by_year = pd.DataFrame(year_rows)

        # Expert agreement
        expert_preds = predictions[["p_seed", "p_eff", "p_unc"]].values
        expert_agreement = compute_expert_agreement(expert_preds)
        expert_agreement.index = EXPERT_TYPES
        expert_agreement.columns = EXPERT_TYPES

        # Calibration
        calibration = compute_calibration_report(
            y_true, y_pred,
            expert_preds=expert_preds,
            expert_names=EXPERT_TYPES,
        )

        return BacktestResult(
            predictions=predictions,
            metrics_by_year=metrics_by_year,
            metrics_overall=metrics_overall,
            expert_agreement=expert_agreement,
            calibration=calibration,
        )
