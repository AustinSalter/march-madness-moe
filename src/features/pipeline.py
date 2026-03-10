"""Orchestrate the full feature engineering pipeline."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.merge import merge_kenpom_with_matchups
from src.features.kenpom_deltas import compute_deltas, detect_stat_pairs
from src.features.context_features import add_context_features, build_upset_rate_lookup
from src.features.ranking_criteria import compute_ranking_targets

logger = logging.getLogger(__name__)

# Columns that belong in meta (not features)
_META_COLS = [
    "season", "round", "region",
    "seed_a", "seed_b",
    "team_id_a", "team_id_b",
    "team_name_a", "team_name_b",
    "score_a", "score_b",
]

# Ranking target column names
_RANKING_TARGET_COLS = [
    "seed_implied_prob",
    "efficiency_delta_rank",
    "game_certainty_score",
]

# Gating network uses a small subset of features
_GATING_FEATURES = ["seed_diff", "round", "adjem_delta", "luck_delta"]


@dataclass
class FeatureSet:
    """Partitioned feature engineering output."""
    X: pd.DataFrame                         # ~39 feature columns
    y: pd.Series                            # higher_seed_won (binary)
    meta: pd.DataFrame                      # identifiers and outcome info
    ranking_targets: pd.DataFrame           # 3 expert Spearman targets
    feature_names: list[str] = field(default_factory=list)
    gating_features: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
        if not self.gating_features:
            self.gating_features = [f for f in _GATING_FEATURES if f in self.X.columns]


def build_features(
    merged_df: pd.DataFrame | None = None,
    upset_rate_lookup: dict | None = None,
) -> FeatureSet:
    """Full pipeline: load merged → deltas → context → ranking targets → partition.

    Args:
        merged_df: Output of merge_kenpom_with_matchups(). Loaded if None.
        upset_rate_lookup: From build_upset_rate_lookup(). If None, built from
            the full dataset (only safe for non-split usage).

    Returns:
        FeatureSet with X, y, meta, ranking_targets partitioned.
    """
    if merged_df is None:
        merged_df = merge_kenpom_with_matchups()

    logger.info("Building features from %d games", len(merged_df))

    # Step 1: Compute deltas
    df = compute_deltas(merged_df)

    # Step 2: Build upset rate lookup if not provided
    if upset_rate_lookup is None:
        upset_rate_lookup = build_upset_rate_lookup(df)

    # Step 3: Add context features
    df = add_context_features(df, upset_rate_lookup=upset_rate_lookup)

    # Step 4: Add ranking targets
    df = compute_ranking_targets(df, upset_rate_lookup=upset_rate_lookup)

    # Step 5: Partition into FeatureSet
    return _partition(df)


def build_features_for_split(
    train_seasons: list[int],
    test_seasons: list[int],
    merged_df: pd.DataFrame | None = None,
) -> tuple[FeatureSet, FeatureSet]:
    """Leakage-safe: compute upset lookup from train, apply to both sets.

    Args:
        train_seasons: Seasons for training data.
        test_seasons: Seasons for test data.
        merged_df: Output of merge_kenpom_with_matchups(). Loaded if None.

    Returns:
        (train_features, test_features) tuple of FeatureSets.
    """
    if merged_df is None:
        merged_df = merge_kenpom_with_matchups()

    train_df = merged_df[merged_df["season"].isin(train_seasons)].copy()
    test_df = merged_df[merged_df["season"].isin(test_seasons)].copy()

    logger.info(
        "Building features for split: %d train games (%d seasons), %d test games (%d seasons)",
        len(train_df), len(train_seasons), len(test_df), len(test_seasons),
    )

    # Build upset rate lookup from TRAINING data only
    # Need deltas first for the lookup to work with ranking criteria later
    train_deltas = compute_deltas(train_df)
    upset_rate_lookup = build_upset_rate_lookup(train_deltas)

    # Build both sets using the train-derived lookup
    train_features = build_features(train_df, upset_rate_lookup=upset_rate_lookup)
    test_features = build_features(test_df, upset_rate_lookup=upset_rate_lookup)

    return train_features, test_features


def _partition(df: pd.DataFrame) -> FeatureSet:
    """Partition a fully-engineered DataFrame into FeatureSet components."""
    # Extract target
    y = df["higher_seed_won"].copy()

    # Extract meta columns (keep round in both meta and X)
    meta_cols = [c for c in _META_COLS if c in df.columns]
    meta = df[meta_cols].copy()

    # Extract ranking targets
    ranking_cols = [c for c in _RANKING_TARGET_COLS if c in df.columns]
    ranking_targets = df[ranking_cols].copy()

    # Build feature set: deltas + expected_margin + context features + round
    exclude = set()
    # Exclude meta columns (but keep round — it's both meta and feature)
    exclude.update(c for c in _META_COLS if c != "round")
    # Exclude target
    exclude.add("higher_seed_won")
    # Exclude ranking targets
    exclude.update(_RANKING_TARGET_COLS)
    # Exclude raw _a/_b columns (redundant with deltas)
    exclude.update(c for c in df.columns if c.endswith("_a") or c.endswith("_b"))

    feature_cols = sorted(c for c in df.columns if c not in exclude)
    X = df[feature_cols].copy()

    logger.info(
        "Partitioned: %d feature cols, %d meta cols, %d ranking targets",
        len(feature_cols), len(meta_cols), len(ranking_cols),
    )

    return FeatureSet(
        X=X,
        y=y,
        meta=meta,
        ranking_targets=ranking_targets,
    )


def save_features(
    feature_set: FeatureSet,
    output_dir: Path | None = None,
    prefix: str = "",
) -> None:
    """Save FeatureSet to data/processed/ as parquet files.

    Args:
        feature_set: FeatureSet to save.
        output_dir: Output directory. Defaults to PROCESSED_DIR.
        prefix: Optional prefix for filenames (e.g., "train_", "test_").
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_set.X.to_parquet(output_dir / f"{prefix}X.parquet")
    feature_set.y.to_frame().to_parquet(output_dir / f"{prefix}y.parquet")
    feature_set.meta.to_parquet(output_dir / f"{prefix}meta.parquet")
    feature_set.ranking_targets.to_parquet(output_dir / f"{prefix}ranking_targets.parquet")

    logger.info(
        "Saved %sFeatureSet to %s: X=%s, y=%s",
        prefix, output_dir, feature_set.X.shape, feature_set.y.shape,
    )
