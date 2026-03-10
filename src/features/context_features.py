"""Game context features: seed diff, round number, conference flags, historical upset rates."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TOURNAMENT_DIR

logger = logging.getLogger(__name__)


def build_upset_rate_lookup(train_df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Compute historical upset rate from training data.

    Groups by (seed_a, seed_b) and computes 1 - mean(higher_seed_won).

    Args:
        train_df: Training data with seed_a, seed_b, higher_seed_won columns.

    Returns:
        Dict mapping (seed_a, seed_b) -> upset_rate.
    """
    required = {"seed_a", "seed_b", "higher_seed_won"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"train_df missing columns: {required - set(train_df.columns)}")

    grouped = train_df.groupby(["seed_a", "seed_b"])["higher_seed_won"].mean()
    lookup = {(int(sa), int(sb)): 1.0 - rate for (sa, sb), rate in grouped.items()}

    logger.info("Built upset rate lookup with %d seed pairings", len(lookup))
    return lookup


def _get_fallback_upset_rates(
    upset_rate_lookup: dict[tuple[int, int], float],
) -> dict[int, float]:
    """Compute fallback upset rates grouped by abs(seed_diff) magnitude."""
    fallback_by_diff: dict[int, list[float]] = {}
    for (sa, sb), rate in upset_rate_lookup.items():
        diff = abs(sb - sa)
        fallback_by_diff.setdefault(diff, []).append(rate)
    return {d: sum(rates) / len(rates) for d, rates in fallback_by_diff.items()}


def _load_conferences(
    conf_path: Path | None = None,
) -> pd.DataFrame:
    """Load MTeamConferences.csv for same-conference lookup."""
    if conf_path is None:
        conf_path = TOURNAMENT_DIR / "MTeamConferences.csv"
    if not conf_path.exists():
        logger.warning("MTeamConferences.csv not found at %s", conf_path)
        return pd.DataFrame()
    return pd.read_csv(conf_path)


# Standard NCAA bracket slot assignment (seed -> 1-indexed position in 16-team region)
_SEED_TO_SLOT = {
    1: 1, 16: 2, 8: 3, 9: 4, 5: 5, 12: 6, 4: 7, 13: 8,
    6: 9, 11: 10, 3: 11, 14: 12, 7: 13, 10: 14, 2: 15, 15: 16,
}


def _compute_chalk_round(seed_a: int, seed_b: int) -> int:
    """Earliest tournament round two seeds would meet if the bracket held to chalk.

    Returns 5 for cross-region matchups (Final Four+) or unrecognized seeds.
    """
    if seed_a not in _SEED_TO_SLOT or seed_b not in _SEED_TO_SLOT:
        return 5
    slot_a = _SEED_TO_SLOT[seed_a] - 1  # 0-indexed
    slot_b = _SEED_TO_SLOT[seed_b] - 1
    for r in range(1, 5):
        if slot_a // (2 ** r) == slot_b // (2 ** r):
            return r
    return 5


def add_context_features(
    df: pd.DataFrame,
    upset_rate_lookup: dict | None = None,
    conf_path: Path | None = None,
) -> pd.DataFrame:
    """Add all context features: seed_diff, same_conf, hist_upset_rate.

    Note: 'round' already exists in the data.

    Args:
        df: DataFrame with seed_a, seed_b, season, team_id_a, team_id_b.
        upset_rate_lookup: From build_upset_rate_lookup(). If None, hist_upset_rate
            is filled with NaN.
        conf_path: Path to MTeamConferences.csv. Uses TOURNAMENT_DIR default.

    Returns:
        Copy of df with context columns added.
    """
    out = df.copy()

    # seed_diff: always >= 0 by convention (team_a is higher seed)
    out["seed_diff"] = out["seed_b"] - out["seed_a"]

    # same_conf: whether teams share a conference
    conf_df = _load_conferences(conf_path)
    if not conf_df.empty:
        conf_a = conf_df.rename(
            columns={"Season": "season", "TeamID": "team_id_a", "ConfAbbrev": "conf_a_lookup"}
        )
        out = out.merge(
            conf_a[["season", "team_id_a", "conf_a_lookup"]],
            on=["season", "team_id_a"],
            how="left",
        )

        conf_b = conf_df.rename(
            columns={"Season": "season", "TeamID": "team_id_b", "ConfAbbrev": "conf_b_lookup"}
        )
        out = out.merge(
            conf_b[["season", "team_id_b", "conf_b_lookup"]],
            on=["season", "team_id_b"],
            how="left",
        )

        out["same_conf"] = (out["conf_a_lookup"] == out["conf_b_lookup"]).astype(int)
        out["same_conf"] = out["same_conf"].fillna(0).astype(int)
        out = out.drop(columns=["conf_a_lookup", "conf_b_lookup"])
    else:
        out["same_conf"] = 0

    # hist_upset_rate: historical upset probability for this seed pairing
    if upset_rate_lookup is not None:
        fallback_rates = _get_fallback_upset_rates(upset_rate_lookup)
        overall_avg = (
            sum(upset_rate_lookup.values()) / len(upset_rate_lookup)
            if upset_rate_lookup
            else 0.5
        )

        def _get_upset_rate(row):
            key = (int(row["seed_a"]), int(row["seed_b"]))
            if key in upset_rate_lookup:
                return upset_rate_lookup[key]
            diff = abs(key[1] - key[0])
            return fallback_rates.get(diff, overall_avg)

        out["hist_upset_rate"] = out.apply(_get_upset_rate, axis=1)
    else:
        out["hist_upset_rate"] = float("nan")

    # higher_seed: the favorite's actual seed number
    out["higher_seed"] = out["seed_a"]

    # seed_sum: total seeds in matchup (bracket position proxy)
    out["seed_sum"] = out["seed_a"] + out["seed_b"]

    # log_seed_ratio: log(underdog / favorite) — multiplicative mismatch
    out["log_seed_ratio"] = np.log(out["seed_b"] / out["seed_a"])

    # chalk_round: earliest round this matchup would occur under chalk bracket
    out["chalk_round"] = out.apply(
        lambda row: _compute_chalk_round(int(row["seed_a"]), int(row["seed_b"])),
        axis=1,
    )

    n_same_conf = out["same_conf"].sum()
    logger.info(
        "Added context features: seed_diff range [%d, %d], %d same-conf games",
        out["seed_diff"].min(),
        out["seed_diff"].max(),
        n_same_conf,
    )
    return out
