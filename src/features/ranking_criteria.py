"""Per-expert ranking target computation for Spearman correlation evaluation."""

import logging

import numpy as np
import pandas as pd

from src.features.context_features import build_upset_rate_lookup

logger = logging.getLogger(__name__)


def compute_seed_implied_prob(
    df: pd.DataFrame, upset_rate_lookup: dict | None = None
) -> pd.Series:
    """Expert 1 target: P(higher seed wins) = 1 - upset_rate for this matchup.

    Uses same lookup as hist_upset_rate (complement).

    Args:
        df: DataFrame with seed_a, seed_b columns.
        upset_rate_lookup: From build_upset_rate_lookup(). If None, builds from df.

    Returns:
        Series of seed-implied win probabilities in [0, 1].
    """
    if upset_rate_lookup is None:
        upset_rate_lookup = build_upset_rate_lookup(df)

    # Build fallback rates by seed_diff
    fallback_by_diff: dict[int, list[float]] = {}
    for (sa, sb), rate in upset_rate_lookup.items():
        diff = abs(sb - sa)
        fallback_by_diff.setdefault(diff, []).append(rate)
    fallback_rates = {d: sum(r) / len(r) for d, r in fallback_by_diff.items()}
    overall_avg = (
        sum(upset_rate_lookup.values()) / len(upset_rate_lookup)
        if upset_rate_lookup
        else 0.5
    )

    def _get_prob(row):
        key = (int(row["seed_a"]), int(row["seed_b"]))
        if key in upset_rate_lookup:
            upset_rate = upset_rate_lookup[key]
        else:
            diff = abs(key[1] - key[0])
            upset_rate = fallback_rates.get(diff, overall_avg)
        return 1.0 - upset_rate

    return df.apply(_get_prob, axis=1).rename("seed_implied_prob")


def compute_efficiency_delta_rank(
    df: pd.DataFrame, max_adjem_delta: float | None = None
) -> pd.Series:
    """Expert 2 target: normalized |adjem_delta| / max. Values in [0, 1].

    Args:
        df: DataFrame with adjem_delta column.
        max_adjem_delta: Normalization denominator. If None, computed from df.

    Returns:
        Series of normalized efficiency delta values in [0, 1].
    """
    if "adjem_delta" not in df.columns:
        logger.warning("adjem_delta not found — returning NaN series")
        return pd.Series(float("nan"), index=df.index, name="efficiency_delta_rank")

    abs_delta = df["adjem_delta"].abs()
    if max_adjem_delta is None:
        max_adjem_delta = abs_delta.max()
    if max_adjem_delta == 0:
        max_adjem_delta = 1.0

    result = (abs_delta / max_adjem_delta).clip(0, 1)
    return result.rename("efficiency_delta_rank")


def compute_game_certainty_score(
    df: pd.DataFrame, max_seed_diff: float = 15.0
) -> pd.Series:
    """Expert 3 target: composite predictability score.

    = 0.4 * |seed_diff| / 15
    + 0.3 * |adjem_delta| / max
    + 0.2 * (1 - |luck_delta| / max)
    + 0.1 * (1 - |ncsos_adjem_delta| / max)

    NaN luck/ncsos terms → neutral 0.5. Clamped to [0, 1].

    Args:
        df: DataFrame with seed_diff, adjem_delta, and optionally luck_delta
            and ncsos_adjem_delta columns.
        max_seed_diff: Max possible seed difference (default 15).

    Returns:
        Series of certainty scores in [0, 1].
    """
    # seed_diff component
    seed_diff = df.get("seed_diff", pd.Series(0, index=df.index))
    seed_term = seed_diff.abs() / max_seed_diff

    # adjem_delta component
    if "adjem_delta" in df.columns:
        abs_adjem = df["adjem_delta"].abs()
        max_adjem = abs_adjem.max()
        if max_adjem == 0:
            max_adjem = 1.0
        adjem_term = abs_adjem / max_adjem
    else:
        adjem_term = pd.Series(0.5, index=df.index)

    # luck_delta component (NaN → neutral 0.5)
    if "luck_delta" in df.columns:
        abs_luck = df["luck_delta"].abs()
        max_luck = abs_luck.max()
        if max_luck == 0:
            max_luck = 1.0
        luck_term = 1.0 - abs_luck / max_luck
        luck_term = luck_term.fillna(0.5)
    else:
        luck_term = pd.Series(0.5, index=df.index)

    # ncsos_adjem_delta component (NaN → neutral 0.5)
    if "ncsos_adjem_delta" in df.columns:
        abs_ncsos = df["ncsos_adjem_delta"].abs()
        max_ncsos = abs_ncsos.max()
        if max_ncsos == 0:
            max_ncsos = 1.0
        ncsos_term = 1.0 - abs_ncsos / max_ncsos
        ncsos_term = ncsos_term.fillna(0.5)
    else:
        ncsos_term = pd.Series(0.5, index=df.index)

    score = 0.4 * seed_term + 0.3 * adjem_term + 0.2 * luck_term + 0.1 * ncsos_term
    return score.clip(0, 1).rename("game_certainty_score")


def compute_ranking_targets(
    df: pd.DataFrame, upset_rate_lookup: dict | None = None
) -> pd.DataFrame:
    """Add all three ranking target columns to df.

    Args:
        df: DataFrame with delta and context columns already computed.
        upset_rate_lookup: From build_upset_rate_lookup(). If None, builds from df.

    Returns:
        Copy of df with seed_implied_prob, efficiency_delta_rank,
        game_certainty_score columns added.
    """
    out = df.copy()
    out["seed_implied_prob"] = compute_seed_implied_prob(out, upset_rate_lookup)
    out["efficiency_delta_rank"] = compute_efficiency_delta_rank(out)
    out["game_certainty_score"] = compute_game_certainty_score(out)

    logger.info(
        "Added ranking targets: seed_implied_prob [%.2f, %.2f], "
        "efficiency_delta_rank [%.2f, %.2f], game_certainty_score [%.2f, %.2f]",
        out["seed_implied_prob"].min(), out["seed_implied_prob"].max(),
        out["efficiency_delta_rank"].min(), out["efficiency_delta_rank"].max(),
        out["game_certainty_score"].min(), out["game_certainty_score"].max(),
    )
    return out
