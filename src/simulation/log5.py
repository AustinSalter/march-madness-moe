"""Log5 baseline win probability computation.

Bill James' Log5 formula estimates the probability of team A beating team B
based on their respective winning percentages against common competition.
Used as a sanity-check comparator for the MOE system.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import TOURNAMENT_DIR

logger = logging.getLogger(__name__)


def log5_win_probability(wp_a: float, wp_b: float) -> float:
    """Log5: P(A beats B) given each team's overall win percentage.

    Formula: P(A) = (wp_a - wp_a*wp_b) / (wp_a + wp_b - 2*wp_a*wp_b)

    Edge cases:
    - Both 1.0 or both 0.0 → 0.5
    - wp_a=1.0, wp_b<1.0 → 1.0
    - wp_a=0.0, wp_b>0.0 → 0.0

    Args:
        wp_a: Team A regular-season win percentage in [0, 1].
        wp_b: Team B regular-season win percentage in [0, 1].

    Returns:
        P(A beats B) in [0, 1].
    """
    denom = wp_a + wp_b - 2 * wp_a * wp_b
    if abs(denom) < 1e-10:
        return 0.5
    return (wp_a - wp_a * wp_b) / denom


def build_log5_win_prob_fn(
    season: int,
    results_path: Path | None = None,
) -> callable:
    """Build a (team_a, team_b, round) → float function from regular season records.

    The `round` argument is accepted but ignored — Log5 is round-independent.

    Args:
        season: Season to compute win percentages for.
        results_path: Path to MRegularSeasonCompactResults.csv.

    Returns:
        Callable (team_id_a, team_id_b, round_num) → P(team_a beats team_b).
    """
    if results_path is None:
        results_path = TOURNAMENT_DIR / "MRegularSeasonCompactResults.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Regular season results not found: {results_path}")

    results = pd.read_csv(results_path)
    season_results = results[results["Season"] == season]

    if season_results.empty:
        logger.warning("No regular season results for %d, using flat 0.5", season)
        return lambda a, b, r: 0.5

    # Compute win percentage per team
    wins = season_results.groupby("WTeamID").size()
    losses = season_results.groupby("LTeamID").size()

    all_teams = set(wins.index) | set(losses.index)
    win_pcts: dict[int, float] = {}
    for team in all_teams:
        w = wins.get(team, 0)
        l = losses.get(team, 0)
        total = w + l
        win_pcts[team] = w / total if total > 0 else 0.5

    logger.info(
        "Log5 baseline: season=%d, %d teams, mean win%%=%.3f",
        season, len(win_pcts), sum(win_pcts.values()) / len(win_pcts),
    )

    def _win_prob_fn(team_id_a: int, team_id_b: int, round_num: int) -> float:
        wp_a = win_pcts.get(team_id_a, 0.5)
        wp_b = win_pcts.get(team_id_b, 0.5)
        return log5_win_probability(wp_a, wp_b)

    return _win_prob_fn
