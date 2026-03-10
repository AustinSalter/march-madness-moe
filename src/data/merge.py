"""Join KenPom team-level stats with tournament matchup records.

Uses MTeamSpellings.csv for robust name matching between KenPom ESPN names
and March ML Mania TeamIDs. The DEV table uses "Mapped ESPN Team Name" which
closely matches ESPN-style names.
"""

import logging

import pandas as pd

from src.data.kaggle_loader import load_kenpom
from src.data.tournament_data import load_matchups
from src.utils.team_names import build_name_to_id, resolve_team_id

logger = logging.getLogger(__name__)


def merge_kenpom_with_matchups(
    kenpom_df: pd.DataFrame | None = None,
    matchups_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join KenPom stats onto tournament matchups for both teams.

    Args:
        kenpom_df: KenPom data from load_kenpom(). Loaded if None.
        matchups_df: Tournament matchups from load_matchups(). Loaded if None.

    Returns:
        DataFrame with one row per tournament game, containing:
        - Matchup info: season, round, region, seeds, team names, scores, outcome
        - KenPom stats for team_a (higher seed): suffixed with _a
        - KenPom stats for team_b (lower seed): suffixed with _b
    """
    if kenpom_df is None:
        kenpom_df = load_kenpom()
    if matchups_df is None:
        matchups_df = load_matchups()

    # Build comprehensive name → TeamID lookup
    name_to_id = build_name_to_id()

    # Resolve KenPom team names to TeamIDs
    kenpom = kenpom_df.copy()
    kenpom["team_id"] = kenpom["team"].apply(lambda n: resolve_team_id(n, name_to_id))

    # Report unmatched teams
    unmatched = kenpom[kenpom["team_id"].isna()]["team"].unique()
    if len(unmatched) > 0:
        logger.warning(
            "%d KenPom team names could not be matched to TeamID: %s",
            len(unmatched), list(unmatched[:15]),
        )

    # Drop unmatched and convert to int
    kenpom = kenpom.dropna(subset=["team_id"]).copy()
    kenpom["team_id"] = kenpom["team_id"].astype(int)

    # Identify stat columns to merge (everything except identifiers)
    id_cols = {
        "season", "team", "team_id", "full_team_name", "conf", "conf_full",
        "ncaa_seed", "ncaa_region", "postseason", "rank",
    }
    stat_cols = [c for c in kenpom.columns if c not in id_cols]
    merge_cols = ["season", "team_id"] + stat_cols

    # Merge for team_a
    merged = matchups_df.merge(
        kenpom[merge_cols].rename(
            columns={"team_id": "team_id_a", **{c: f"{c}_a" for c in stat_cols}}
        ),
        on=["season", "team_id_a"],
        how="left",
    )

    # Merge for team_b
    merged = merged.merge(
        kenpom[merge_cols].rename(
            columns={"team_id": "team_id_b", **{c: f"{c}_b" for c in stat_cols}}
        ),
        on=["season", "team_id_b"],
        how="left",
    )

    # Report merge quality
    n_total = len(merged)
    sample_col = f"{stat_cols[0]}_a" if stat_cols else None
    if sample_col:
        n_missing_a = merged[sample_col].isna().sum()
        n_missing_b = merged[f"{stat_cols[0]}_b"].isna().sum()

        if n_missing_a > 0 or n_missing_b > 0:
            logger.warning(
                "Merge gaps: %d/%d missing team_a stats, %d/%d missing team_b stats",
                n_missing_a, n_total, n_missing_b, n_total,
            )
            if n_missing_a > 0:
                missing = merged.loc[merged[sample_col].isna(), ["season", "team_name_a"]].drop_duplicates()
                logger.warning("Missing team_a: %s", missing.values.tolist()[:10])
            if n_missing_b > 0:
                missing = merged.loc[merged[f"{stat_cols[0]}_b"].isna(), ["season", "team_name_b"]].drop_duplicates()
                logger.warning("Missing team_b: %s", missing.values.tolist()[:10])
        else:
            logger.info("Merge complete: all %d games have KenPom stats for both teams", n_total)

    return merged
