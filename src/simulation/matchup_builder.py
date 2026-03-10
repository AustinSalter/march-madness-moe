"""Build FeatureSets for hypothetical tournament matchups.

Bridges the feature pipeline (which expects historical data) with the MC engine
(which needs features for teams that haven't played yet). Constructs 1-row
DataFrames mimicking merge.py output and runs them through the same
compute_deltas() → add_context_features() → compute_ranking_targets() pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TOURNAMENT_DIR
from src.data.kaggle_loader import load_kenpom
from src.data.tournament_data import load_seeds, load_teams
from src.features.context_features import add_context_features, build_upset_rate_lookup
from src.features.kenpom_deltas import compute_deltas, detect_stat_pairs
from src.features.pipeline import FeatureSet, _partition
from src.features.ranking_criteria import compute_ranking_targets
from src.utils.team_names import build_name_to_id, resolve_team_id

logger = logging.getLogger(__name__)

# Columns that are identifiers, not stats (same logic as merge.py line 60-64)
_ID_COLS = {
    "season", "team", "team_id", "full_team_name", "conf", "conf_full",
    "ncaa_seed", "ncaa_region", "postseason", "rank",
}


class TeamStatsLookup:
    """Lookup table mapping team_id → KenPom stats for a single season.

    Enables building FeatureSets for hypothetical matchups between any two
    tournament teams.
    """

    def __init__(
        self,
        season: int,
        kenpom_df: pd.DataFrame,
        upset_rate_lookup: dict[tuple[int, int], float],
        tournament_dir: Path = TOURNAMENT_DIR,
    ):
        """Initialize lookup from KenPom data for a single season.

        Args:
            season: Tournament year.
            kenpom_df: KenPom DataFrame (from load_kenpom() or scrape_2026()).
            upset_rate_lookup: Historical upset rates from training data.
            tournament_dir: Path to tournament data directory.
        """
        self.season = season
        self.upset_rate_lookup = upset_rate_lookup

        # Build name → ID lookup
        name_to_id = build_name_to_id(tournament_dir)

        # Add team_id to kenpom if missing
        kp = kenpom_df.copy()
        if "team_id" not in kp.columns:
            kp["team_id"] = kp["team"].apply(lambda n: resolve_team_id(n, name_to_id))
            kp = kp.dropna(subset=["team_id"])
            kp["team_id"] = kp["team_id"].astype(int)

        # Filter to this season
        if "season" in kp.columns:
            kp = kp[kp["season"] == season].copy()

        # Identify stat columns (everything except identifiers)
        self.stat_cols = [c for c in kp.columns if c not in _ID_COLS]

        # Build team_id → stats dict
        self.team_stats: dict[int, dict] = {}
        for _, row in kp.iterrows():
            tid = int(row["team_id"])
            self.team_stats[tid] = {c: row[c] for c in self.stat_cols}

        # Load seeds: team_id → (region, seed_num)
        self.team_seeds: dict[int, tuple[str, int]] = {}
        try:
            seeds_df = load_seeds(tournament_dir)
            season_seeds = seeds_df[seeds_df["Season"] == season]
            for _, row in season_seeds.iterrows():
                self.team_seeds[int(row["TeamID"])] = (row["Region"], int(row["SeedNum"]))
        except FileNotFoundError:
            logger.warning("Could not load seeds for season %d", season)

        # Load team names: team_id → name
        self.team_names: dict[int, str] = {}
        try:
            teams_df = load_teams(tournament_dir)
            self.team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
        except FileNotFoundError:
            logger.warning("Could not load team names")

        # Detect stat pairs for delta computation (from a sample row)
        # Build a mock 2-team row to detect _a/_b stat pairs
        sample_cols_a = {f"{c}_a" for c in self.stat_cols}
        sample_cols_b = {f"{c}_b" for c in self.stat_cols}
        self._stat_bases = sorted(
            {c[:-2] for c in sample_cols_a} & {c[:-2] for c in sample_cols_b}
            - {"score", "seed", "team_id"}
        )

        logger.info(
            "TeamStatsLookup: season=%d, %d teams with stats, %d with seeds, %d stat cols",
            season, len(self.team_stats), len(self.team_seeds), len(self.stat_cols),
        )

    def get_matchup_features(
        self,
        team_id_a: int,
        team_id_b: int,
        round_num: int,
    ) -> FeatureSet:
        """Build a FeatureSet for a hypothetical matchup.

        Orients the matchup so team_a is the higher seed (lower seed number),
        matching the project convention.

        Args:
            team_id_a: First team ID.
            team_id_b: Second team ID.
            round_num: Tournament round (1-6).

        Returns:
            FeatureSet with 1 row of features.
        """
        # Determine seeds
        seed_a_info = self.team_seeds.get(team_id_a, ("?", 8))
        seed_b_info = self.team_seeds.get(team_id_b, ("?", 8))

        seed_num_a = seed_a_info[1]
        seed_num_b = seed_b_info[1]

        # Orient: team_a should be higher seed (lower number)
        if seed_num_a > seed_num_b:
            team_id_a, team_id_b = team_id_b, team_id_a
            seed_a_info, seed_b_info = seed_b_info, seed_a_info
            seed_num_a, seed_num_b = seed_num_b, seed_num_a
        elif seed_num_a == seed_num_b and team_id_a > team_id_b:
            team_id_a, team_id_b = team_id_b, team_id_a
            seed_a_info, seed_b_info = seed_b_info, seed_a_info

        # Build 1-row DataFrame mimicking merge.py output
        row = {
            "season": self.season,
            "round": round_num,
            "region": seed_a_info[0],
            "seed_a": seed_num_a,
            "team_id_a": team_id_a,
            "team_name_a": self.team_names.get(team_id_a, f"Team{team_id_a}"),
            "score_a": 0,
            "seed_b": seed_num_b,
            "team_id_b": team_id_b,
            "team_name_b": self.team_names.get(team_id_b, f"Team{team_id_b}"),
            "score_b": 0,
            "higher_seed_won": 0,  # Dummy target
        }

        # Add KenPom stats for both teams
        stats_a = self.team_stats.get(team_id_a, {})
        stats_b = self.team_stats.get(team_id_b, {})
        for col in self.stat_cols:
            row[f"{col}_a"] = stats_a.get(col, np.nan)
            row[f"{col}_b"] = stats_b.get(col, np.nan)

        df = pd.DataFrame([row])

        # Run through the same pipeline as build_features()
        df = compute_deltas(df, stat_names=self._stat_bases)
        df = add_context_features(df, upset_rate_lookup=self.upset_rate_lookup)
        df = compute_ranking_targets(df, upset_rate_lookup=self.upset_rate_lookup)

        return _partition(df)

    def get_win_prob(
        self,
        team_id_a: int,
        team_id_b: int,
        round_num: int,
        moe,
    ) -> float:
        """Get P(team_id_a wins) for a hypothetical matchup.

        The MOE returns P(higher_seed_wins). This method orients the result
        so it returns P(team_id_a wins) regardless of seed ordering.

        Args:
            team_id_a: First team ID (the team we want win prob for).
            team_id_b: Second team ID.
            round_num: Tournament round (1-6).
            moe: Trained MOEEnsemble.

        Returns:
            P(team_id_a wins) in [0, 1].
        """
        # Determine who is higher seed
        seed_a_info = self.team_seeds.get(team_id_a, ("?", 8))
        seed_b_info = self.team_seeds.get(team_id_b, ("?", 8))
        seed_num_a = seed_a_info[1]
        seed_num_b = seed_b_info[1]

        # Build features (auto-orients to higher seed = team_a)
        fs = self.get_matchup_features(team_id_a, team_id_b, round_num)

        # Fill NaN in gating features to prevent issues
        for col in fs.gating_features:
            if col in fs.X.columns:
                fs.X[col] = fs.X[col].fillna(0)

        p_higher_seed_wins = float(moe.predict_proba(fs)[0])

        # Orient: if team_id_a IS the higher seed, return p directly
        # If team_id_a is the LOWER seed, return 1 - p
        a_is_higher = (
            seed_num_a < seed_num_b
            or (seed_num_a == seed_num_b and team_id_a < team_id_b)
        )

        if a_is_higher:
            return p_higher_seed_wins
        else:
            return 1.0 - p_higher_seed_wins
