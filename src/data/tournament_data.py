"""Load historical tournament matchups from Kaggle March ML Mania data.

Expected files in data/raw/tournament/:
  - MNCAATourneyCompactResults.csv: Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
  - MNCAATourneySeeds.csv: Season, Seed, TeamID
  - MTeams.csv: TeamID, TeamName, FirstD1Season, LastD1Season

Returns matchup-level data with team names, seeds, and round info.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import TOURNAMENT_DIR, FIRST_YEAR, LAST_YEAR, SKIP_YEARS

logger = logging.getLogger(__name__)


def parse_seed(seed_str: str) -> tuple[str, int]:
    """Parse seed string into (region, seed_number).

    Args:
        seed_str: e.g. "W01", "X16a"

    Returns:
        Tuple of (region letter, seed integer).
    """
    region = seed_str[0]
    seed_num = int(seed_str[1:3])
    return region, seed_num


def _assign_rounds(season_df: pd.DataFrame) -> pd.Series:
    """Assign rounds to games within a single season using game ordering.

    More robust than DayNum ranges — uses the known tournament structure:
    R64=32 games, R32=16, S16=8, E8=4, F4=2, Championship=1.
    First Four play-in games (DayNum < 134) get round 0 and are dropped later.
    """
    df = season_df.sort_values("DayNum").reset_index(drop=True)
    rounds = []
    # Expected game counts per round
    structure = [(1, 32), (2, 16), (3, 8), (4, 4), (5, 2), (6, 1)]

    # Filter out First Four (very early DayNum games)
    min_day = df["DayNum"].min()
    # First Four games are typically 2 days before R64
    first_four_mask = df["DayNum"] < min_day + 2 if len(df) > 63 else pd.Series([False] * len(df))

    # For seasons with First Four (2011+), there are 67 games total
    # For earlier seasons, 63 games
    n_games = len(df)

    if n_games <= 63:
        # No First Four
        idx = 0
        for round_num, count in structure:
            for _ in range(count):
                if idx < n_games:
                    rounds.append(round_num)
                    idx += 1
    else:
        # Has First Four (4 play-in games)
        n_playin = n_games - 63
        for i in range(n_playin):
            rounds.append(0)  # Play-in / First Four
        idx = n_playin
        for round_num, count in structure:
            for _ in range(count):
                if idx < n_games:
                    rounds.append(round_num)
                    idx += 1

    # Pad if somehow short
    while len(rounds) < n_games:
        rounds.append(0)

    return pd.Series(rounds, index=df.index)


def load_teams(tournament_dir: Path = TOURNAMENT_DIR) -> pd.DataFrame:
    """Load team ID → name mapping from MTeams.csv."""
    csv_path = tournament_dir / "MTeams.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MTeams.csv not found in {tournament_dir}")
    return pd.read_csv(csv_path)[["TeamID", "TeamName"]]


def load_seeds(tournament_dir: Path = TOURNAMENT_DIR) -> pd.DataFrame:
    """Load tournament seeds from MNCAATourneySeeds.csv.

    Returns DataFrame with columns: Season, TeamID, Region, SeedNum
    """
    csv_path = tournament_dir / "MNCAATourneySeeds.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MNCAATourneySeeds.csv not found in {tournament_dir}")

    df = pd.read_csv(csv_path)
    parsed = df["Seed"].apply(parse_seed)
    df["Region"] = parsed.apply(lambda x: x[0])
    df["SeedNum"] = parsed.apply(lambda x: x[1])
    return df[["Season", "TeamID", "Region", "SeedNum"]]


def load_results(tournament_dir: Path = TOURNAMENT_DIR) -> pd.DataFrame:
    """Load tournament game results from MNCAATourneyCompactResults.csv."""
    csv_path = tournament_dir / "MNCAATourneyCompactResults.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MNCAATourneyCompactResults.csv not found in {tournament_dir}")
    return pd.read_csv(csv_path)


def load_matchups(tournament_dir: Path = TOURNAMENT_DIR) -> pd.DataFrame:
    """Load and assemble tournament matchups with seeds, teams, and round info.

    Returns DataFrame with columns:
        season, round, region,
        seed_a (higher seed), team_id_a, team_name_a, score_a,
        seed_b (lower seed), team_id_b, team_name_b, score_b,
        higher_seed_won

    Convention: team_a is ALWAYS the higher (lower-numbered) seed.
    For same-seed matchups, team_a has the lower TeamID (arbitrary but consistent).
    """
    results = load_results(tournament_dir)
    seeds = load_seeds(tournament_dir)
    teams = load_teams(tournament_dir)

    # Filter to valid years
    valid_years = set(range(FIRST_YEAR, LAST_YEAR + 1)) - SKIP_YEARS
    results = results[results["Season"].isin(valid_years)].copy()

    # Assign rounds per season using game ordering
    results = results.sort_values(["Season", "DayNum"]).reset_index(drop=True)
    round_vals = pd.Series(0, index=results.index)
    for season, group in results.groupby("Season"):
        rounds = _assign_rounds(group)
        round_vals.iloc[group.index] = rounds.values
    results["Round"] = round_vals

    # Merge seeds for both teams
    results = results.merge(
        seeds.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed", "Region": "WRegion"}),
        on=["Season", "WTeamID"], how="left",
    )
    results = results.merge(
        seeds.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed", "Region": "LRegion"}),
        on=["Season", "LTeamID"], how="left",
    )

    # Merge team names
    id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))
    results["WTeamName"] = results["WTeamID"].map(id_to_name)
    results["LTeamName"] = results["LTeamID"].map(id_to_name)

    # Orient: team_a = higher seed (lower number), team_b = lower seed
    rows = []
    for _, r in results.iterrows():
        if pd.isna(r["WSeed"]) or pd.isna(r["LSeed"]):
            continue  # Skip games without seed info

        w_seed, l_seed = int(r["WSeed"]), int(r["LSeed"])
        rnd = int(r["Round"])

        if w_seed < l_seed:
            row = _make_row(r, rnd, "W_is_A")
        elif l_seed < w_seed:
            row = _make_row(r, rnd, "L_is_A")
        else:
            # Same seed — lower TeamID is team_a
            if r["WTeamID"] < r["LTeamID"]:
                row = _make_row(r, rnd, "W_is_A")
            else:
                row = _make_row(r, rnd, "L_is_A")
        rows.append(row)

    matchups = pd.DataFrame(rows)

    # Drop play-in games (round 0)
    n_playin = (matchups["round"] == 0).sum()
    if n_playin > 0:
        logger.info("Dropping %d First Four / play-in games", n_playin)
        matchups = matchups[matchups["round"] > 0]

    matchups = matchups.sort_values(["season", "round", "seed_a"]).reset_index(drop=True)

    logger.info(
        "Loaded %d tournament matchups across %d seasons (%d–%d)",
        len(matchups), matchups["season"].nunique(),
        matchups["season"].min(), matchups["season"].max(),
    )
    return matchups


def _make_row(r, rnd: int, orientation: str) -> dict:
    """Create an oriented matchup row from a results row."""
    if orientation == "W_is_A":
        return {
            "season": int(r["Season"]),
            "round": rnd,
            "region": r["WRegion"],
            "seed_a": int(r["WSeed"]),
            "team_id_a": int(r["WTeamID"]),
            "team_name_a": r["WTeamName"],
            "score_a": int(r["WScore"]),
            "seed_b": int(r["LSeed"]),
            "team_id_b": int(r["LTeamID"]),
            "team_name_b": r["LTeamName"],
            "score_b": int(r["LScore"]),
            "higher_seed_won": 1,
        }
    else:  # L_is_A
        return {
            "season": int(r["Season"]),
            "round": rnd,
            "region": r["LRegion"],
            "seed_a": int(r["LSeed"]),
            "team_id_a": int(r["LTeamID"]),
            "team_name_a": r["LTeamName"],
            "score_a": int(r["LScore"]),
            "seed_b": int(r["WSeed"]),
            "team_id_b": int(r["WTeamID"]),
            "team_name_b": r["WTeamName"],
            "score_b": int(r["WScore"]),
            "higher_seed_won": 0,
        }
