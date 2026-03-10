"""Bracket data model: regions, slots, game progression.

Parses MNCAATourneySlots.csv into a graph-like structure where each slot
references either seed strings (R1 games) or prior slot names (later rounds).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import TOURNAMENT_DIR
from src.data.tournament_data import load_seeds, load_teams, parse_seed

logger = logging.getLogger(__name__)

# Most recent season with slot data — used as template for newer seasons
_TEMPLATE_SEASON = 2023


@dataclass
class BracketSlot:
    """A single game slot in the bracket."""
    slot_name: str       # "R1W1", "R2W1", "R6CH", "W16" (play-in)
    round_num: int       # 1-6 (0 for play-in)
    strong_source: str   # Seed string ("W01") or prior slot name ("R1W1")
    weak_source: str     # Seed string ("W16") or prior slot name ("R1W8")


@dataclass
class Bracket:
    """Full tournament bracket for a single season."""
    season: int
    slots: dict[str, BracketSlot] = field(default_factory=dict)
    teams: dict[str, tuple[int, str, int]] = field(default_factory=dict)
    round_slots: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_tournament_slots(self) -> int:
        """Number of non-play-in slots (should be 63)."""
        return sum(1 for s in self.slots.values() if s.round_num > 0)

    @property
    def n_playin_slots(self) -> int:
        """Number of play-in (First Four) slots."""
        return sum(1 for s in self.slots.values() if s.round_num == 0)

    def get_team(self, seed_str: str) -> tuple[int, str, int] | None:
        """Look up team by seed string (e.g., 'W01')."""
        return self.teams.get(seed_str)


def _parse_round_from_slot(slot_name: str) -> int:
    """Extract round number from a slot name.

    R1W1 → 1, R2W1 → 2, R5WX → 5, R6CH → 6.
    Play-in slots (e.g., W16, X16, Y11, Z11) → 0.
    """
    if slot_name.startswith("R") and slot_name[1].isdigit():
        return int(slot_name[1])
    return 0


def is_seed_reference(source: str) -> bool:
    """Check if a source string is a seed ref ('W01') vs a slot ref ('R1W1').

    Seed refs: region letter + 2-digit seed number, optionally with a/b suffix
    for play-in teams (e.g., 'W16a', 'X16b').
    """
    if len(source) < 3:
        return False
    # Slot refs always start with R + digit
    if source[0] == "R" and len(source) > 1 and source[1].isdigit():
        return False
    # Seed refs: letter + digits (+ optional a/b)
    return source[0].isalpha() and source[1:3].isdigit()


def _build_seed_str(region: str, seed_num: int) -> str:
    """Build seed string like 'W01' from region and seed number."""
    return f"{region}{seed_num:02d}"


def load_bracket(
    season: int,
    tournament_dir: Path = TOURNAMENT_DIR,
) -> Bracket:
    """Load bracket structure for a given season.

    For seasons not in MNCAATourneySlots.csv (e.g., 2026), uses the most
    recent available season's slot structure as a template — bracket topology
    is identical year-to-year, only team assignments change.

    Args:
        season: Tournament year.
        tournament_dir: Path to tournament data directory.

    Returns:
        Bracket with slots, teams, and round ordering.
    """
    # Load slot structure
    slots_csv = tournament_dir / "MNCAATourneySlots.csv"
    if not slots_csv.exists():
        raise FileNotFoundError(f"MNCAATourneySlots.csv not found in {tournament_dir}")

    all_slots = pd.read_csv(slots_csv)
    available_seasons = sorted(all_slots["Season"].unique())

    if season in available_seasons:
        slot_season = season
    else:
        # Use most recent available season as template
        slot_season = max(available_seasons)
        logger.info(
            "Season %d not in slots CSV, using %d template", season, slot_season,
        )

    season_slots = all_slots[all_slots["Season"] == slot_season]

    # Parse slots
    slots: dict[str, BracketSlot] = {}
    for _, row in season_slots.iterrows():
        slot_name = row["Slot"]
        round_num = _parse_round_from_slot(slot_name)
        slots[slot_name] = BracketSlot(
            slot_name=slot_name,
            round_num=round_num,
            strong_source=row["StrongSeed"],
            weak_source=row["WeakSeed"],
        )

    # Group slots by round in execution order
    round_slots: dict[int, list[str]] = {}
    for slot_name, slot in slots.items():
        round_slots.setdefault(slot.round_num, []).append(slot_name)
    # Sort each round's slots for deterministic ordering
    for rnd in round_slots:
        round_slots[rnd] = sorted(round_slots[rnd])

    # Load team assignments for this season
    # Use raw Seed column (e.g., "W16a", "W16b") to preserve play-in suffixes
    teams: dict[str, tuple[int, str, int]] = {}
    try:
        raw_seeds = pd.read_csv(tournament_dir / "MNCAATourneySeeds.csv")
        teams_df = load_teams(tournament_dir)
        id_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

        season_seeds = raw_seeds[raw_seeds["Season"] == season]
        for _, row in season_seeds.iterrows():
            seed_str = row["Seed"]  # e.g., "W01", "W16a", "W16b"
            team_id = int(row["TeamID"])
            _, seed_num = parse_seed(seed_str)
            team_name = id_to_name.get(team_id, f"Team{team_id}")
            teams[seed_str] = (team_id, team_name, seed_num)
    except FileNotFoundError:
        logger.warning("Could not load seeds/teams for season %d", season)

    bracket = Bracket(
        season=season,
        slots=slots,
        teams=teams,
        round_slots=round_slots,
    )

    logger.info(
        "Loaded bracket for %d: %d tournament slots, %d play-in slots, %d teams",
        season, bracket.n_tournament_slots, bracket.n_playin_slots, len(teams),
    )
    return bracket


def get_slot_round_order(bracket: Bracket) -> list[int]:
    """Return rounds in execution order: 0 (play-in), 1, 2, 3, 4, 5, 6."""
    return sorted(bracket.round_slots.keys())
