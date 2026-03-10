"""Team name normalization across KenPom (Pilafas), Kaggle March ML Mania, and NCAA sources.

Uses MTeamSpellings.csv from March ML Mania as the primary lookup — it maps
many alternative spellings to TeamID. Falls back to normalized fuzzy matching.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import TOURNAMENT_DIR

logger = logging.getLogger(__name__)


def normalize(name: str) -> str:
    """Normalize a team name for fuzzy matching.

    Lowercases, strips punctuation (except hyphens), collapses whitespace.
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_spellings(tournament_dir: Path = TOURNAMENT_DIR) -> dict[str, int]:
    """Load MTeamSpellings.csv into a normalized-name → TeamID lookup.

    Returns dict mapping lowercased spelling → TeamID.
    """
    csv_path = tournament_dir / "MTeamSpellings.csv"
    if not csv_path.exists():
        logger.warning("MTeamSpellings.csv not found at %s", csv_path)
        return {}

    df = pd.read_csv(csv_path, encoding="latin1")
    mapping = {}
    for _, row in df.iterrows():
        key = str(row["TeamNameSpelling"]).strip().lower()
        mapping[key] = int(row["TeamID"])
    return mapping


def load_teams(tournament_dir: Path = TOURNAMENT_DIR) -> pd.DataFrame:
    """Load MTeams.csv → DataFrame with [TeamID, TeamName]."""
    csv_path = tournament_dir / "MTeams.csv"
    return pd.read_csv(csv_path)[["TeamID", "TeamName"]]


def build_name_to_id(tournament_dir: Path = TOURNAMENT_DIR) -> dict[str, int]:
    """Build a comprehensive team name → TeamID mapping.

    Combines MTeamSpellings (many-to-one) with MTeams (canonical names).
    Keys are lowercased for case-insensitive lookup.
    """
    mapping = load_spellings(tournament_dir)

    # Also add canonical names from MTeams
    teams = load_teams(tournament_dir)
    for _, row in teams.iterrows():
        key = str(row["TeamName"]).strip().lower()
        mapping[key] = int(row["TeamID"])

    # Manual overrides for names the Pilafas DEV table uses that don't
    # appear in MTeamSpellings.csv
    # Look up IDs from the teams table for accuracy
    id_lookup = {str(row["TeamName"]).strip().lower(): int(row["TeamID"]) for _, row in teams.iterrows()}

    manual = {
        "miami": id_lookup.get("miami fl", 1274),
        "ualbany": id_lookup.get("suny albany", id_lookup.get("albany ny", 1100)),
        "app state": id_lookup.get("appalachian st", 1106),
        "ul monroe": id_lookup.get("la monroe", 1349),
        "kansas city": id_lookup.get("umkc", 1254),
        "maryland eastern shore": id_lookup.get("md e shore", 1269),
    }
    for key, tid in manual.items():
        mapping[key] = tid

    return mapping


def resolve_team_id(name: str, name_to_id: dict[str, int]) -> int | None:
    """Resolve a team name to a TeamID using multi-strategy matching.

    Tries in order:
    1. Exact lowercase match against spellings + canonical names
    2. Normalized match (strip punctuation)
    3. None if no match found
    """
    key = name.strip().lower()

    # 1. Exact match
    if key in name_to_id:
        return name_to_id[key]

    # 2. Normalized match
    norm = normalize(name)
    if norm in name_to_id:
        return name_to_id[norm]

    # 3. Try without common suffixes like "St." → "St", "State" variations
    for variant in _generate_variants(key):
        if variant in name_to_id:
            return name_to_id[variant]

    return None


def _generate_variants(name: str) -> list[str]:
    """Generate common spelling variants for a team name."""
    variants = []
    # "St." ↔ "St" ↔ "Saint"
    if "st." in name:
        variants.append(name.replace("st.", "st"))
        variants.append(name.replace("st.", "saint"))
    if "saint " in name:
        variants.append(name.replace("saint ", "st. "))
        variants.append(name.replace("saint ", "st "))
    # Parenthetical ↔ space: "Miami (OH)" ↔ "Miami OH"
    variants.append(re.sub(r"\s*\(([^)]+)\)", r" \1", name))
    # "State" ↔ "St"
    if name.endswith(" state"):
        variants.append(name.replace(" state", " st"))
    if name.endswith(" st"):
        variants.append(name.replace(" st", " state"))
    return variants
