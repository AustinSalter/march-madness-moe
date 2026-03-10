"""Load KenPom metrics from Pilafas Kaggle dataset + scraped pomeroy ratings.

Data sources:
  1. 'DEV _ March Madness.csv' (Pilafas) — 165 columns, ~8300 rows
     Has: AdjEM, AdjOE, AdjDE, AdjTempo, four factors, misc stats, height/experience
     Missing: Luck, SOS, NCSOS

  2. 'data/cache/kenpom_pomeroy_ratings.parquet' (scraped via kenpompy)
     Has: Luck, SOS-AdjEM, SOS-OppO, SOS-OppD, NCSOS-AdjEM

The two are merged on (season, team) to produce a complete feature set.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import KENPOM_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

# ── Column mapping: raw Pilafas → standardized names ─────────────────────
DEV_COLUMN_MAP = {
    "Season": "season",
    "Mapped ESPN Team Name": "team",
    "Full Team Name": "full_team_name",
    "Short Conference Name": "conf",
    "Mapped Conference Name": "conf_full",
    # Efficiency (pre-tournament preferred, fallback to final)
    "Pre-Tournament.AdjEM": "adjem",
    "Pre-Tournament.AdjOE": "adjoe",
    "Pre-Tournament.AdjDE": "adjde",
    "Pre-Tournament.AdjTempo": "adjtempo",
    "Pre-Tournament.RankAdjEM": "rank",
    # Final-season efficiency (used as fallback)
    "AdjEM": "adjem_final",
    "AdjOE": "adjoe_final",
    "AdjDE": "adjde_final",
    "AdjTempo": "adjtempo_final",
    "RankAdjEM": "rank_final",
    # Four factors — offense
    "eFGPct": "off_efgpct",
    "TOPct": "off_topct",
    "ORPct": "off_orpct",
    "FTRate": "off_ftrate",
    # Misc stats
    "FG2Pct": "off_fg2pct",
    "FG3Pct": "off_fg3pct",
    "FTPct": "off_ftpct",
    "BlockPct": "off_blockpct",
    "OppFG2Pct": "def_fg2pct",
    "OppFG3Pct": "def_fg3pct",
    "OppFTPct": "def_ftpct",
    "OppBlockPct": "def_blockpct",
    "FG3Rate": "off_fg3rate",
    "OppFG3Rate": "def_fg3rate",
    "ARate": "off_arate",
    "OppARate": "def_arate",
    "StlRate": "off_stlrate",
    "OppStlRate": "def_stlrate",
    "NSTRate": "off_nstrate",
    "OppNSTRate": "def_nstrate",
    # Height and experience
    "AvgHeight": "avg_height",
    "EffectiveHeight": "eff_height",
    "Experience": "experience",
    "Bench": "bench",
    # Net rating
    "Net Rating": "net_rating",
    # Tournament info
    "Seed": "ncaa_seed",
    "Region": "ncaa_region",
    "Post-Season Tournament": "postseason",
}

# ── Pomeroy ratings column mapping ───────────────────────────────────────
POMEROY_COLUMN_MAP = {
    "Season": "season",
    "Team": "team_kenpom",
    "Conf": "conf_kenpom",
    "Luck": "luck",
    "SOS-AdjEM": "sos_adjem",
    "SOS-OppO": "sos_opp_o",
    "SOS-OppD": "sos_opp_d",
    "NCSOS-AdjEM": "ncsos_adjem",
}


def _parse_signed_float(val) -> float:
    """Parse KenPom's signed float strings like '+.026' or '-.013'."""
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_dev_table(kenpom_dir: Path = KENPOM_DIR) -> pd.DataFrame:
    """Load the main DEV combined table with standardized column names."""
    csv_path = kenpom_dir / "DEV _ March Madness.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"DEV table not found at {csv_path}")

    logger.info("Loading DEV table from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Rename mapped columns
    rename = {k: v for k, v in DEV_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Use pre-tournament metrics where available, fall back to final
    for metric in ["adjem", "adjoe", "adjde", "adjtempo"]:
        final_col = f"{metric}_final"
        if metric in df.columns and final_col in df.columns:
            df[metric] = df[metric].fillna(df[final_col])
        elif metric not in df.columns and final_col in df.columns:
            df[metric] = df[final_col]
    if "rank" in df.columns and "rank_final" in df.columns:
        df["rank"] = df["rank"].fillna(df["rank_final"])
    elif "rank" not in df.columns and "rank_final" in df.columns:
        df["rank"] = df["rank_final"]

    # Drop _final columns now that we've used them for fallback
    final_cols = [c for c in df.columns if c.endswith("_final")]
    df = df.drop(columns=final_cols, errors="ignore")

    # Keep only renamed columns
    keep = set(rename.values()) - {c for c in rename.values() if c.endswith("_final")}
    drop_cols = [c for c in df.columns if c not in keep]
    df = df.drop(columns=drop_cols, errors="ignore")

    df["season"] = df["season"].astype(int)
    df["team"] = df["team"].astype(str).str.strip()

    logger.info(
        "Loaded DEV table: %d rows, %d seasons (%d–%d), %d columns",
        len(df), df["season"].nunique(),
        df["season"].min(), df["season"].max(), len(df.columns),
    )
    return df


def load_pomeroy_ratings(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Load scraped pomeroy ratings (Luck, SOS, NCSOS) from cache.

    Returns DataFrame keyed by (season, team_kenpom) with:
        luck, sos_adjem, sos_opp_o, sos_opp_d, ncsos_adjem
    """
    parquet_path = cache_dir / "kenpom_pomeroy_ratings.parquet"
    if not parquet_path.exists():
        logger.warning(
            "Pomeroy ratings cache not found at %s. "
            "Run scraping script to populate. Luck/SOS/NCSOS will be missing.",
            parquet_path,
        )
        return pd.DataFrame()

    logger.info("Loading pomeroy ratings from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Rename columns
    rename = {k: v for k, v in POMEROY_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Parse signed float strings
    for col in ["luck", "sos_adjem", "sos_opp_o", "sos_opp_d", "ncsos_adjem"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_signed_float)

    # Keep only what we need
    keep_cols = ["season", "team_kenpom"] + [
        c for c in ["luck", "sos_adjem", "sos_opp_o", "sos_opp_d", "ncsos_adjem"]
        if c in df.columns
    ]
    df = df[keep_cols]
    df["team_kenpom"] = df["team_kenpom"].astype(str).str.strip()

    logger.info("Loaded pomeroy ratings: %d rows, %d seasons", len(df), df["season"].nunique())
    return df


def _normalize_team_name(name: str) -> str:
    """Normalize team name for matching between ESPN and KenPom conventions.

    Handles: "State" ↔ "St.", hyphens ↔ spaces, common abbreviations.
    """
    import re
    s = name.strip().lower()
    # Remove periods entirely: "St." → "St", "N.C." → "NC"
    s = s.replace(".", "")
    # "St" at word boundary → "State"
    s = re.sub(r"\bst\b", "state", s)
    # Remove hyphens and extra spaces
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Common abbreviations → full names
    replacements = {
        "uconn": "connecticut",
        "umass": "massachusetts",
        "csun": "cal state northridge",
        "fiu": "florida international",
        "fau": "florida atlantic",
        "utep": "texas el paso",
        "ucf": "central florida",
        "lsu": "louisiana state",
        "smu": "southern methodist",
        "vcu": "virginia commonwealth",
        "unlv": "nevada las vegas",
        "etsu": "east tennessee state",
        "utsa": "texas san antonio",
        "siue": "southern illinois edwardsville",
        "siu edwardsville": "southern illinois edwardsville",
        "ole miss": "mississippi",
        "nc state": "nc state",
    }
    for abbr, full in replacements.items():
        if s == abbr or s.startswith(abbr + " "):
            s = s.replace(abbr, full, 1)
            break
    return s


def _fuzzy_match_pomeroy(dev_df: pd.DataFrame, pom_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pomeroy ratings onto DEV table by matching team names.

    The DEV table uses ESPN names ('team') while pomeroy uses KenPom names ('team_kenpom').
    Uses normalized name matching to handle "State" vs "St." etc.
    """
    if pom_df.empty:
        return dev_df

    stat_cols = [c for c in pom_df.columns if c not in ("season", "team_kenpom")]

    pom = pom_df.copy()
    pom["_key"] = pom["team_kenpom"].apply(_normalize_team_name)

    dev = dev_df.copy()
    dev["_key"] = dev["team"].apply(_normalize_team_name)

    merged = dev.merge(
        pom[["season", "_key"] + stat_cols],
        on=["season", "_key"],
        how="left",
    )

    n_matched = merged[stat_cols[0]].notna().sum() if stat_cols else 0
    n_total = len(merged)
    n_missing = n_total - n_matched

    if n_missing > 0:
        # Log a sample of unmatched teams for debugging
        missing_teams = merged.loc[merged[stat_cols[0]].isna(), "team"].unique()
        logger.info(
            "Pomeroy merge: %d/%d matched. %d unmatched (mostly non-tournament): %s",
            n_matched, n_total, n_missing, list(missing_teams[:10]),
        )

    merged = merged.drop(columns=["_key"])
    return merged


def load_kenpom(kenpom_dir: Path = KENPOM_DIR) -> pd.DataFrame:
    """Load all KenPom data into a single DataFrame keyed by (season, team).

    Combines:
    1. DEV table (Pilafas) — efficiency, four factors, misc stats
    2. Pomeroy ratings (scraped) — Luck, SOS, NCSOS

    Returns:
        DataFrame with one row per team per season, standardized column names.
    """
    df = load_dev_table(kenpom_dir)

    # Merge in Luck/SOS/NCSOS from scraped pomeroy ratings
    pom = load_pomeroy_ratings()
    df = _fuzzy_match_pomeroy(df, pom)

    # Deduplicate: some teams appear twice in the Pilafas data (e.g. South
    # Dakota St 2017).  Keep first occurrence so downstream merges stay 1-to-1.
    n_before = len(df)
    df = df.drop_duplicates(subset=["season", "team"], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d duplicate (season, team) rows", n_dropped)

    df = df.sort_values(["season", "team"]).reset_index(drop=True)

    logger.info(
        "Final KenPom dataset: %d rows, %d columns, seasons %d–%d",
        len(df), len(df.columns),
        df["season"].min(), df["season"].max(),
    )
    return df
