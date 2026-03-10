"""Core features: higher-seed minus lower-seed KenPom metric deltas."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that should NOT produce deltas (outcome-leaking or identity)
_BLACKLIST = {"score", "seed", "team_id"}


def detect_stat_pairs(df: pd.DataFrame) -> list[str]:
    """Find stat base names that have both _a and _b numeric columns.

    Excludes blacklist: score, seed, team_id (outcome-leaking or identity).

    Returns:
        Sorted list of base names.
    """
    a_cols = {c[:-2] for c in df.columns if c.endswith("_a") and pd.api.types.is_numeric_dtype(df[c])}
    b_cols = {c[:-2] for c in df.columns if c.endswith("_b") and pd.api.types.is_numeric_dtype(df[c])}
    pairs = sorted(a_cols & b_cols - _BLACKLIST)
    logger.info("Detected %d stat pairs for deltas: %s", len(pairs), pairs)
    return pairs


def compute_deltas(
    df: pd.DataFrame, stat_names: list[str] | None = None
) -> pd.DataFrame:
    """Add '{stat}_delta' columns (team_a - team_b) for each detected stat pair.

    NaN in either _a or _b propagates to NaN in delta (XGBoost handles natively).
    After computing deltas, also adds the expected_margin interaction feature.

    Args:
        df: Merged DataFrame with _a/_b columns.
        stat_names: Optional explicit list of base stat names. Auto-detected if None.

    Returns:
        Copy of df with delta columns added. Original _a/_b columns preserved.
    """
    out = df.copy()

    if stat_names is None:
        stat_names = detect_stat_pairs(out)

    for stat in stat_names:
        out[f"{stat}_delta"] = out[f"{stat}_a"] - out[f"{stat}_b"]

    logger.info("Computed %d delta columns", len(stat_names))

    # Add derived interaction feature
    out = compute_expected_margin(out)

    return out


def compute_expected_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Add expected_margin: tempo-adjusted predicted point spread.

    expected_margin = adjem_delta * (adjtempo_a + adjtempo_b) / 200

    Captures how efficiency gaps translate to actual point margins based on
    game pace. Helps the tiny gating network which can't learn interactions.

    Dividing by 200 (not 100) because we average two tempos then scale
    per-100-possessions.
    """
    if "adjem_delta" not in df.columns:
        logger.warning("adjem_delta not found — skipping expected_margin")
        return df

    df["expected_margin"] = (
        df["adjem_delta"] * (df["adjtempo_a"] + df["adjtempo_b"]) / 200
    )
    logger.info("Added expected_margin feature")
    return df
