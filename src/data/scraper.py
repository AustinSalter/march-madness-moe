"""Wraps kenpompy to scrape 2026 KenPom data. Caches to data/cache/.

Requires KenPom subscription ($20/year). Credentials in .env:
  KENPOM_EMAIL=...
  KENPOM_PASSWORD=...

Only scrapes the current (2026) season — historical data comes from Kaggle.
"""

import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "kenpom_2026.parquet"


def scrape_2026(use_cache: bool = True) -> pd.DataFrame:
    """Scrape 2026 KenPom data via kenpompy.

    Args:
        use_cache: If True and cache exists, load from cache instead of scraping.

    Returns:
        DataFrame with same schema as kaggle_loader.load_kenpom() output,
        but for the 2026 season only.
    """
    if use_cache and CACHE_FILE.exists():
        logger.info("Loading cached 2026 KenPom data from %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    import os
    load_dotenv()

    email = os.environ.get("KENPOM_EMAIL")
    password = os.environ.get("KENPOM_PASSWORD")
    if not email or not password:
        raise ValueError(
            "KENPOM_EMAIL and KENPOM_PASSWORD must be set in .env file. "
            "KenPom subscription required ($20/year)."
        )

    try:
        import kenpompy.summary as kp_summary
        import kenpompy.misc as kp_misc
        from kenpompy.utils import login
    except ImportError:
        raise ImportError("kenpompy is required for scraping. Install with: pip install kenpompy")

    logger.info("Logging into KenPom...")
    browser = login(email, password)

    logger.info("Scraping 2026 efficiency ratings...")
    eff_df = kp_summary.get_efficiency(browser, season=2026)

    logger.info("Scraping 2026 four factors...")
    ff_df = kp_summary.get_fourfactors(browser, season=2026)

    logger.info("Scraping 2026 team stats...")
    try:
        misc_df = kp_misc.get_teamstats(browser, season=2026)
    except Exception:
        logger.warning("Could not scrape misc team stats, continuing without them")
        misc_df = None

    # Normalize to match kaggle_loader schema
    # (Exact normalization will depend on kenpompy output format —
    #  adjust column mappings here after first successful scrape)
    result = eff_df.copy()
    result["season"] = 2026

    # Cache result
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(CACHE_FILE, index=False)
    logger.info("Cached 2026 data to %s", CACHE_FILE)

    return result
