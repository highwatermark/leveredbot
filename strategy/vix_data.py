"""
VIX data fetcher with append-only local cache.

Downloads ^VIX daily close from Yahoo Finance, stores in a JSON file,
and only fetches new dates on subsequent calls.

Usage:
    vix = get_vix_data()           # dict[str, float] date→close
    vix_close = vix.get("2025-01-15", None)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from config import DATA_DIR

logger = logging.getLogger(__name__)

VIX_CACHE_PATH = DATA_DIR / "vix_cache.json"

ET = pytz.timezone("America/New_York")


def get_vix_data(lookback_days: int = 500) -> dict[str, float]:
    """
    Get VIX daily close data as a date→close mapping.

    Loads from local cache, fetches only missing dates from Yahoo Finance,
    appends new data, and saves back.

    Args:
        lookback_days: How far back to fetch on first run (default 500 ~2yr).

    Returns:
        Dict mapping ISO date strings to VIX close values.
        Empty dict if all fetches fail.
    """
    cache = _load_cache()
    today = datetime.now(ET).date()

    if cache:
        # Find the last cached date and only fetch from there
        last_date = max(cache.keys())
        start = (datetime.fromisoformat(last_date) + timedelta(days=1)).date()
    else:
        # First time: fetch full history
        start = today - timedelta(days=lookback_days)

    if start >= today:
        return cache  # Already up to date

    new_data = _fetch_vix(start.isoformat(), today.isoformat())
    if new_data:
        cache.update(new_data)
        _save_cache(cache)
        logger.info(f"VIX cache updated: {len(new_data)} new dates (total {len(cache)})")

    return cache


def _fetch_vix(start: str, end: str) -> dict[str, float]:
    """Fetch VIX data from Yahoo Finance."""
    try:
        import yfinance as yf

        vix = yf.download("^VIX", start=start, end=end, progress=False)
        if vix.empty:
            logger.warning("yfinance returned empty VIX data")
            return {}

        result = {}
        for idx, row in vix.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            # yfinance may return MultiIndex columns; handle both cases
            close_val = row["Close"]
            if hasattr(close_val, "item"):
                close_val = close_val.item()
            result[date_str] = round(float(close_val), 2)
        return result

    except Exception as e:
        logger.warning(f"VIX fetch failed (non-fatal): {e}")
        return {}


def _load_cache() -> dict[str, float]:
    """Load VIX cache from disk."""
    if VIX_CACHE_PATH.exists():
        try:
            with open(VIX_CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"VIX cache read failed: {e}")
    return {}


def _save_cache(data: dict[str, float]) -> None:
    """Save VIX cache to disk."""
    try:
        VIX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VIX_CACHE_PATH, "w") as f:
            json.dump(data, f, sort_keys=True)
    except IOError as e:
        logger.warning(f"VIX cache write failed: {e}")
