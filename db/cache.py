"""
Local bar cache for QQQ/TQQQ daily bars + microstructure features.

Stores fetched bars in SQLite to avoid redundant API calls.
Daily runs fetch only the latest day; backtest loads full history once.

Microstructure cache stores pre-computed intraday features (~900 rows)
instead of raw intraday bars (~70K rows).
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta

import pytz

from config import DB_PATH
from db.models import get_connection

ET = pytz.timezone("America/New_York")

logger = logging.getLogger(__name__)


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bar_cache (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()


def get_cached_bars(
    symbol: str,
    days: int,
    conn: sqlite3.Connection | None = None,
    start_date: str | None = None,
) -> list[dict]:
    """Read cached bars for a symbol. Uses date filter when provided, else most recent N days."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    _ensure_cache_table(conn)

    if start_date:
        rows = conn.execute(
            "SELECT * FROM bar_cache WHERE symbol = ? AND date >= ? ORDER BY date ASC",
            [symbol, start_date],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM bar_cache WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            [symbol, days],
        ).fetchall()
        rows = list(reversed(rows))

    if close_after:
        conn.close()

    return [dict(r) for r in rows]


def get_cached_date_range(
    symbol: str, conn: sqlite3.Connection | None = None
) -> tuple[str | None, str | None]:
    """Get the min and max dates in cache for a symbol."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    _ensure_cache_table(conn)

    row = conn.execute(
        "SELECT MIN(date) as min_date, MAX(date) as max_date FROM bar_cache WHERE symbol = ?",
        [symbol],
    ).fetchone()

    if close_after:
        conn.close()

    if row and row["min_date"]:
        return row["min_date"], row["max_date"]
    return None, None


def update_cache(
    symbol: str, bars: list[dict], conn: sqlite3.Connection | None = None
) -> int:
    """Upsert bars into cache. Returns number of rows upserted."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    _ensure_cache_table(conn)

    if not bars:
        if close_after:
            conn.close()
        return 0

    conn.executemany(
        "INSERT OR REPLACE INTO bar_cache (symbol, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (symbol, b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"])
            for b in bars
        ],
    )
    conn.commit()

    count = len(bars)
    if close_after:
        conn.close()
    return count


def get_bars_with_cache(
    symbol: str,
    days: int,
    fetch_fn,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """
    Get bars using cache, fetching only missing days from Alpaca.

    fetch_fn(symbol, start_date, end_date) -> list[dict] is called to
    get any bars not already in the cache.
    """
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    _ensure_cache_table(conn)

    today = datetime.now(ET).date()
    start_date = today - timedelta(days=days)

    # Check what we have cached
    min_cached, max_cached = get_cached_date_range(symbol, conn)

    if min_cached is None:
        # Nothing cached — fetch everything
        bars = fetch_fn(symbol, start_date.isoformat(), today.isoformat())
        update_cache(symbol, bars, conn)
    else:
        # Fetch anything newer than our latest cached date
        max_date = date.fromisoformat(max_cached)
        if max_date < today - timedelta(days=1):
            fetch_start = max_date + timedelta(days=1)
            new_bars = fetch_fn(symbol, fetch_start.isoformat(), today.isoformat())
            update_cache(symbol, new_bars, conn)

        # Also check if we need older data
        min_date = date.fromisoformat(min_cached)
        if min_date > start_date:
            old_bars = fetch_fn(symbol, start_date.isoformat(), (min_date - timedelta(days=1)).isoformat())
            update_cache(symbol, old_bars, conn)

    # Read from cache with SQL date filter
    result = get_cached_bars(symbol, days, conn, start_date=start_date.isoformat())

    if close_after:
        conn.close()

    return result


# ─── Microstructure cache ───────────────────────────────────────────────

def _ensure_microstructure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS microstructure_cache (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            last_hour_volume_ratio REAL DEFAULT 0.0,
            vwap_deviation REAL DEFAULT 0.0,
            closing_momentum REAL DEFAULT 0.0,
            volume_acceleration REAL DEFAULT 0.0,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()


def get_cached_microstructure(
    symbol: str,
    start_date: str,
    conn: sqlite3.Connection,
) -> dict[str, dict[str, float]]:
    """Read cached microstructure features. Returns date->features dict."""
    _ensure_microstructure_table(conn)
    rows = conn.execute(
        "SELECT date, last_hour_volume_ratio, vwap_deviation, "
        "closing_momentum, volume_acceleration "
        "FROM microstructure_cache WHERE symbol = ? AND date >= ? ORDER BY date ASC",
        [symbol, start_date],
    ).fetchall()
    result = {}
    for r in rows:
        result[r["date"]] = {
            "last_hour_volume_ratio": r["last_hour_volume_ratio"],
            "vwap_deviation": r["vwap_deviation"],
            "closing_momentum": r["closing_momentum"],
            "volume_acceleration": r["volume_acceleration"],
        }
    return result


def update_microstructure_cache(
    symbol: str,
    features_by_date: dict[str, dict[str, float]],
    conn: sqlite3.Connection,
) -> int:
    """Upsert microstructure features into cache. Returns rows upserted."""
    _ensure_microstructure_table(conn)
    if not features_by_date:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO microstructure_cache "
        "(symbol, date, last_hour_volume_ratio, vwap_deviation, closing_momentum, volume_acceleration) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                symbol, dt,
                feats.get("last_hour_volume_ratio", 0.0),
                feats.get("vwap_deviation", 0.0),
                feats.get("closing_momentum", 0.0),
                feats.get("volume_acceleration", 0.0),
            )
            for dt, feats in features_by_date.items()
        ],
    )
    conn.commit()
    return len(features_by_date)


def get_microstructure_with_cache(
    symbol: str,
    days: int,
    intraday_fetch_fn,
    conn: sqlite3.Connection | None = None,
) -> dict[str, dict[str, float]]:
    """
    Get microstructure features using cache, computing only missing dates.

    1. Reads cached dates from microstructure_cache
    2. Reads all dates from bar_cache for the same symbol/range
    3. Identifies missing dates (in bar_cache but not microstructure_cache)
    4. Fetches intraday bars only for missing dates (chunked by month)
    5. Groups intraday bars by date, computes features per day
    6. Caches results, returns full date->features dict

    Args:
        symbol: e.g. "QQQ"
        days: calendar days of history
        intraday_fetch_fn: callable(symbol, start, end) -> list[dict]
        conn: SQLite connection

    Returns:
        Dict mapping date string -> {feature_name: value}
    """
    from strategy.microstructure import compute_microstructure_features

    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    try:
        _ensure_cache_table(conn)
        _ensure_microstructure_table(conn)

        today = datetime.now(ET).date()
        start_date = today - timedelta(days=days)
        start_str = start_date.isoformat()

        # Get already-cached microstructure features
        cached = get_cached_microstructure(symbol, start_str, conn)

        # Get all daily bar dates we have in cache (these are the trading days)
        bar_rows = conn.execute(
            "SELECT DISTINCT date FROM bar_cache WHERE symbol = ? AND date >= ? ORDER BY date ASC",
            [symbol, start_str],
        ).fetchall()
        bar_dates = {r["date"] for r in bar_rows}

        # Don't compute for today (intraday bars incomplete)
        today_str = today.isoformat()
        bar_dates.discard(today_str)

        # Find missing dates
        missing_dates = sorted(bar_dates - set(cached.keys()))

        if missing_dates:
            logger.info(f"Computing microstructure for {len(missing_dates)} missing dates ({symbol})")

            # Chunk by month to avoid huge API calls
            chunks = []
            current_chunk = [missing_dates[0]]
            for d in missing_dates[1:]:
                if d[:7] == current_chunk[0][:7]:
                    current_chunk.append(d)
                else:
                    chunks.append(current_chunk)
                    current_chunk = [d]
            chunks.append(current_chunk)

            new_features = {}
            for chunk in chunks:
                chunk_start = chunk[0]
                # End date = day after last date in chunk to ensure inclusion
                chunk_end_date = date.fromisoformat(chunk[-1]) + timedelta(days=1)
                chunk_end = chunk_end_date.isoformat()

                try:
                    intraday_bars = intraday_fetch_fn(symbol, chunk_start, chunk_end)
                except Exception as e:
                    logger.warning(f"Intraday fetch failed for {symbol} {chunk_start}-{chunk_end}: {e}")
                    # Default all missing dates in this chunk to zeros
                    for d in chunk:
                        new_features[d] = {
                            "last_hour_volume_ratio": 0.0,
                            "vwap_deviation": 0.0,
                            "closing_momentum": 0.0,
                            "volume_acceleration": 0.0,
                        }
                    continue

                # Group intraday bars by date
                bars_by_date: dict[str, list[dict]] = {}
                for bar in intraday_bars:
                    d = bar.get("date", "")
                    if d in chunk:
                        bars_by_date.setdefault(d, []).append(bar)

                # Compute features per day
                for d in chunk:
                    day_bars = bars_by_date.get(d, [])
                    feats = compute_microstructure_features(day_bars)
                    new_features[d] = feats

            # Cache the new features
            if new_features:
                update_microstructure_cache(symbol, new_features, conn)
                cached.update(new_features)

        return cached

    finally:
        if close_after:
            conn.close()
