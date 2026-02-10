"""
Local bar cache for QQQ/TQQQ daily bars.

Stores fetched bars in SQLite to avoid redundant API calls.
Daily runs fetch only the latest day; backtest loads full history once.
"""

import sqlite3
from datetime import date, datetime, timedelta

import pytz

from config import DB_PATH
from db.models import get_connection

ET = pytz.timezone("America/New_York")


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
    symbol: str, days: int, conn: sqlite3.Connection | None = None
) -> list[dict]:
    """Read cached bars for a symbol, most recent N days."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    _ensure_cache_table(conn)

    rows = conn.execute(
        "SELECT * FROM bar_cache WHERE symbol = ? ORDER BY date DESC LIMIT ?",
        [symbol, days],
    ).fetchall()

    if close_after:
        conn.close()

    # Return in chronological order
    return [dict(r) for r in reversed(rows)]


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

    # Now read from cache
    result = get_cached_bars(symbol, days * 2, conn)  # Over-fetch to ensure enough

    if close_after:
        conn.close()

    # Filter to requested range
    start_str = start_date.isoformat()
    return [b for b in result if b["date"] >= start_str]
