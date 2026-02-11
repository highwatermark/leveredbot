"""
Database schema and helpers for the leveraged ETF strategy.

Uses raw SQL with sqlite3 for simplicity. Three tables:
- decisions: daily strategy decisions with all signals
- regimes: regime transition log
- performance: daily P&L tracking
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path

import pytz

from config import DB_PATH

ET = pytz.timezone("America/New_York")


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db(conn: sqlite3.Connection | None = None):
    """Context manager for DB connections. Commits on success, closes if we opened it."""
    should_close = conn is None
    if should_close:
        conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if should_close:
            conn.close()


def init_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create all tables if they don't exist."""
    with get_db(conn) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            timestamp TEXT NOT NULL,

            qqq_close REAL,
            qqq_sma_50 REAL,
            qqq_sma_250 REAL,
            qqq_pct_above_sma50 REAL,
            qqq_pct_above_sma250 REAL,
            qqq_roc_5 REAL,
            qqq_roc_20 REAL,

            realized_vol_20d REAL,
            vol_regime TEXT,

            options_flow_put_premium REAL,
            options_flow_call_premium REAL,
            options_flow_ratio REAL,
            options_flow_bearish INTEGER DEFAULT 0,
            options_flow_adjustment REAL DEFAULT 1.0,

            regime TEXT,
            regime_changed INTEGER DEFAULT 0,
            previous_regime TEXT,

            momentum_score REAL,
            momentum_factor REAL,

            gates_passed INTEGER,
            gates_failed TEXT,

            target_allocation_pct REAL,
            target_dollar_value REAL,
            target_shares INTEGER,
            current_shares INTEGER,
            order_action TEXT,
            order_shares INTEGER,
            order_value REAL,

            order_id TEXT,
            fill_price REAL,
            fill_time TEXT,
            execution_window TEXT,

            account_equity REAL,
            allocated_capital REAL,
            tqqq_position_value REAL,
            tqqq_pnl_pct REAL,
            other_positions_value REAL,
            cash_balance REAL,
            day_trades_remaining INTEGER,

            trading_days_fetched INTEGER,
            is_half_day INTEGER DEFAULT 0,

            knn_direction TEXT DEFAULT 'FLAT',
            knn_confidence REAL DEFAULT 0.5,
            knn_adjustment REAL DEFAULT 1.0,

            symbol TEXT DEFAULT 'TQQQ',
            sqqq_position_value REAL DEFAULT 0,
            sqqq_pnl_pct REAL DEFAULT 0,

            status TEXT DEFAULT 'COMPLETE'
        );

        CREATE TABLE IF NOT EXISTS regimes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            old_regime TEXT,
            new_regime TEXT,
            qqq_close REAL,
            qqq_sma_50 REAL,
            qqq_sma_250 REAL,
            trigger_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            tqqq_shares INTEGER,
            tqqq_avg_cost REAL,
            tqqq_current_price REAL,
            tqqq_position_value REAL,
            tqqq_pnl_day REAL,
            tqqq_pnl_total REAL,
            tqqq_pnl_pct REAL,
            regime TEXT,
            allocated_capital REAL,
            realized_vol REAL,
            benchmark_qqq_pct REAL,
            strategy_total_return_pct REAL,
            sqqq_shares INTEGER DEFAULT 0,
            sqqq_position_value REAL DEFAULT 0,
            sqqq_pnl_pct REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS pregame (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            timestamp TEXT NOT NULL,

            -- Flow sentiment (aggregated from multiple polls)
            flow_samples INTEGER DEFAULT 0,
            flow_put_premium_total REAL DEFAULT 0,
            flow_call_premium_total REAL DEFAULT 0,
            flow_avg_ratio REAL DEFAULT 1.0,
            flow_trend TEXT,
            flow_bearish_samples INTEGER DEFAULT 0,

            -- Intraday QQQ movement
            qqq_open REAL,
            qqq_current REAL,
            qqq_intraday_pct REAL,
            qqq_intraday_high REAL,
            qqq_intraday_low REAL,
            qqq_intraday_range_pct REAL,

            -- TQQQ intraday
            tqqq_open REAL,
            tqqq_current REAL,
            tqqq_intraday_pct REAL,

            -- Volume analysis
            qqq_volume INTEGER,
            qqq_avg_volume INTEGER,
            qqq_relative_volume REAL,

            -- Late-day momentum (last hour direction)
            qqq_last_hour_pct REAL,
            selling_into_close INTEGER DEFAULT 0,

            -- Summary
            pregame_sentiment TEXT,
            pregame_notes TEXT
        );

        CREATE TABLE IF NOT EXISTS position_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            window TEXT NOT NULL,
            event_type TEXT NOT NULL,
            shares_before INTEGER,
            shares_sold INTEGER,
            shares_after INTEGER,
            price REAL,
            pnl_pct REAL,
            order_id TEXT,
            order_status TEXT,
            trigger_value REAL,
            trigger_detail TEXT,
            profit_tier_pct REAL,
            high_watermark REAL
        );

        CREATE TABLE IF NOT EXISTS position_watermarks (
            symbol TEXT PRIMARY KEY,
            high_price REAL NOT NULL,
            high_date TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            qqq_close REAL,
            tqqq_close REAL,
            regime TEXT,
            target_shares INTEGER,
            held_shares INTEGER,
            portfolio_value REAL,
            cash REAL,
            pnl_day REAL,
            pnl_total_pct REAL,
            drawdown_pct REAL,
            qqq_buy_hold_pct REAL,
            tqqq_buy_hold_pct REAL
        );
    """)
        # Migration: add status column for existing databases
        try:
            c.execute("SELECT status FROM decisions LIMIT 0")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE decisions ADD COLUMN status TEXT DEFAULT 'COMPLETE'")

        # Migration: add k-NN columns for existing databases
        for col, default in [
            ("knn_direction", "'FLAT'"),
            ("knn_confidence", "0.5"),
            ("knn_adjustment", "1.0"),
        ]:
            try:
                c.execute(f"SELECT {col} FROM decisions LIMIT 0")
            except sqlite3.OperationalError:
                c.execute(f"ALTER TABLE decisions ADD COLUMN {col} TEXT DEFAULT {default}"
                          if col == "knn_direction" else
                          f"ALTER TABLE decisions ADD COLUMN {col} REAL DEFAULT {default}")

        # Migration: add SQQQ columns for existing databases
        for table, cols in [
            ("decisions", [
                ("symbol", "TEXT", "'TQQQ'"),
                ("sqqq_position_value", "REAL", "0"),
                ("sqqq_pnl_pct", "REAL", "0"),
            ]),
            ("performance", [
                ("sqqq_shares", "INTEGER", "0"),
                ("sqqq_position_value", "REAL", "0"),
                ("sqqq_pnl_pct", "REAL", "0"),
            ]),
        ]:
            for col, col_type, default in cols:
                try:
                    c.execute(f"SELECT {col} FROM {table} LIMIT 0")
                except sqlite3.OperationalError:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}")


def log_daily_decision(data: dict, conn: sqlite3.Connection | None = None) -> int:
    """Insert a daily decision record. Returns the row ID."""
    with get_db(conn) as c:
        # Ensure gates_failed is JSON string
        if "gates_failed" in data and isinstance(data["gates_failed"], list):
            data["gates_failed"] = json.dumps(data["gates_failed"])

        columns = list(data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)

        cursor = c.execute(
            f"INSERT INTO decisions ({col_str}) VALUES ({placeholders})",
            [data[col] for col in columns],
        )
        return cursor.lastrowid


def update_decision(decision_id: int, updates: dict, conn: sqlite3.Connection | None = None) -> None:
    """Update an existing decision record (e.g., after trade execution)."""
    with get_db(conn) as c:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [decision_id]
        c.execute(
            f"UPDATE decisions SET {set_clause} WHERE id = ?",
            values,
        )


def log_regime_change(
    old: str | None, new: str, data: dict, conn: sqlite3.Connection | None = None
) -> None:
    """Log a regime transition."""
    with get_db(conn) as c:
        c.execute(
            "INSERT INTO regimes (date, old_regime, new_regime, qqq_close, qqq_sma_50, qqq_sma_250, trigger_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                data.get("date", datetime.now(ET).date().isoformat()),
                old,
                new,
                data.get("qqq_close"),
                data.get("qqq_sma_50"),
                data.get("qqq_sma_250"),
                data.get("trigger_reason", ""),
            ],
        )


def log_daily_performance(data: dict, conn: sqlite3.Connection | None = None) -> None:
    """Insert a daily performance record."""
    with get_db(conn) as c:
        columns = list(data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)

        c.execute(
            f"INSERT INTO performance ({col_str}) VALUES ({placeholders})",
            [data[col] for col in columns],
        )


def get_last_regime(conn: sqlite3.Connection | None = None) -> str | None:
    """Get the most recent regime from the decisions table."""
    with get_db(conn) as c:
        row = c.execute(
            "SELECT regime FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["regime"] if row else None


def get_regime_duration_days(conn: sqlite3.Connection | None = None) -> int:
    """How many days the current regime has been active."""
    with get_db(conn) as c:
        row = c.execute(
            "SELECT date FROM regimes ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if not row:
            return 999  # No regime changes recorded, treat as long-standing

        change_date = date.fromisoformat(row["date"])
        return (datetime.now(ET).date() - change_date).days


def get_consecutive_losing_days(conn: sqlite3.Connection | None = None) -> int:
    """Count consecutive days where TQQQ position P&L was negative."""
    with get_db(conn) as c:
        rows = c.execute(
            "SELECT tqqq_pnl_day FROM performance ORDER BY id DESC LIMIT 30"
        ).fetchall()

        count = 0
        for row in rows:
            if row["tqqq_pnl_day"] is not None and row["tqqq_pnl_day"] < 0:
                count += 1
            else:
                break
        return count


def get_strategy_state(conn: sqlite3.Connection | None = None) -> dict:
    """Get regime, duration, and losing days in a single DB pass.

    Replaces 3 separate calls: get_last_regime, get_regime_duration_days,
    get_consecutive_losing_days.
    """
    with get_db(conn) as c:
        # Last regime from decisions
        regime_row = c.execute(
            "SELECT regime FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_regime = regime_row["regime"] if regime_row else None

        # Duration from regimes
        duration_row = c.execute(
            "SELECT date FROM regimes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if duration_row:
            change_date = date.fromisoformat(duration_row["date"])
            regime_duration = (datetime.now(ET).date() - change_date).days
        else:
            regime_duration = 999

        # Consecutive losing days from performance
        perf_rows = c.execute(
            "SELECT tqqq_pnl_day FROM performance ORDER BY id DESC LIMIT 30"
        ).fetchall()
        losing_days = 0
        for row in perf_rows:
            if row["tqqq_pnl_day"] is not None and row["tqqq_pnl_day"] < 0:
                losing_days += 1
            else:
                break

        return {
            "last_regime": last_regime,
            "regime_duration_days": regime_duration,
            "consecutive_losing_days": losing_days,
        }


def get_regime_history(days: int = 30, conn: sqlite3.Connection | None = None) -> list[dict]:
    """Get regime transitions from the last N days."""
    with get_db(conn) as c:
        cutoff = (datetime.now(ET).date() - timedelta(days=days)).isoformat()
        rows = c.execute(
            "SELECT date, old_regime, new_regime, qqq_close, qqq_sma_50, qqq_sma_250, trigger_reason "
            "FROM regimes WHERE date >= ? ORDER BY id DESC",
            [cutoff],
        ).fetchall()
        return [dict(r) for r in rows]


def get_position_entry_date(conn: sqlite3.Connection | None = None) -> str | None:
    """Get the date when the current TQQQ position was opened."""
    with get_db(conn) as c:
        # Look for the most recent transition from 0 shares to >0 shares
        rows = c.execute(
            "SELECT date, current_shares FROM decisions ORDER BY id DESC LIMIT 60"
        ).fetchall()

        if not rows:
            return None

        # Walk backwards to find when shares went from 0 to >0
        entry_date = None
        for row in rows:
            shares = row["current_shares"] or 0
            if shares > 0:
                entry_date = row["date"]
            else:
                break

        return entry_date


def get_performance_summary(
    days: int = 30, conn: sqlite3.Connection | None = None
) -> dict:
    """Get aggregated performance stats over the last N days."""
    with get_db(conn) as c:
        rows = c.execute(
            "SELECT * FROM performance ORDER BY id DESC LIMIT ?", [days]
        ).fetchall()

        if not rows:
            return {"days": 0, "total_pnl": 0, "avg_daily_pnl": 0, "best_day": 0, "worst_day": 0}

        pnls = [r["tqqq_pnl_day"] or 0 for r in rows]
        return {
            "days": len(rows),
            "total_pnl": sum(pnls),
            "avg_daily_pnl": sum(pnls) / len(pnls),
            "best_day": max(pnls),
            "worst_day": min(pnls),
            "latest_total_return_pct": rows[0]["strategy_total_return_pct"] or 0,
        }


def save_pregame(data: dict, conn: sqlite3.Connection | None = None) -> None:
    """Save pregame intelligence report."""
    with get_db(conn) as c:
        columns = list(data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)

        c.execute(
            f"INSERT INTO pregame ({col_str}) VALUES ({placeholders})",
            [data[col] for col in columns],
        )


def get_today_pregame(conn: sqlite3.Connection | None = None) -> dict | None:
    """Get today's pregame report if it exists."""
    with get_db(conn) as c:
        today_str = datetime.now(ET).date().isoformat()
        row = c.execute(
            "SELECT * FROM pregame WHERE date = ? ORDER BY id DESC LIMIT 1",
            [today_str],
        ).fetchone()
        return dict(row) if row else None


# ── Position Manager helpers ──


def save_position_event(data: dict, conn: sqlite3.Connection | None = None) -> int:
    """Insert a position event record. Returns the row ID."""
    with get_db(conn) as c:
        columns = list(data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        cursor = c.execute(
            f"INSERT INTO position_events ({col_str}) VALUES ({placeholders})",
            [data[col] for col in columns],
        )
        return cursor.lastrowid


def update_position_event(event_id: int, updates: dict, conn: sqlite3.Connection | None = None) -> None:
    """Update an existing position event record."""
    with get_db(conn) as c:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [event_id]
        c.execute(
            f"UPDATE position_events SET {set_clause} WHERE id = ?",
            values,
        )


def get_today_position_events(symbol: str, conn: sqlite3.Connection | None = None) -> list[dict]:
    """Get all position events for a symbol today."""
    with get_db(conn) as c:
        today_str = datetime.now(ET).date().isoformat()
        rows = c.execute(
            "SELECT * FROM position_events WHERE symbol = ? AND date = ? ORDER BY id",
            [symbol, today_str],
        ).fetchall()
        return [dict(r) for r in rows]


def get_position_watermark(symbol: str, conn: sqlite3.Connection | None = None) -> dict | None:
    """Get the high watermark for a symbol."""
    with get_db(conn) as c:
        row = c.execute(
            "SELECT * FROM position_watermarks WHERE symbol = ?",
            [symbol],
        ).fetchone()
        return dict(row) if row else None


def update_position_watermark(symbol: str, high_price: float, high_date: str, conn: sqlite3.Connection | None = None) -> None:
    """Upsert the high watermark for a symbol."""
    with get_db(conn) as c:
        now_str = datetime.now(ET).isoformat()
        c.execute(
            "INSERT INTO position_watermarks (symbol, high_price, high_date, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET high_price = ?, high_date = ?, updated_at = ?",
            [symbol, high_price, high_date, now_str, high_price, high_date, now_str],
        )


def get_profit_tiers_taken(symbol: str, since_date: str, conn: sqlite3.Connection | None = None) -> list[float]:
    """Get profit tier percentages already taken for a symbol since a date."""
    with get_db(conn) as c:
        rows = c.execute(
            "SELECT profit_tier_pct FROM position_events "
            "WHERE symbol = ? AND date >= ? AND event_type = 'PARTIAL_PROFIT' AND order_status = 'EXECUTED'",
            [symbol, since_date],
        ).fetchall()
        return [r["profit_tier_pct"] for r in rows if r["profit_tier_pct"] is not None]
