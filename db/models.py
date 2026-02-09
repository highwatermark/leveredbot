"""
Database schema and helpers for the leveraged ETF strategy.

Uses raw SQL with sqlite3 for simplicity. Three tables:
- decisions: daily strategy decisions with all signals
- regimes: regime transition log
- performance: daily P&L tracking
"""

import json
import sqlite3
from datetime import datetime, date
from pathlib import Path

from config import DB_PATH


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_tables(conn: sqlite3.Connection | None = None) -> None:
    """Create all tables if they don't exist."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    conn.executescript("""
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
            is_half_day INTEGER DEFAULT 0
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
            strategy_total_return_pct REAL
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
    conn.commit()

    if close_after:
        conn.close()


def log_daily_decision(data: dict, conn: sqlite3.Connection | None = None) -> None:
    """Insert a daily decision record."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    # Ensure gates_failed is JSON string
    if "gates_failed" in data and isinstance(data["gates_failed"], list):
        data["gates_failed"] = json.dumps(data["gates_failed"])

    columns = list(data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)

    conn.execute(
        f"INSERT INTO decisions ({col_str}) VALUES ({placeholders})",
        [data[c] for c in columns],
    )
    conn.commit()

    if close_after:
        conn.close()


def log_regime_change(
    old: str | None, new: str, data: dict, conn: sqlite3.Connection | None = None
) -> None:
    """Log a regime transition."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    conn.execute(
        "INSERT INTO regimes (date, old_regime, new_regime, qqq_close, qqq_sma_50, qqq_sma_250, trigger_reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            data.get("date", date.today().isoformat()),
            old,
            new,
            data.get("qqq_close"),
            data.get("qqq_sma_50"),
            data.get("qqq_sma_250"),
            data.get("trigger_reason", ""),
        ],
    )
    conn.commit()

    if close_after:
        conn.close()


def log_daily_performance(data: dict, conn: sqlite3.Connection | None = None) -> None:
    """Insert a daily performance record."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    columns = list(data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)

    conn.execute(
        f"INSERT INTO performance ({col_str}) VALUES ({placeholders})",
        [data[c] for c in columns],
    )
    conn.commit()

    if close_after:
        conn.close()


def get_last_regime(conn: sqlite3.Connection | None = None) -> str | None:
    """Get the most recent regime from the decisions table."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    row = conn.execute(
        "SELECT regime FROM decisions ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if close_after:
        conn.close()

    return row["regime"] if row else None


def get_regime_duration_days(conn: sqlite3.Connection | None = None) -> int:
    """How many days the current regime has been active."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    row = conn.execute(
        "SELECT date FROM regimes ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if close_after:
        conn.close()

    if not row:
        return 999  # No regime changes recorded, treat as long-standing

    change_date = date.fromisoformat(row["date"])
    return (date.today() - change_date).days


def get_consecutive_losing_days(conn: sqlite3.Connection | None = None) -> int:
    """Count consecutive days where TQQQ position P&L was negative."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    rows = conn.execute(
        "SELECT tqqq_pnl_day FROM performance ORDER BY id DESC LIMIT 30"
    ).fetchall()

    if close_after:
        conn.close()

    count = 0
    for row in rows:
        if row["tqqq_pnl_day"] is not None and row["tqqq_pnl_day"] < 0:
            count += 1
        else:
            break
    return count


def get_position_entry_date(conn: sqlite3.Connection | None = None) -> str | None:
    """Get the date when the current TQQQ position was opened."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    # Look for the most recent transition from 0 shares to >0 shares
    rows = conn.execute(
        "SELECT date, current_shares FROM decisions ORDER BY id DESC LIMIT 60"
    ).fetchall()

    if close_after:
        conn.close()

    if not rows:
        return None

    # Walk backwards to find when shares went from 0 to >0
    entry_date = None
    for i, row in enumerate(rows):
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
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    rows = conn.execute(
        "SELECT * FROM performance ORDER BY id DESC LIMIT ?", [days]
    ).fetchall()

    if close_after:
        conn.close()

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
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    columns = list(data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)

    conn.execute(
        f"INSERT INTO pregame ({col_str}) VALUES ({placeholders})",
        [data[c] for c in columns],
    )
    conn.commit()

    if close_after:
        conn.close()


def get_today_pregame(conn: sqlite3.Connection | None = None) -> dict | None:
    """Get today's pregame report if it exists."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    today_str = date.today().isoformat()
    row = conn.execute(
        "SELECT * FROM pregame WHERE date = ? ORDER BY id DESC LIMIT 1",
        [today_str],
    ).fetchone()

    if close_after:
        conn.close()

    return dict(row) if row else None
