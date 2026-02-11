"""Integration test for the full cmd_run pipeline.

Mocks all external APIs (Alpaca, UW, Telegram) and verifies:
- Data flows correctly between stages
- DB records are written
- Regime detection + gates + sizing + execution all integrate correctly
"""

import sys
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from db.models import init_tables


def _make_bull_closes(n=310):
    """Generate realistic bull market QQQ closes."""
    np.random.seed(42)
    prices = [480.0]
    for _ in range(n):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.008)))
    return prices


def _make_bars_from_closes(closes):
    """Convert closes list to bar dicts with dates ending near today."""
    from datetime import date, timedelta
    import pytz
    today = datetime.now(pytz.timezone("America/New_York")).date()
    # Work backwards from today so data is never stale
    n = len(closes)
    bars = []
    for i, close in enumerate(closes):
        d = today - timedelta(days=(n - 1 - i) * 7 // 5)  # ~weekdays, ending today
        bars.append({
            "date": d.isoformat(),
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 45_000_000,
        })
    return bars


def _read_db(db_path):
    """Open a fresh connection to read test DB after cmd_run closes its connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _make_test_db(tmp_path):
    """Create a file-based test DB and return (db_path, factory_fn).

    factory_fn returns a new connection each time, matching get_connection() behavior.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    init_tables(conn)
    conn.commit()
    conn.close()

    def factory():
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    return db_path, factory


CLOSES = _make_bull_closes()
BARS = _make_bars_from_closes(CLOSES)

MOCK_ACCOUNT = {
    "equity": 129000.0,
    "cash": 91000.0,
    "buying_power": 91000.0,
    "daytrade_count": 0,
    "pattern_day_trader": True,
}
MOCK_POSITIONS = [
    {"symbol": "TQQQ", "qty": 200, "market_value": 10128.0,
     "avg_entry_price": 48.0, "current_price": 50.64,
     "unrealized_pl": 528.0, "unrealized_plpc": 0.055},
]
MOCK_CALENDAR = {"date": "2025-01-15", "open": "09:30", "close": "16:00", "is_half_day": False}
MOCK_SNAPSHOT = {
    "QQQ": {
        "latest_trade_price": CLOSES[-1],
        "latest_trade_time": "2025-01-15T15:45:00-05:00",
        "daily_bar_close": CLOSES[-1],
        "daily_bar_open": CLOSES[-1] * 0.999,
        "daily_bar_high": CLOSES[-1] * 1.002,
        "daily_bar_low": CLOSES[-1] * 0.998,
        "daily_bar_volume": 45000000,
        "prev_daily_bar_close": CLOSES[-2],
    },
    "TQQQ": {
        "latest_trade_price": 50.64,
        "latest_trade_time": "2025-01-15T15:45:00-05:00",
        "daily_bar_close": 50.50,
        "daily_bar_open": 49.80,
        "daily_bar_high": 50.80,
        "daily_bar_low": 49.60,
        "daily_bar_volume": 120000000,
        "prev_daily_bar_close": 49.90,
    },
}
MOCK_FLOW = {
    "put_premium": 5000000,
    "call_premium": 6000000,
    "ratio": 0.83,
    "is_bearish": False,
    "adjustment_factor": 1.0,
    "alert_count": 45,
    "error": None,
}
MOCK_ORDER = {
    "order_id": "test-order-123",
    "status": "filled",
    "symbol": "TQQQ",
    "qty": 50,
    "side": "buy",
    "filled_avg_price": 50.64,
    "filled_qty": 50,
}


class TestCmdRunIntegration:
    """Test the full cmd_run pipeline with mocked externals."""

    @patch("notifications._send_message", return_value=True)
    @patch("notifications.send_regime_alert", return_value=True)
    def test_full_bull_pipeline(self, mock_regime_alert, mock_send, tmp_path):
        """Full pipeline in bull market: should compute signals, pass gates, and log decision."""
        db_path, db_factory = _make_test_db(tmp_path)

        with patch("db.models.get_connection", side_effect=db_factory), \
             patch("alpaca_client.get_account", return_value=MOCK_ACCOUNT), \
             patch("alpaca_client.get_positions", return_value=MOCK_POSITIONS), \
             patch("alpaca_client.get_calendar", return_value=MOCK_CALENDAR), \
             patch("alpaca_client.get_snapshot", return_value=MOCK_SNAPSHOT), \
             patch("alpaca_client.submit_market_order", return_value=MOCK_ORDER), \
             patch("db.cache.get_bars_with_cache", return_value=BARS), \
             patch("uw_client.get_tqqq_flow", return_value=MOCK_FLOW):

            import job
            job.cmd_run(halfday_check=False)

        # Verify DB has a decision record
        conn = _read_db(db_path)
        try:
            row = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
            assert row is not None, "Expected a decision record to be logged"
            assert row["regime"] is not None
            assert row["qqq_close"] > 0
            assert row["qqq_sma_50"] > 0
            assert row["qqq_sma_250"] > 0
            assert row["target_shares"] >= 0
            assert row["status"] in ("COMPLETE", "EXECUTED", "FAILED"), f"Unexpected status: {row['status']}"

            perf = conn.execute("SELECT * FROM performance ORDER BY id DESC LIMIT 1").fetchone()
            assert perf is not None, "Expected a performance record to be logged"
            assert perf["regime"] is not None
        finally:
            conn.close()

    @patch("notifications._send_message", return_value=True)
    def test_market_closed_exits_early(self, mock_send, tmp_path):
        """When calendar returns None, cmd_run should exit without logging."""
        db_path, db_factory = _make_test_db(tmp_path)

        with patch("db.models.get_connection", side_effect=db_factory), \
             patch("alpaca_client.get_account", return_value=MOCK_ACCOUNT), \
             patch("alpaca_client.get_positions", return_value=MOCK_POSITIONS), \
             patch("alpaca_client.get_calendar", return_value=None), \
             patch("alpaca_client.get_snapshot", return_value=MOCK_SNAPSHOT), \
             patch("db.cache.get_bars_with_cache", return_value=BARS), \
             patch("uw_client.get_tqqq_flow", return_value=MOCK_FLOW):

            import job
            job.cmd_run(halfday_check=False)

        conn = _read_db(db_path)
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM decisions").fetchone()
            assert row["cnt"] == 0
        finally:
            conn.close()

    @patch("notifications._send_message", return_value=True)
    def test_insufficient_data_exits(self, mock_send, tmp_path):
        """When fewer than 250 bars, cmd_run should exit with error."""
        db_path, db_factory = _make_test_db(tmp_path)
        short_bars = _make_bars_from_closes(CLOSES[:100])

        with patch("db.models.get_connection", side_effect=db_factory), \
             patch("alpaca_client.get_account", return_value=MOCK_ACCOUNT), \
             patch("alpaca_client.get_positions", return_value=MOCK_POSITIONS), \
             patch("alpaca_client.get_calendar", return_value=MOCK_CALENDAR), \
             patch("alpaca_client.get_snapshot", return_value=MOCK_SNAPSHOT), \
             patch("db.cache.get_bars_with_cache", return_value=short_bars), \
             patch("uw_client.get_tqqq_flow", return_value=MOCK_FLOW):

            import job
            job.cmd_run(halfday_check=False)

        conn = _read_db(db_path)
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM decisions").fetchone()
            assert row["cnt"] == 0
        finally:
            conn.close()

    @patch("notifications._send_message", return_value=True)
    @patch("notifications.send_regime_alert", return_value=True)
    def test_halfday_check_skips_normal_day(self, mock_regime_alert, mock_send, tmp_path):
        """With --halfday-check on a normal day, cmd_run should exit early."""
        db_path, db_factory = _make_test_db(tmp_path)

        with patch("db.models.get_connection", side_effect=db_factory), \
             patch("alpaca_client.get_account", return_value=MOCK_ACCOUNT), \
             patch("alpaca_client.get_positions", return_value=MOCK_POSITIONS), \
             patch("alpaca_client.get_calendar", return_value=MOCK_CALENDAR), \
             patch("alpaca_client.get_snapshot", return_value=MOCK_SNAPSHOT), \
             patch("db.cache.get_bars_with_cache", return_value=BARS), \
             patch("uw_client.get_tqqq_flow", return_value=MOCK_FLOW):

            import job
            job.cmd_run(halfday_check=True)

        conn = _read_db(db_path)
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM decisions").fetchone()
            assert row["cnt"] == 0
        finally:
            conn.close()
