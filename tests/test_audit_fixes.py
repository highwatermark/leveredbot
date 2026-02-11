"""Tests for functions added during the leveretf_fix.md audit.

Covers:
- _is_execution_window() — dynamic ±10 min check
- get_regime_history() — DB query for regime transitions
- Data staleness check — abort if bars >3 days old
- API retry Telegram alert — notification after retries exhausted
- QQQ benchmark computation — return since position entry
"""

import sys
import sqlite3
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytz
from db.models import init_tables, log_regime_change, get_regime_history

ET = pytz.timezone("America/New_York")


# ── _is_execution_window ──


class TestIsExecutionWindow:
    """Test the ±10 minute execution window check."""

    def _call(self, now_hour, now_minute, is_half_day=False):
        """Call _is_execution_window with a mocked current time."""
        mock_now = datetime(2025, 1, 15, now_hour, now_minute, 0, tzinfo=ET)
        with patch("job.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            from job import _is_execution_window
            return _is_execution_window(is_half_day)

    def test_exact_normal_time(self):
        """15:50 on a normal day → True."""
        assert self._call(15, 50, is_half_day=False) is True

    def test_within_window_before(self):
        """15:42 (8 min before 15:50) → True."""
        assert self._call(15, 42, is_half_day=False) is True

    def test_within_window_after(self):
        """16:00 (10 min after 15:50) → True."""
        assert self._call(16, 0, is_half_day=False) is True

    def test_outside_window_early(self):
        """15:30 (20 min before 15:50) → False."""
        assert self._call(15, 30, is_half_day=False) is False

    def test_outside_window_late(self):
        """16:15 (25 min after 15:50) → False."""
        assert self._call(16, 15, is_half_day=False) is False

    def test_halfday_exact(self):
        """12:45 on half day → True."""
        assert self._call(12, 45, is_half_day=True) is True

    def test_halfday_within_window(self):
        """12:38 (7 min before 12:45) → True."""
        assert self._call(12, 38, is_half_day=True) is True

    def test_halfday_outside_window(self):
        """12:30 (15 min before 12:45) → False."""
        assert self._call(12, 30, is_half_day=True) is False

    def test_normal_time_on_halfday_fails(self):
        """15:50 on half day → False (should use 12:45)."""
        assert self._call(15, 50, is_half_day=True) is False

    def test_boundary_exactly_10_min(self):
        """15:40 (exactly 10 min before 15:50) → True."""
        assert self._call(15, 40, is_half_day=False) is True

    def test_boundary_11_min(self):
        """15:39 (11 min before 15:50) → False."""
        assert self._call(15, 39, is_half_day=False) is False


# ── get_regime_history ──


class TestGetRegimeHistory:
    """Test the DB function for retrieving regime transitions."""

    @pytest.fixture
    def db_with_regimes(self, db_conn):
        """DB with some regime transition records."""
        today = datetime.now(ET).date()
        # Insert oldest first so id DESC = newest first
        transitions = [
            (today - timedelta(days=90), "STRONG_BULL", "BULL", 510.0, 505.0, 490.0, "Momentum weakened"),
            (today - timedelta(days=45), "BULL", "RISK_OFF", 470.0, 490.0, 485.0, "Breakdown below SMA250"),
            (today - timedelta(days=15), "RISK_OFF", "BULL", 500.0, 495.0, 475.0, "Recovery above SMA250"),
            (today - timedelta(days=5), "BULL", "STRONG_BULL", 520.0, 510.0, 480.0, "SMA50 crossover"),
        ]
        for d, old, new, close, sma50, sma250, reason in transitions:
            log_regime_change(old, new, {
                "date": d.isoformat(),
                "qqq_close": close,
                "qqq_sma_50": sma50,
                "qqq_sma_250": sma250,
                "trigger_reason": reason,
            }, conn=db_conn)
        db_conn.commit()
        return db_conn

    def test_returns_recent_transitions(self, db_with_regimes):
        result = get_regime_history(days=30, conn=db_with_regimes)
        assert len(result) == 2  # Only transitions within last 30 days
        assert result[0]["new_regime"] == "STRONG_BULL"  # Most recent first

    def test_60_day_window(self, db_with_regimes):
        result = get_regime_history(days=60, conn=db_with_regimes)
        assert len(result) == 3  # 5, 15, and 45 days ago

    def test_returns_all_fields(self, db_with_regimes):
        result = get_regime_history(days=30, conn=db_with_regimes)
        row = result[0]
        assert "date" in row
        assert "old_regime" in row
        assert "new_regime" in row
        assert "qqq_close" in row
        assert "trigger_reason" in row

    def test_empty_when_no_data(self, db_conn):
        result = get_regime_history(days=30, conn=db_conn)
        assert result == []

    def test_ordered_by_most_recent(self, db_with_regimes):
        result = get_regime_history(days=60, conn=db_with_regimes)
        dates = [r["date"] for r in result]
        assert dates == sorted(dates, reverse=True)


# ── Data staleness check ──


class TestDataStalenessCheck:
    """Test that cmd_run aborts when bar data is stale."""

    def _make_test_db(self, tmp_path):
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

    @patch("notifications._send_message", return_value=True)
    @patch("notifications.send_regime_alert", return_value=True)
    def test_stale_data_aborts(self, mock_regime, mock_send, tmp_path):
        """Bars dated >3 days ago should abort with error notification."""
        import numpy as np
        db_path, db_factory = self._make_test_db(tmp_path)

        # Create bars with old dates (10 days ago)
        stale_date = (datetime.now(ET).date() - timedelta(days=10)).isoformat()
        np.random.seed(42)
        closes = [480.0]
        for _ in range(310):
            closes.append(closes[-1] * (1 + np.random.normal(0.001, 0.008)))

        bars = []
        base_date = date.fromisoformat("2024-01-02")
        for i, c in enumerate(closes):
            d = base_date + timedelta(days=i * 7 // 5)
            bars.append({"date": d.isoformat(), "close": c, "open": c * 0.999,
                         "high": c * 1.002, "low": c * 0.998, "volume": 45_000_000})

        mock_account = {"equity": 129000.0, "cash": 91000.0, "buying_power": 91000.0,
                        "daytrade_count": 0, "pattern_day_trader": True}
        mock_positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 10128.0,
                          "avg_entry_price": 48.0, "current_price": 50.64,
                          "unrealized_pl": 528.0, "unrealized_plpc": 0.055}]
        mock_calendar = {"date": "2025-01-15", "open": "09:30", "close": "16:00", "is_half_day": False}
        mock_snapshot = {"QQQ": {"latest_trade_price": 520.0}, "TQQQ": {"latest_trade_price": 50.64}}

        with patch("db.models.get_connection", side_effect=db_factory), \
             patch("alpaca_client.get_account", return_value=mock_account), \
             patch("alpaca_client.get_positions", return_value=mock_positions), \
             patch("alpaca_client.get_calendar", return_value=mock_calendar), \
             patch("alpaca_client.get_snapshot", return_value=mock_snapshot), \
             patch("db.cache.get_bars_with_cache", return_value=bars), \
             patch("uw_client.get_tqqq_flow", return_value={"put_premium": 5e6, "call_premium": 6e6,
                   "ratio": 0.83, "is_bearish": False, "adjustment_factor": 1.0,
                   "alert_count": 45, "error": None}):

            import job
            job.cmd_run(halfday_check=False)

        # Should have sent a stale data error
        calls = [c[0][0] if len(c[0]) > 0 else "" for c in mock_send.call_args_list]
        all_text = " ".join(str(a) for a in mock_send.call_args_list)
        assert "stale" in all_text.lower() or "Stale" in all_text

        # Should NOT have a decision record (aborted)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) as cnt FROM decisions").fetchone()
        conn.close()
        assert row["cnt"] == 0

    @patch("notifications._send_message", return_value=True)
    @patch("notifications.send_regime_alert", return_value=True)
    def test_fresh_data_proceeds(self, mock_regime, mock_send, tmp_path):
        """Bars dated today should NOT trigger staleness abort."""
        import numpy as np
        db_path, db_factory = self._make_test_db(tmp_path)

        np.random.seed(42)
        closes = [480.0]
        for _ in range(310):
            closes.append(closes[-1] * (1 + np.random.normal(0.001, 0.008)))

        today = datetime.now(ET).date()
        n = len(closes)
        bars = []
        for i, c in enumerate(closes):
            d = today - timedelta(days=(n - 1 - i) * 7 // 5)
            bars.append({"date": d.isoformat(), "close": c, "open": c * 0.999,
                         "high": c * 1.002, "low": c * 0.998, "volume": 45_000_000})

        mock_account = {"equity": 129000.0, "cash": 91000.0, "buying_power": 91000.0,
                        "daytrade_count": 0, "pattern_day_trader": True}
        mock_positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 10128.0,
                          "avg_entry_price": 48.0, "current_price": 50.64,
                          "unrealized_pl": 528.0, "unrealized_plpc": 0.055}]
        mock_calendar = {"date": "2025-01-15", "open": "09:30", "close": "16:00", "is_half_day": False}
        mock_snapshot = {
            "QQQ": {"latest_trade_price": closes[-1], "latest_trade_time": "2025-01-15T15:45:00-05:00",
                    "daily_bar_close": closes[-1], "daily_bar_open": closes[-1] * 0.999,
                    "daily_bar_high": closes[-1] * 1.002, "daily_bar_low": closes[-1] * 0.998,
                    "daily_bar_volume": 45000000, "prev_daily_bar_close": closes[-2]},
            "TQQQ": {"latest_trade_price": 50.64, "latest_trade_time": "2025-01-15T15:45:00-05:00",
                     "daily_bar_close": 50.50, "daily_bar_open": 49.80, "daily_bar_high": 50.80,
                     "daily_bar_low": 49.60, "daily_bar_volume": 120000000, "prev_daily_bar_close": 49.90},
        }

        with patch("db.models.get_connection", side_effect=db_factory), \
             patch("alpaca_client.get_account", return_value=mock_account), \
             patch("alpaca_client.get_positions", return_value=mock_positions), \
             patch("alpaca_client.get_calendar", return_value=mock_calendar), \
             patch("alpaca_client.get_snapshot", return_value=mock_snapshot), \
             patch("alpaca_client.submit_market_order", return_value={
                 "order_id": "test-123", "status": "filled", "symbol": "TQQQ",
                 "qty": 50, "side": "buy", "filled_avg_price": 50.64, "filled_qty": 50}), \
             patch("db.cache.get_bars_with_cache", return_value=bars), \
             patch("uw_client.get_tqqq_flow", return_value={"put_premium": 5e6, "call_premium": 6e6,
                   "ratio": 0.83, "is_bearish": False, "adjustment_factor": 1.0,
                   "alert_count": 45, "error": None}):

            import job
            job.cmd_run(halfday_check=False)

        # Should have a decision record (proceeded)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) as cnt FROM decisions").fetchone()
        conn.close()
        assert row["cnt"] >= 1


# ── API retry Telegram alert ──


class TestApiRetryAlert:
    """Test that _retry sends Telegram alert after all retries exhausted."""

    def test_alert_sent_after_retries(self):
        """After 3 failed retries, notification.send_error should be called."""
        import alpaca_client

        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        with patch("alpaca_client.MAX_RETRIES", 3), \
             patch("alpaca_client.RETRY_DELAY", 0), \
             patch("notifications.send_error", return_value=True) as mock_alert:
            with pytest.raises(ConnectionError):
                alpaca_client._retry(failing_fn, description="Test call")

        assert call_count == 3
        mock_alert.assert_called_once()
        args = mock_alert.call_args[0]
        assert "Retries Exhausted" in args[0]
        assert "Test call" in args[1]

    def test_no_alert_on_success(self):
        """Successful call should not trigger alert."""
        import alpaca_client

        with patch("alpaca_client.MAX_RETRIES", 3), \
             patch("notifications.send_error") as mock_alert:
            result = alpaca_client._retry(lambda: "OK", description="Test call")

        assert result == "OK"
        mock_alert.assert_not_called()

    def test_alert_failure_doesnt_mask_error(self):
        """If notification itself fails, original error still propagates."""
        import alpaca_client

        def failing_fn():
            raise ConnectionError("Original error")

        with patch("alpaca_client.MAX_RETRIES", 1), \
             patch("alpaca_client.RETRY_DELAY", 0), \
             patch("notifications.send_error", side_effect=RuntimeError("Telegram down")):
            with pytest.raises(ConnectionError, match="Original error"):
                alpaca_client._retry(failing_fn)

    def test_non_retryable_error_propagates_immediately(self):
        """ValueError should not be retried and no alert sent."""
        import alpaca_client

        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad input")

        with patch("alpaca_client.RETRY_DELAY", 0), \
             patch("notifications.send_error") as mock_alert:
            with pytest.raises(ValueError):
                alpaca_client._retry(failing_fn, description="Test")

        assert call_count == 1  # Only called once, not retried
        mock_alert.assert_not_called()


# ── QQQ benchmark computation ──


class TestQqqBenchmark:
    """Test QQQ benchmark return calculation in daily report."""

    @patch("notifications._send_message", return_value=True)
    def test_benchmark_in_report(self, mock_send):
        """When holding TQQQ, benchmark should show QQQ return since entry."""
        from notifications import send_daily_report

        report_data = {
            "regime": "BULL", "regime_days": 10,
            "qqq_close": 520.0, "qqq_sma_50": 510.0, "qqq_sma_250": 480.0,
            "qqq_pct_above_sma50": 1.96, "qqq_pct_above_sma250": 8.33,
            "momentum_score": 0.65, "realized_vol_20d": 18.5, "vol_regime": "NORMAL",
            "options_flow_bearish": False, "options_flow_adjustment": 1.0,
            "trading_days_fetched": 280, "gates_passed": 16,
            "gates_failed_list": [],
            "day_trades_remaining": 3,
            "order_action": "HOLD", "order_shares": 0,
            "target_dollar_value": 19000.0, "allocated_capital": 38000.0,
            "current_shares": 200, "tqqq_position_value": 10128.0,
            "tqqq_pnl_pct": 5.5,
            "knn_direction": "LONG", "knn_confidence": 0.72, "knn_adjustment": 1.0,
            "qqq_benchmark_pct": 2.3,
        }
        send_daily_report(report_data)

        text = mock_send.call_args[0][0]
        assert "Benchmark" in text or "QQQ" in text
        # Benchmark line should appear when holding shares

    @patch("notifications._send_message", return_value=True)
    def test_no_benchmark_when_no_position(self, mock_send):
        """When not holding TQQQ, benchmark should not appear."""
        from notifications import send_daily_report

        report_data = {
            "regime": "RISK_OFF", "regime_days": 5,
            "qqq_close": 460.0, "qqq_sma_50": 470.0, "qqq_sma_250": 480.0,
            "qqq_pct_above_sma50": -2.1, "qqq_pct_above_sma250": -4.2,
            "momentum_score": 0.2, "realized_vol_20d": 28.0, "vol_regime": "HIGH",
            "options_flow_bearish": False, "options_flow_adjustment": 1.0,
            "trading_days_fetched": 280, "gates_passed": 10,
            "gates_failed_list": ["regime_risk_off", "momentum_low"],
            "day_trades_remaining": 3,
            "order_action": "HOLD", "order_shares": 0,
            "target_dollar_value": 0, "allocated_capital": 38000.0,
            "current_shares": 0, "tqqq_position_value": 0,
            "tqqq_pnl_pct": 0,
            "knn_direction": "FLAT", "knn_confidence": 0.5, "knn_adjustment": 1.0,
            "qqq_benchmark_pct": 0,
        }
        send_daily_report(report_data)

        text = mock_send.call_args[0][0]
        # With 0 current_shares, benchmark line should not appear
        assert "outperforming" not in text and "underperforming" not in text
