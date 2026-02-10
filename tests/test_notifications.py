"""Tests for notification formatting and edge cases."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import notifications


@pytest.fixture(autouse=True)
def mock_telegram():
    """Disable actual Telegram sends for all tests."""
    with patch.object(notifications, "TELEGRAM_BOT_TOKEN", "fake-token"), \
         patch.object(notifications, "TELEGRAM_ADMIN_ID", "12345"), \
         patch("notifications.httpx.Client") as mock_client:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_client


class TestSendDailyReport:
    def _make_report_data(self, **overrides):
        data = {
            "regime": "STRONG_BULL",
            "regime_days": 15,
            "qqq_close": 520.50,
            "qqq_sma_50": 510.0,
            "qqq_sma_250": 480.0,
            "qqq_pct_above_sma50": 2.1,
            "qqq_pct_above_sma250": 8.4,
            "momentum_score": 0.72,
            "realized_vol_20d": 18.5,
            "vol_regime": "NORMAL",
            "options_flow_ratio": 0.85,
            "options_flow_bearish": False,
            "trading_days_fetched": 310,
            "gates_passed": 14,
            "gates_failed_list": [],
            "day_trades_remaining": 3,
            "order_action": "BUY",
            "order_shares": 50,
            "target_dollar_value": 15000,
            "allocated_capital": 38700,
            "current_shares": 200,
            "tqqq_position_value": 10128,
            "tqqq_pnl_pct": 5.5,
        }
        data.update(overrides)
        return data

    def test_basic_report_sends(self):
        result = notifications.send_daily_report(self._make_report_data())
        assert result is True

    def test_report_with_zero_values(self):
        data = self._make_report_data(
            qqq_close=0, qqq_sma_50=0, qqq_sma_250=0,
            momentum_score=0, tqqq_position_value=0, tqqq_pnl_pct=0,
        )
        result = notifications.send_daily_report(data)
        assert result is True

    def test_report_with_missing_keys(self):
        """Report should handle missing keys via .get() defaults."""
        result = notifications.send_daily_report({})
        assert result is True

    def test_report_with_pregame(self):
        data = self._make_report_data(
            pregame_sentiment="BEARISH",
            pregame_notes="Persistent bearish flow",
        )
        result = notifications.send_daily_report(data)
        assert result is True

    def test_report_hold_action(self):
        data = self._make_report_data(order_action="HOLD", order_shares=0)
        result = notifications.send_daily_report(data)
        assert result is True

    def test_report_sell_action(self):
        data = self._make_report_data(order_action="SELL", order_shares=100)
        result = notifications.send_daily_report(data)
        assert result is True

    def test_gates_failed(self):
        data = self._make_report_data(
            gates_passed=12,
            gates_failed_list=["MOMENTUM", "VOL_EXTREME"],
        )
        result = notifications.send_daily_report(data)
        assert result is True

    def test_report_with_sqqq_position(self):
        data = self._make_report_data(
            sqqq_current_shares=500,
            sqqq_position_value=6250,
            sqqq_pnl_pct=3.2,
            sqqq_action="BUY",
            sqqq_order_shares=500,
        )
        result = notifications.send_daily_report(data)
        assert result is True

    def test_report_with_sqqq_exit(self):
        data = self._make_report_data(
            sqqq_current_shares=0,
            sqqq_action="EXIT",
            sqqq_order_shares=300,
        )
        result = notifications.send_daily_report(data)
        assert result is True


class TestSendRegimeAlert:
    def test_basic_regime_alert(self):
        data = {
            "qqq_close": 520.0,
            "qqq_sma_250": 480.0,
            "realized_vol_20d": 18.5,
            "vol_regime": "NORMAL",
            "options_flow_ratio": 0.85,
            "options_flow_bearish": False,
            "action_description": "Buy 50 shares TQQQ",
        }
        result = notifications.send_regime_alert("CAUTIOUS", "BULL", data)
        assert result is True

    def test_regime_alert_with_none_old(self):
        data = {
            "qqq_close": 520.0,
            "qqq_sma_250": 480.0,
            "realized_vol_20d": 18.5,
            "vol_regime": "NORMAL",
            "options_flow_ratio": 0.85,
            "options_flow_bearish": False,
        }
        result = notifications.send_regime_alert(None, "CAUTIOUS", data)
        assert result is True

    def test_regime_alert_zero_sma250(self):
        """Division by zero protection when SMA-250 is 0."""
        data = {
            "qqq_close": 520.0,
            "qqq_sma_250": 0,
            "realized_vol_20d": 0,
            "vol_regime": "",
            "options_flow_ratio": 0,
        }
        result = notifications.send_regime_alert("BULL", "RISK_OFF", data)
        assert result is True


class TestSendHalfday:
    def test_halfday_alert(self):
        result = notifications.send_halfday_alert()
        assert result is True


class TestSendError:
    def test_error_sends(self):
        result = notifications.send_error("Test Error", "Something went wrong")
        assert result is True


class TestSendBacktestSummary:
    def _make_stats(self, **overrides):
        stats = {
            "total_return_pct": 12.5,
            "max_drawdown_pct": 8.3,
            "qqq_buy_hold_pct": 28.4,
            "tqqq_buy_hold_pct": -26.0,
            "num_trades": 306,
            "days_in_market": 200,
            "total_days": 400,
            "start_date": "2024-08-01",
            "end_date": "2026-02-01",
        }
        stats.update(overrides)
        return stats

    def test_basic_backtest_summary(self):
        result = notifications.send_backtest_summary(self._make_stats())
        assert result is True

    def test_zero_total_days_no_crash(self):
        """BUG FIX: total_days=0 caused ZeroDivisionError."""
        result = notifications.send_backtest_summary(self._make_stats(total_days=0, days_in_market=0))
        assert result is True

    def test_empty_stats(self):
        result = notifications.send_backtest_summary({})
        assert result is True


class TestSendMessageFallback:
    def test_no_telegram_configured(self):
        """When token is empty, should print and return False."""
        with patch.object(notifications, "TELEGRAM_BOT_TOKEN", ""), \
             patch.object(notifications, "TELEGRAM_ADMIN_ID", ""):
            result = notifications._send_message("test message")
            assert result is False
