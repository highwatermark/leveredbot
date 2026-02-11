"""Tests for the Telegram bot command handlers.

bot.py uses lazy imports inside each function body, so patches must
target the source module (e.g. ``db.models.init_tables``) not ``bot.init_tables``.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── Helpers ──


def _make_strategy_signals():
    """Minimal StrategySignals for cmd_leverage."""
    from pipeline_types import StrategySignals

    return StrategySignals(
        qqq_close=520.0,
        sma_50=510.0,
        sma_250=480.0,
        pct_above_sma50=1.96,
        pct_above_sma250=8.33,
        momentum={"roc_fast": 0.012, "roc_slow": 0.035, "raw_score": 0.026, "score": 0.65},
        momentum_score=0.65,
        realized_vol=18.5,
        vol_regime="NORMAL",
        vol_adjustment=1.0,
        flow={"put_premium": 5e6, "call_premium": 6e6, "ratio": 0.83,
              "is_bearish": False, "adjustment_factor": 1.0, "alert_count": 45, "error": None},
        options_flow_bearish=False,
        options_flow_adjustment=1.0,
        options_flow_ratio=0.83,
        raw_regime="BULL",
        effective_regime="BULL",
        previous_regime="BULL",
        regime_hold_days=12,
        regime_changed=False,
        capital={"allocated_capital": 38000.0, "cash_available": 91000.0,
                 "other_positions_value": 38000.0, "tqqq_position_value": 10128.0},
        allocated_capital=38000.0,
        current_shares=200,
        tqqq_price=50.64,
        daily_loss_pct=0.005,
        qqq_closes=[500.0] * 300,
        trading_days_fetched=280,
        day_trades_remaining=3,
        account_equity=129000.0,
        cash_balance=91000.0,
        consecutive_losing_days=0,
        sqqq_price=12.50,
        sqqq_current_shares=0,
        has_tqqq_position=True,
        knn_direction="LONG",
        knn_confidence=0.72,
        knn_adjustment=1.0,
    )


def _make_market_data():
    """Minimal MarketData for cmd_leverage / cmd_leveragevol."""
    from pipeline_types import MarketData

    return MarketData(
        account={"equity": 129000, "cash": 91000, "buying_power": 91000,
                 "daytrade_count": 0, "pattern_day_trader": True},
        positions=[{"symbol": "TQQQ", "qty": 200, "market_value": 10128.0,
                    "avg_entry_price": 48.0, "current_price": 50.64,
                    "unrealized_pl": 528.0, "unrealized_plpc": 0.055}],
        calendar={"date": "2025-01-15", "open": "09:30", "close": "16:00", "is_half_day": False},
        snapshots={},
        qqq_bars=[{"date": "2025-01-15", "close": 520.0}],
        qqq_closes=[520.0] * 300,
        tqqq_position={"symbol": "TQQQ", "qty": 200, "market_value": 10128.0,
                       "avg_entry_price": 48.0, "current_price": 50.64,
                       "unrealized_pl": 528.0, "unrealized_plpc": 0.055},
        tqqq_price=50.64,
        is_half_day=False,
        qqq_current=520.0,
        daily_loss_pct=0.005,
        trading_days_fetched=280,
        sqqq_position=None,
        sqqq_price=None,
    )


# ── Admin auth ──


class TestAdminAuth:
    def test_admin_allowed(self):
        with patch("bot.TELEGRAM_ADMIN_ID", "12345"):
            from bot import _is_admin
            assert _is_admin(12345) is True
            assert _is_admin("12345") is True

    def test_non_admin_rejected(self):
        with patch("bot.TELEGRAM_ADMIN_ID", "12345"):
            from bot import _is_admin
            assert _is_admin(99999) is False
            assert _is_admin("other") is False


# ── Command dispatch ──


class TestCommandDispatch:
    def test_known_commands_registered(self):
        from bot import COMMANDS
        expected = [
            "/leverage", "/leverageperf", "/leverageregime",
            "/leverageexit", "/leveragebacktest",
            "/leveragevol", "/leverageflow",
        ]
        for cmd in expected:
            assert cmd in COMMANDS, f"{cmd} not in COMMANDS dict"

    def test_unknown_leverage_command(self):
        """Unknown /leverage* command sends available commands list."""
        from bot import COMMANDS
        with patch("bot._send", return_value=True) as mock_send:
            cmd = "/leveragefoo"
            chat_id = 123
            if cmd in COMMANDS:
                COMMANDS[cmd](chat_id)
            elif cmd.startswith("/leverage"):
                mock_send(chat_id, f"Unknown command: {cmd}\nAvailable: {', '.join(COMMANDS.keys())}")

            mock_send.assert_called_once()
            args = mock_send.call_args[0]
            assert "Unknown command" in args[1]


# ── /leverage ──


class TestCmdLeverage:
    @patch("bot._send", return_value=True)
    def test_shows_status(self, mock_send):
        signals = _make_strategy_signals()
        data = _make_market_data()

        with patch("db.models.init_tables"), \
             patch("job._fetch_all_data", return_value=data), \
             patch("job._compute_signals", return_value=signals):
            from bot import cmd_leverage
            cmd_leverage(123)

        mock_send.assert_called_once()
        text = mock_send.call_args[0][1]
        assert "BULL" in text
        assert "520.00" in text
        assert "LONG" in text

    @patch("bot._send", return_value=True)
    def test_handles_fetch_error(self, mock_send):
        with patch("db.models.init_tables"), \
             patch("job._fetch_all_data", side_effect=RuntimeError("API down")):
            from bot import cmd_leverage
            cmd_leverage(123)

        text = mock_send.call_args[0][1]
        assert "Error" in text

    @patch("bot._send", return_value=True)
    def test_shows_sqqq_when_held(self, mock_send):
        """When SQQQ position exists, it should appear in status."""
        signals = _make_strategy_signals()
        data = _make_market_data()
        data.sqqq_position = {"symbol": "SQQQ", "qty": 100, "market_value": 1250.0,
                              "avg_entry_price": 12.0, "current_price": 12.50,
                              "unrealized_pl": 50.0, "unrealized_plpc": 0.04}

        with patch("db.models.init_tables"), \
             patch("job._fetch_all_data", return_value=data), \
             patch("job._compute_signals", return_value=signals):
            from bot import cmd_leverage
            cmd_leverage(123)

        text = mock_send.call_args[0][1]
        assert "SQQQ" in text


# ── /leverageperf ──


class TestCmdLeverageperf:
    @patch("bot._send", return_value=True)
    def test_shows_performance(self, mock_send):
        perf = {
            "days": 30, "total_pnl": 1250.0, "avg_daily_pnl": 41.67,
            "best_day": 320.0, "worst_day": -180.0, "latest_total_return_pct": 5.2,
        }
        with patch("db.models.init_tables"), \
             patch("db.models.get_performance_summary", return_value=perf):
            from bot import cmd_leverageperf
            cmd_leverageperf(456)

        text = mock_send.call_args[0][1]
        assert "1,250.00" in text
        assert "+5.2%" in text


# ── /leverageregime ──


class TestCmdLeverageregime:
    @patch("bot._send", return_value=True)
    def test_shows_regime_history(self, mock_send):
        history = [
            {"date": "2025-01-10", "old_regime": "BULL", "new_regime": "STRONG_BULL", "qqq_close": 520.0},
            {"date": "2025-01-05", "old_regime": "RISK_OFF", "new_regime": "BULL", "qqq_close": 500.0},
        ]
        with patch("db.models.init_tables"), \
             patch("db.models.get_regime_history", return_value=history):
            from bot import cmd_leverageregime
            cmd_leverageregime(789)

        text = mock_send.call_args[0][1]
        assert "STRONG_BULL" in text
        assert "2025-01-10" in text

    @patch("bot._send", return_value=True)
    def test_empty_history(self, mock_send):
        with patch("db.models.init_tables"), \
             patch("db.models.get_regime_history", return_value=[]):
            from bot import cmd_leverageregime
            cmd_leverageregime(789)

        text = mock_send.call_args[0][1]
        assert "No regime transitions" in text


# ── /leverageexit ──


class TestCmdLeverageexit:
    @patch("bot._send", return_value=True)
    def test_force_exit_success(self, mock_send):
        result = {
            "tqqq": {"executed": True, "shares_sold": 200},
            "sqqq": {"executed": False, "reason": "No position"},
        }
        with patch("job.cmd_force_exit", return_value=result):
            from bot import cmd_leverageexit
            cmd_leverageexit(123)

        assert mock_send.call_count == 2
        results_text = mock_send.call_args_list[1][0][1]
        assert "200" in results_text

    @patch("bot._send", return_value=True)
    def test_force_exit_error(self, mock_send):
        with patch("job.cmd_force_exit", side_effect=RuntimeError("Network error")):
            from bot import cmd_leverageexit
            cmd_leverageexit(123)

        results_text = mock_send.call_args[0][1]
        assert "failed" in results_text.lower()


# ── /leveragebacktest ──


class TestCmdLeveragebacktest:
    @patch("bot._send", return_value=True)
    def test_backtest_success(self, mock_send):
        stats = {
            "start_date": "2023-06-01", "end_date": "2025-01-15",
            "total_return_pct": 45.2, "max_drawdown_pct": 18.5,
            "num_trades": 42, "qqq_buy_hold_pct": 30.1,
            "tqqq_buy_hold_pct": 60.5,
        }
        with patch("job.cmd_backtest", return_value=stats):
            from bot import cmd_leveragebacktest
            cmd_leveragebacktest(123)

        assert mock_send.call_count == 2
        text = mock_send.call_args_list[1][0][1]
        assert "+45.2%" in text
        assert "42" in text

    @patch("bot._send", return_value=True)
    def test_backtest_no_results(self, mock_send):
        with patch("job.cmd_backtest", return_value=None):
            from bot import cmd_leveragebacktest
            cmd_leveragebacktest(123)

        text = mock_send.call_args[0][1]
        assert "no results" in text.lower()

    @patch("bot._send", return_value=True)
    def test_backtest_error(self, mock_send):
        with patch("job.cmd_backtest", side_effect=RuntimeError("OOM")):
            from bot import cmd_leveragebacktest
            cmd_leveragebacktest(123)

        text = mock_send.call_args[0][1]
        assert "failed" in text.lower()


# ── /leveragevol ──


class TestCmdLeveragevol:
    @patch("bot._send", return_value=True)
    def test_vol_breakdown(self, mock_send):
        data = _make_market_data()
        with patch("db.models.init_tables"), \
             patch("job._fetch_all_data", return_value=data), \
             patch("strategy.signals.calculate_realized_vol", side_effect=[18.5, 20.0, 22.0]), \
             patch("strategy.signals.classify_vol_regime", return_value="NORMAL"), \
             patch("strategy.signals.get_vol_adjustment", return_value=1.0):
            from bot import cmd_leveragevol
            cmd_leveragevol(123)

        text = mock_send.call_args[0][1]
        assert "NORMAL" in text
        assert "100%" in text

    @patch("bot._send", return_value=True)
    def test_vol_fetch_error(self, mock_send):
        with patch("db.models.init_tables"), \
             patch("job._fetch_all_data", side_effect=RuntimeError("timeout")):
            from bot import cmd_leveragevol
            cmd_leveragevol(123)

        text = mock_send.call_args[0][1]
        assert "Error" in text


# ── /leverageflow ──


class TestCmdLeverageflow:
    @patch("bot._send", return_value=True)
    def test_neutral_flow(self, mock_send):
        flow = {
            "put_premium": 5000000, "call_premium": 6000000,
            "ratio": 0.83, "is_bearish": False,
            "adjustment_factor": 1.0, "alert_count": 45, "error": None,
        }
        with patch("db.models.init_tables"), \
             patch("uw_client.get_tqqq_flow", return_value=flow), \
             patch("strategy.signals.check_options_flow", return_value=(False, 1.0)):
            from bot import cmd_leverageflow
            cmd_leverageflow(123)

        text = mock_send.call_args[0][1]
        assert "NEUTRAL" in text
        assert "0.83" in text

    @patch("bot._send", return_value=True)
    def test_bearish_flow(self, mock_send):
        flow = {
            "put_premium": 12000000, "call_premium": 4000000,
            "ratio": 3.0, "is_bearish": True,
            "adjustment_factor": 0.75, "alert_count": 70, "error": None,
        }
        with patch("db.models.init_tables"), \
             patch("uw_client.get_tqqq_flow", return_value=flow), \
             patch("strategy.signals.check_options_flow", return_value=(True, 0.75)):
            from bot import cmd_leverageflow
            cmd_leverageflow(123)

        text = mock_send.call_args[0][1]
        assert "BEARISH" in text
        assert "75%" in text

    @patch("bot._send", return_value=True)
    def test_flow_with_error_warning(self, mock_send):
        flow = {
            "put_premium": 5000000, "call_premium": 6000000,
            "ratio": 0.83, "is_bearish": False,
            "adjustment_factor": 1.0, "alert_count": 0,
            "error": "Rate limited",
        }
        with patch("db.models.init_tables"), \
             patch("uw_client.get_tqqq_flow", return_value=flow), \
             patch("strategy.signals.check_options_flow", return_value=(False, 1.0)):
            from bot import cmd_leverageflow
            cmd_leverageflow(123)

        text = mock_send.call_args[0][1]
        assert "Rate limited" in text

    @patch("bot._send", return_value=True)
    def test_flow_fetch_error(self, mock_send):
        with patch("db.models.init_tables"), \
             patch("uw_client.get_tqqq_flow", side_effect=RuntimeError("UW down")):
            from bot import cmd_leverageflow
            cmd_leverageflow(123)

        text = mock_send.call_args[0][1]
        assert "Error" in text


# ── _send function ──


class TestSend:
    def test_send_success(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("bot.TELEGRAM_BOT_TOKEN", "fake-token"), \
             patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            from bot import _send
            result = _send(123, "Hello")

        assert result is True

    def test_send_failure(self):
        with patch("bot.TELEGRAM_BOT_TOKEN", "fake-token"), \
             patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = Exception("Network error")
            mock_client_cls.return_value = mock_client

            from bot import _send
            result = _send(123, "Hello")

        assert result is False
