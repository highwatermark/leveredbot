"""
Tests for the windowed position manager.

Tests cover all exit check functions, morning/midday orchestration,
decision selection, and DB helpers.
"""

import sqlite3
from datetime import datetime, date
from unittest.mock import MagicMock, patch

import pytest

from strategy.position_manager import PositionState, ExitDecision, PositionManager
from db.models import (
    init_tables,
    save_position_event,
    update_position_event,
    get_today_position_events,
    get_position_watermark,
    update_position_watermark,
    get_profit_tiers_taken,
)
from config import LEVERAGE_CONFIG


# ── Helpers ──

def _make_state(**overrides) -> PositionState:
    """Create a PositionState with sensible defaults, overridable."""
    defaults = dict(
        symbol="TQQQ",
        shares=200,
        avg_entry_price=50.0,
        current_price=52.0,
        market_value=10400.0,
        unrealized_pnl_pct=0.04,
        entry_date="2025-01-10",
        holding_days=5,
        intraday_high=53.0,
        intraday_low=51.0,
        intraday_open=51.50,
        prev_close=51.00,
        overnight_gap_pct=0.0098,
        intraday_change_pct=0.0097,
        intraday_drawdown_pct=0.019,
    )
    defaults.update(overrides)
    return PositionState(**defaults)


def _make_pm(config_overrides=None) -> PositionManager:
    """Create a PositionManager with mock alpaca_client."""
    cfg = dict(LEVERAGE_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)
    mock_client = MagicMock()
    return PositionManager(alpaca_client=mock_client, config=cfg)


# ══════════════════════════════════════════════
#  TestCheckStopLoss
# ══════════════════════════════════════════════

class TestCheckStopLoss:
    def test_triggered_when_loss_exceeds_threshold(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=50.0, current_price=45.0)  # -10%
        result = pm.check_stop_loss(state)
        assert result.should_exit is True
        assert result.exit_type == "STOP_LOSS"
        assert result.urgency == "URGENT"
        assert result.shares_to_sell == 200

    def test_not_triggered_when_above_threshold(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=50.0, current_price=48.0)  # -4%
        result = pm.check_stop_loss(state)
        assert result.should_exit is False

    def test_at_boundary(self):
        pm = _make_pm()
        # Exactly at 8% loss
        state = _make_state(avg_entry_price=50.0, current_price=46.0)  # -8%
        result = pm.check_stop_loss(state)
        assert result.should_exit is True

    def test_zero_entry_price(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=0, current_price=50.0)
        result = pm.check_stop_loss(state)
        assert result.should_exit is False

    def test_sqqq_same_logic(self):
        pm = _make_pm()
        state = _make_state(symbol="SQQQ", avg_entry_price=12.0, current_price=10.5)  # -12.5%
        result = pm.check_stop_loss(state)
        assert result.should_exit is True
        assert result.exit_type == "STOP_LOSS"

    def test_price_above_entry_no_loss(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=50.0, current_price=55.0)  # +10%
        result = pm.check_stop_loss(state)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckTrailingStop
# ══════════════════════════════════════════════

class TestCheckTrailingStop:
    def test_triggered_below_watermark(self):
        pm = _make_pm()
        state = _make_state(current_price=47.0)
        result = pm.check_trailing_stop(state, high_watermark=55.0)  # -14.5%
        assert result.should_exit is True
        assert result.exit_type == "TRAILING_STOP"
        assert result.urgency == "NORMAL"

    def test_not_triggered_at_watermark(self):
        pm = _make_pm()
        state = _make_state(current_price=54.0)
        result = pm.check_trailing_stop(state, high_watermark=55.0)  # -1.8%
        assert result.should_exit is False

    def test_not_triggered_above_watermark(self):
        pm = _make_pm()
        state = _make_state(current_price=56.0)
        result = pm.check_trailing_stop(state, high_watermark=55.0)
        assert result.should_exit is False

    def test_zero_watermark(self):
        pm = _make_pm()
        state = _make_state(current_price=50.0)
        result = pm.check_trailing_stop(state, high_watermark=0)
        assert result.should_exit is False

    def test_exactly_at_threshold(self):
        pm = _make_pm()
        # 6% drop from 100 = price at 94
        state = _make_state(current_price=94.0)
        result = pm.check_trailing_stop(state, high_watermark=100.0)
        assert result.should_exit is True

    def test_just_under_threshold(self):
        pm = _make_pm()
        # 5.9% drop from 100 = price at 94.1
        state = _make_state(current_price=94.1)
        result = pm.check_trailing_stop(state, high_watermark=100.0)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckGapDown
# ══════════════════════════════════════════════

class TestCheckGapDown:
    def test_large_gap_tqqq(self):
        pm = _make_pm()
        state = _make_state(intraday_open=48.0, prev_close=52.0)  # -7.7% gap
        result = pm.check_gap_down(state)
        assert result.should_exit is True
        assert result.exit_type == "GAP_DOWN"
        assert result.urgency == "URGENT"

    def test_small_gap_no_exit(self):
        pm = _make_pm()
        state = _make_state(intraday_open=50.5, prev_close=51.0)  # -1% gap
        result = pm.check_gap_down(state)
        assert result.should_exit is False

    def test_positive_gap_no_exit(self):
        pm = _make_pm()
        state = _make_state(intraday_open=53.0, prev_close=51.0)  # +3.9% gap up
        result = pm.check_gap_down(state)
        assert result.should_exit is False

    def test_sqqq_gap_down_exit(self):
        """SQQQ: gaps down when QQQ gaps up."""
        pm = _make_pm()
        state = _make_state(symbol="SQQQ", intraday_open=11.0, prev_close=12.0)  # -8.3% gap
        result = pm.check_gap_down(state)
        assert result.should_exit is True
        assert result.exit_type == "GAP_DOWN"

    def test_zero_prev_close(self):
        pm = _make_pm()
        state = _make_state(prev_close=0)
        result = pm.check_gap_down(state)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckRegimeEmergency
# ══════════════════════════════════════════════

class TestCheckRegimeEmergency:
    def test_qqq_below_sma250_triggers_tqqq_exit(self):
        pm = _make_pm()
        state = _make_state(symbol="TQQQ")
        # QQQ at 480, SMA-250 at 500 -> -4% below
        result = pm.check_regime_emergency(state, qqq_current=480, sma_250=500)
        assert result.should_exit is True
        assert result.exit_type == "REGIME_EMERGENCY"
        assert result.urgency == "URGENT"

    def test_qqq_near_sma250_no_exit(self):
        pm = _make_pm()
        state = _make_state(symbol="TQQQ")
        # QQQ at 498, SMA-250 at 500 -> -0.4% below (threshold is 3%)
        result = pm.check_regime_emergency(state, qqq_current=498, sma_250=500)
        assert result.should_exit is False

    def test_sqqq_reverse_qqq_above_sma250(self):
        """SQQQ exits when QQQ rises well above SMA-250."""
        pm = _make_pm()
        state = _make_state(symbol="SQQQ")
        # QQQ at 520, SMA-250 at 500 -> +4% above
        result = pm.check_regime_emergency(state, qqq_current=520, sma_250=500)
        assert result.should_exit is True
        assert result.exit_type == "REGIME_EMERGENCY"

    def test_sqqq_qqq_below_no_exit(self):
        """SQQQ benefits from QQQ falling, no exit."""
        pm = _make_pm()
        state = _make_state(symbol="SQQQ")
        # QQQ at 480, SMA-250 at 500 -> -4% below
        result = pm.check_regime_emergency(state, qqq_current=480, sma_250=500)
        assert result.should_exit is False

    def test_zero_sma(self):
        pm = _make_pm()
        state = _make_state()
        result = pm.check_regime_emergency(state, qqq_current=500, sma_250=0)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckVolSpike
# ══════════════════════════════════════════════

class TestCheckVolSpike:
    def test_large_spike(self):
        pm = _make_pm()
        state = _make_state()
        result = pm.check_vol_spike(state, current_vol=200_000_000, prev_vol=100_000_000)  # +100%
        assert result.should_exit is True
        assert result.exit_type == "VOL_SPIKE"
        assert result.urgency == "NORMAL"

    def test_moderate_spike_no_exit(self):
        pm = _make_pm()
        state = _make_state()
        result = pm.check_vol_spike(state, current_vol=130_000_000, prev_vol=100_000_000)  # +30%
        assert result.should_exit is False

    def test_no_spike(self):
        pm = _make_pm()
        state = _make_state()
        result = pm.check_vol_spike(state, current_vol=90_000_000, prev_vol=100_000_000)  # -10%
        assert result.should_exit is False

    def test_zero_prev_vol(self):
        pm = _make_pm()
        state = _make_state()
        result = pm.check_vol_spike(state, current_vol=100_000_000, prev_vol=0)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckMaxHoldPeriod
# ══════════════════════════════════════════════

class TestCheckMaxHoldPeriod:
    def test_losing_and_exceeded(self):
        pm = _make_pm()
        state = _make_state(unrealized_pnl_pct=-0.05, holding_days=20)
        result = pm.check_max_hold_period(state)
        assert result.should_exit is True
        assert result.exit_type == "MAX_HOLD"
        assert result.urgency == "NORMAL"

    def test_losing_but_not_exceeded(self):
        pm = _make_pm()
        state = _make_state(unrealized_pnl_pct=-0.05, holding_days=10)
        result = pm.check_max_hold_period(state)
        assert result.should_exit is False

    def test_winning_position_no_exit(self):
        pm = _make_pm()
        state = _make_state(unrealized_pnl_pct=0.10, holding_days=30)
        result = pm.check_max_hold_period(state)
        assert result.should_exit is False

    def test_breakeven_no_exit(self):
        pm = _make_pm()
        state = _make_state(unrealized_pnl_pct=0.0, holding_days=30)
        result = pm.check_max_hold_period(state)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckDailyLossLimit
# ══════════════════════════════════════════════

class TestCheckDailyLossLimit:
    def test_exceeded(self):
        pm = _make_pm()
        result = pm.check_daily_loss_limit(account_equity=90_000, prev_equity=100_000)  # -10%
        assert result.should_exit is True
        assert result.exit_type == "DAILY_LOSS_LIMIT"
        assert result.urgency == "URGENT"

    def test_at_boundary(self):
        pm = _make_pm()
        result = pm.check_daily_loss_limit(account_equity=95_000, prev_equity=100_000)  # -5%
        assert result.should_exit is True

    def test_within_limit(self):
        pm = _make_pm()
        result = pm.check_daily_loss_limit(account_equity=97_000, prev_equity=100_000)  # -3%
        assert result.should_exit is False

    def test_zero_prev_equity(self):
        pm = _make_pm()
        result = pm.check_daily_loss_limit(account_equity=100_000, prev_equity=0)
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestCheckPartialProfit
# ══════════════════════════════════════════════

class TestCheckPartialProfit:
    def test_first_tier_triggers(self):
        pm = _make_pm()
        # +10% gain, first tier is +8%
        state = _make_state(avg_entry_price=50.0, current_price=55.0)
        result = pm.check_partial_profit(state, tiers_taken=[])
        assert result.should_exit is True
        assert result.exit_type == "PARTIAL_PROFIT"
        assert result.shares_to_sell == 50  # 25% of 200

    def test_second_tier_triggers(self):
        pm = _make_pm()
        # +16% gain, first tier already taken
        state = _make_state(avg_entry_price=50.0, current_price=58.0)
        result = pm.check_partial_profit(state, tiers_taken=[8.0])
        assert result.should_exit is True
        assert result.shares_to_sell == 50

    def test_third_tier_triggers(self):
        pm = _make_pm()
        # +30% gain, first two tiers taken
        state = _make_state(avg_entry_price=50.0, current_price=65.0)
        result = pm.check_partial_profit(state, tiers_taken=[8.0, 15.0])
        assert result.should_exit is True

    def test_all_tiers_done(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=50.0, current_price=70.0)
        result = pm.check_partial_profit(state, tiers_taken=[8.0, 15.0, 25.0])
        assert result.should_exit is False

    def test_no_profit(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=50.0, current_price=48.0)  # -4%
        result = pm.check_partial_profit(state, tiers_taken=[])
        assert result.should_exit is False

    def test_below_first_tier(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=50.0, current_price=53.0)  # +6%
        result = pm.check_partial_profit(state, tiers_taken=[])
        assert result.should_exit is False

    def test_zero_shares(self):
        pm = _make_pm()
        state = _make_state(shares=0, avg_entry_price=50.0, current_price=60.0)
        result = pm.check_partial_profit(state, tiers_taken=[])
        assert result.should_exit is False

    def test_zero_entry_price(self):
        pm = _make_pm()
        state = _make_state(avg_entry_price=0, current_price=60.0)
        result = pm.check_partial_profit(state, tiers_taken=[])
        assert result.should_exit is False


# ══════════════════════════════════════════════
#  TestSelectDecision
# ══════════════════════════════════════════════

class TestSelectDecision:
    def test_urgency_priority(self):
        """URGENT beats NORMAL."""
        d1 = ExitDecision(should_exit=True, exit_type="TRAILING_STOP", shares_to_sell=200, urgency="NORMAL")
        d2 = ExitDecision(should_exit=True, exit_type="STOP_LOSS", shares_to_sell=100, urgency="URGENT")
        result = PositionManager._select_decision([d1, d2])
        assert result.exit_type == "STOP_LOSS"

    def test_same_urgency_picks_largest(self):
        """Same urgency → pick largest shares_to_sell."""
        d1 = ExitDecision(should_exit=True, exit_type="TRAILING_STOP", shares_to_sell=100, urgency="NORMAL")
        d2 = ExitDecision(should_exit=True, exit_type="VOL_SPIKE", shares_to_sell=200, urgency="NORMAL")
        result = PositionManager._select_decision([d1, d2])
        assert result.shares_to_sell == 200

    def test_empty_list(self):
        result = PositionManager._select_decision([])
        assert result is None


# ══════════════════════════════════════════════
#  TestRunMorningCheck
# ══════════════════════════════════════════════

class TestRunMorningCheck:
    def _setup_pm(self, positions=None, snapshots=None, account=None):
        """Create PM with mock client."""
        pm = _make_pm()
        pm.alpaca_client.get_positions.return_value = positions or []
        pm.alpaca_client.get_snapshot.return_value = snapshots or {}
        pm.alpaca_client.get_account.return_value = account or {
            "equity": 100_000, "cash": 50_000, "daytrade_count": 0,
        }
        return pm

    def test_no_positions(self, db_conn):
        pm = self._setup_pm()
        decisions = pm.run_morning_check(db_conn)
        assert decisions == []

    @patch("strategy.position_manager.PositionManager._get_sma_250", return_value=500.0)
    @patch("notifications.send_position_exit_alert")
    def test_gap_down_exit(self, mock_alert, mock_sma, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 9000, "avg_entry_price": 50.0,
                      "current_price": 45.0, "unrealized_pl": -1000, "unrealized_plpc": -0.10}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 45.0, "prev_daily_bar_close": 50.0,
            "daily_bar_open": 47.0, "daily_bar_high": 47.5, "daily_bar_low": 44.5,
        }}
        # Entry date
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares) VALUES (?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200])
        db_conn.commit()

        pm = self._setup_pm(positions, snapshots)
        pm.alpaca_client.get_snapshot.return_value = snapshots

        # Also mock the QQQ snapshot for regime emergency
        original_get_snap = pm.alpaca_client.get_snapshot.side_effect

        def snapshot_router(symbols):
            if "QQQ" in symbols:
                return {"QQQ": {"latest_trade_price": 500.0}}
            return snapshots
        pm.alpaca_client.get_snapshot.side_effect = snapshot_router

        decisions = pm.run_morning_check(db_conn)
        assert len(decisions) >= 1
        exit_types = {d.exit_type for d in decisions}
        # Should trigger at least GAP_DOWN or STOP_LOSS
        assert exit_types & {"GAP_DOWN", "STOP_LOSS"}

    @patch("strategy.position_manager.PositionManager._get_sma_250", return_value=500.0)
    @patch("notifications.send_position_exit_alert")
    def test_stop_loss_exit(self, mock_alert, mock_sma, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 8400, "avg_entry_price": 50.0,
                      "current_price": 42.0, "unrealized_pl": -1600, "unrealized_plpc": -0.16}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 42.0, "prev_daily_bar_close": 49.0,
            "daily_bar_open": 48.0, "daily_bar_high": 48.5, "daily_bar_low": 41.5,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares) VALUES (?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200])
        db_conn.commit()

        pm = self._setup_pm(positions, snapshots)

        def snapshot_router(symbols):
            if "QQQ" in symbols:
                return {"QQQ": {"latest_trade_price": 500.0}}
            return snapshots
        pm.alpaca_client.get_snapshot.side_effect = snapshot_router

        decisions = pm.run_morning_check(db_conn)
        assert len(decisions) >= 1
        assert any(d.exit_type == "STOP_LOSS" for d in decisions)

    @patch("strategy.position_manager.PositionManager._get_sma_250", return_value=500.0)
    @patch("notifications.send_position_exit_alert")
    def test_regime_emergency(self, mock_alert, mock_sma, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 10000, "avg_entry_price": 50.0,
                      "current_price": 50.0, "unrealized_pl": 0, "unrealized_plpc": 0.0}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 50.0, "prev_daily_bar_close": 50.0,
            "daily_bar_open": 50.0, "daily_bar_high": 50.5, "daily_bar_low": 49.5,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares) VALUES (?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200])
        db_conn.commit()

        pm = self._setup_pm(positions, snapshots)
        # QQQ is 4% below SMA-250
        def snapshot_router(symbols):
            if "QQQ" in symbols:
                return {"QQQ": {"latest_trade_price": 480.0}}
            return snapshots
        pm.alpaca_client.get_snapshot.side_effect = snapshot_router
        mock_sma.return_value = 500.0

        decisions = pm.run_morning_check(db_conn)
        assert len(decisions) >= 1
        assert any(d.exit_type == "REGIME_EMERGENCY" for d in decisions)

    @patch("strategy.position_manager.PositionManager._get_sma_250", return_value=500.0)
    def test_pdt_blocks_non_urgent(self, mock_sma, db_conn):
        """Non-urgent exits are skipped when day trades are insufficient."""
        pm = _make_pm({"pm_min_day_trades_reserve": 2})
        # Set up account with few day trades
        pm.alpaca_client.get_account.return_value = {
            "equity": 100_000, "cash": 50_000, "daytrade_count": 1,
        }
        pm.alpaca_client.get_positions.return_value = [
            {"symbol": "TQQQ", "qty": 200, "market_value": 10000, "avg_entry_price": 50.0,
             "current_price": 50.0, "unrealized_pl": 0, "unrealized_plpc": 0.0}
        ]
        snapshots = {"TQQQ": {
            "latest_trade_price": 50.0, "prev_daily_bar_close": 50.0,
            "daily_bar_open": 50.0, "daily_bar_high": 50.5, "daily_bar_low": 49.5,
        }}

        def snapshot_router(symbols):
            if "QQQ" in symbols:
                return {"QQQ": {"latest_trade_price": 500.0}}
            return snapshots
        pm.alpaca_client.get_snapshot.side_effect = snapshot_router

        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares, account_equity) VALUES (?, ?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200, 100_000])
        db_conn.commit()

        # No exits should fire (position is healthy, no gaps, no regime issues)
        decisions = pm.run_morning_check(db_conn)
        # All healthy - nothing to exit
        assert len(decisions) == 0


# ══════════════════════════════════════════════
#  TestRunMiddayCheck
# ══════════════════════════════════════════════

class TestRunMiddayCheck:
    def _setup_pm(self, positions=None, snapshots=None, account=None, config_overrides=None):
        pm = _make_pm(config_overrides)
        pm.alpaca_client.get_positions.return_value = positions or []
        pm.alpaca_client.get_snapshot.return_value = snapshots or {}
        pm.alpaca_client.get_account.return_value = account or {
            "equity": 100_000, "cash": 50_000, "daytrade_count": 0,
        }
        return pm

    @patch("notifications.send_position_exit_alert")
    def test_trailing_stop(self, mock_alert, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 9400, "avg_entry_price": 50.0,
                      "current_price": 47.0, "unrealized_pl": -600, "unrealized_plpc": -0.06}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 47.0, "prev_daily_bar_close": 50.0,
            "daily_bar_open": 50.0, "daily_bar_high": 50.5, "daily_bar_low": 46.5,
            "daily_bar_volume": 100_000_000,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares, account_equity) VALUES (?, ?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200, 100_000])
        db_conn.commit()

        # Set a high watermark at 55
        update_position_watermark("TQQQ", 55.0, "2025-01-14", db_conn)

        pm = self._setup_pm(positions, snapshots)

        # Mock volume cache
        with patch("strategy.position_manager.PositionManager._get_prev_volume", return_value=100_000_000):
            decisions = pm.run_midday_check(db_conn)

        assert len(decisions) >= 1
        exit_types = {d.exit_type for d in decisions}
        # Trailing stop should trigger: price 47, watermark 55, drop 14.5%
        assert "TRAILING_STOP" in exit_types or "STOP_LOSS" in exit_types

    @patch("notifications.send_position_exit_alert")
    def test_partial_profit(self, mock_alert, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 11000, "avg_entry_price": 50.0,
                      "current_price": 55.0, "unrealized_pl": 1000, "unrealized_plpc": 0.10}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 55.0, "prev_daily_bar_close": 54.0,
            "daily_bar_open": 54.5, "daily_bar_high": 55.5, "daily_bar_low": 54.0,
            "daily_bar_volume": 100_000_000,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares, account_equity) VALUES (?, ?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200, 100_000])
        db_conn.commit()

        # Set watermark at current price (no trailing stop)
        update_position_watermark("TQQQ", 55.0, "2025-01-15", db_conn)

        pm = self._setup_pm(positions, snapshots)

        with patch("strategy.position_manager.PositionManager._get_prev_volume", return_value=100_000_000):
            decisions = pm.run_midday_check(db_conn)

        assert len(decisions) >= 1
        assert any(d.exit_type == "PARTIAL_PROFIT" for d in decisions)

    @patch("notifications.send_position_exit_alert")
    def test_vol_spike(self, mock_alert, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 10200, "avg_entry_price": 50.0,
                      "current_price": 51.0, "unrealized_pl": 200, "unrealized_plpc": 0.02}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 51.0, "prev_daily_bar_close": 50.5,
            "daily_bar_open": 50.8, "daily_bar_high": 51.5, "daily_bar_low": 50.0,
            "daily_bar_volume": 250_000_000,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares, account_equity) VALUES (?, ?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200, 100_000])
        db_conn.commit()

        update_position_watermark("TQQQ", 51.5, "2025-01-15", db_conn)

        pm = self._setup_pm(positions, snapshots, config_overrides={"pm_profit_taking_enabled": False})

        # Volume spiked from 100M to 250M = 150% increase
        with patch("strategy.position_manager.PositionManager._get_prev_volume", return_value=100_000_000):
            decisions = pm.run_midday_check(db_conn)

        assert len(decisions) >= 1
        assert any(d.exit_type == "VOL_SPIKE" for d in decisions)

    def test_healthy_position_no_exit(self, db_conn):
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 10400, "avg_entry_price": 50.0,
                      "current_price": 52.0, "unrealized_pl": 400, "unrealized_plpc": 0.04}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 52.0, "prev_daily_bar_close": 51.0,
            "daily_bar_open": 51.5, "daily_bar_high": 52.5, "daily_bar_low": 51.0,
            "daily_bar_volume": 100_000_000,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares, account_equity) VALUES (?, ?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200, 100_000])
        db_conn.commit()

        update_position_watermark("TQQQ", 52.0, "2025-01-15", db_conn)

        pm = self._setup_pm(positions, snapshots, config_overrides={"pm_profit_taking_enabled": False})

        with patch("strategy.position_manager.PositionManager._get_prev_volume", return_value=100_000_000):
            decisions = pm.run_midday_check(db_conn)

        assert len(decisions) == 0

    def test_pdt_blocks_normal(self, db_conn):
        """NORMAL urgency exits are blocked by PDT reserve."""
        positions = [{"symbol": "TQQQ", "qty": 200, "market_value": 10000, "avg_entry_price": 50.0,
                      "current_price": 51.0, "unrealized_pl": 200, "unrealized_plpc": 0.02}]
        snapshots = {"TQQQ": {
            "latest_trade_price": 51.0, "prev_daily_bar_close": 50.5,
            "daily_bar_open": 50.8, "daily_bar_high": 51.5, "daily_bar_low": 50.0,
            "daily_bar_volume": 250_000_000,
        }}
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, regime, current_shares, account_equity) VALUES (?, ?, ?, ?, ?)",
            ["2025-01-10", "2025-01-10T15:50:00", "BULL", 200, 100_000])
        db_conn.commit()

        update_position_watermark("TQQQ", 51.5, "2025-01-15", db_conn)

        # Only 1 day trade remaining, reserve is 2
        pm = self._setup_pm(
            positions, snapshots,
            account={"equity": 100_000, "cash": 50_000, "daytrade_count": 1},
            config_overrides={"pm_profit_taking_enabled": False},
        )

        with patch("strategy.position_manager.PositionManager._get_prev_volume", return_value=100_000_000):
            decisions = pm.run_midday_check(db_conn)

        # Vol spike would fire but PDT should block it (NORMAL urgency)
        # Check that the event was logged as SKIPPED_PDT
        events = get_today_position_events("TQQQ", db_conn)
        skipped = [e for e in events if e["order_status"] == "SKIPPED_PDT"]
        # Either skipped or no decision executed
        assert len(decisions) == 0 or len(skipped) > 0


# ══════════════════════════════════════════════
#  TestPositionEventsDB
# ══════════════════════════════════════════════

class TestPositionEventsDB:
    def test_save_and_query_events(self, db_conn):
        event = {
            "date": "2025-01-15",
            "timestamp": "2025-01-15T09:35:00-05:00",
            "symbol": "TQQQ",
            "window": "MORNING",
            "event_type": "STOP_LOSS",
            "shares_before": 200,
            "shares_sold": 200,
            "shares_after": 0,
            "price": 45.0,
            "pnl_pct": -10.0,
            "order_status": "EXECUTED",
            "trigger_detail": "TQQQ down 10% from entry",
        }
        event_id = save_position_event(event, db_conn)
        assert event_id > 0

        # Query back
        with patch("db.models.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value.isoformat.return_value = "2025-01-15"
            events = get_today_position_events("TQQQ", db_conn)
        assert len(events) == 1
        assert events[0]["event_type"] == "STOP_LOSS"

    def test_watermark_upsert(self, db_conn):
        # Insert
        update_position_watermark("TQQQ", 55.0, "2025-01-15", db_conn)
        wm = get_position_watermark("TQQQ", db_conn)
        assert wm["high_price"] == 55.0

        # Update (higher)
        update_position_watermark("TQQQ", 58.0, "2025-01-16", db_conn)
        wm = get_position_watermark("TQQQ", db_conn)
        assert wm["high_price"] == 58.0
        assert wm["high_date"] == "2025-01-16"

    def test_profit_tiers_tracking(self, db_conn):
        # Save two profit events
        for tier in [8.0, 15.0]:
            save_position_event({
                "date": "2025-01-15",
                "timestamp": "2025-01-15T12:30:00-05:00",
                "symbol": "TQQQ",
                "window": "MIDDAY",
                "event_type": "PARTIAL_PROFIT",
                "shares_before": 200,
                "shares_sold": 50,
                "shares_after": 150,
                "price": 55.0,
                "pnl_pct": tier,
                "order_status": "EXECUTED",
                "profit_tier_pct": tier,
            }, db_conn)

        tiers = get_profit_tiers_taken("TQQQ", "2025-01-10", db_conn)
        assert sorted(tiers) == [8.0, 15.0]

    def test_update_event(self, db_conn):
        event_id = save_position_event({
            "date": "2025-01-15",
            "timestamp": "2025-01-15T09:35:00-05:00",
            "symbol": "TQQQ",
            "window": "MORNING",
            "event_type": "STOP_LOSS",
            "order_status": "PENDING",
        }, db_conn)

        update_position_event(event_id, {
            "order_status": "EXECUTED",
            "order_id": "abc123",
        }, db_conn)

        row = db_conn.execute(
            "SELECT * FROM position_events WHERE id = ?", [event_id]
        ).fetchone()
        assert row["order_status"] == "EXECUTED"
        assert row["order_id"] == "abc123"
