"""Tests for the cash-yield sweep (idle allocated capital parked in SGOV)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from strategy.cash_sweep import (
    calculate_sweep_target,
    cash_needed_for_buy,
    execute_sweep,
    free_cash_for_buy,
)
from strategy.sizing import get_allocated_capital


@pytest.fixture(autouse=True)
def _sweep_on():
    with patch.dict(
        "config.LEVERAGE_CONFIG",
        {
            "use_cash_sweep": True,
            "sweep_etf": "SGOV",
            "sweep_buffer_pct": 0.02,
            "sweep_min_trade_value": 250,
        },
        clear=False,
    ):
        yield


class TestCalculateSweepTarget:
    def test_idle_capital_swept(self):
        """Allocated 38k, TQQQ 17k → idle minus 2% buffer goes to SGOV."""
        result = calculate_sweep_target(
            allocated_capital=38000,
            strategy_position_value=17000,
            sweep_position_value=0,
            sweep_price=100.0,
        )
        # idle = 38000 - 17000 - 760 (2% buffer) = 20240 → 202 shares
        assert result["action"] == "BUY"
        assert result["target_shares"] == 202
        assert result["delta_shares"] == 202

    def test_no_idle_when_fully_deployed(self):
        result = calculate_sweep_target(
            allocated_capital=38000,
            strategy_position_value=38000,
            sweep_position_value=0,
            sweep_price=100.0,
        )
        assert result["action"] == "HOLD"
        assert result["target_shares"] == 0

    def test_sells_sweep_when_strategy_needs_room(self):
        """TQQQ target grew: SGOV overweight → SELL down to idle."""
        result = calculate_sweep_target(
            allocated_capital=38000,
            strategy_position_value=30000,
            sweep_position_value=20000,
            sweep_price=100.0,
        )
        # idle = 38000 - 30000 - 760 = 7240 → 72 shares; delta = 72 - 200 = -128
        assert result["action"] == "SELL"
        assert result["delta_shares"] == -128

    def test_small_delta_holds(self):
        """Deltas under sweep_min_trade_value don't churn."""
        result = calculate_sweep_target(
            allocated_capital=38000,
            strategy_position_value=17000,
            sweep_position_value=20200,
            sweep_price=100.0,
        )
        # target 202 sh vs current 202 sh → ~0 delta
        assert result["action"] == "HOLD"

    def test_disabled_returns_hold(self):
        with patch.dict("config.LEVERAGE_CONFIG", {"use_cash_sweep": False}, clear=False):
            result = calculate_sweep_target(
                allocated_capital=38000,
                strategy_position_value=0,
                sweep_position_value=0,
                sweep_price=100.0,
            )
            assert result["action"] == "HOLD"

    def test_negative_idle_sells_everything(self):
        """Strategy positions exceed allocation → all sweep shares sold."""
        result = calculate_sweep_target(
            allocated_capital=38000,
            strategy_position_value=40000,
            sweep_position_value=5000,
            sweep_price=100.0,
        )
        assert result["action"] == "SELL"
        assert result["target_shares"] == 0
        assert result["delta_shares"] == -50

    def test_zero_price_no_crash(self):
        result = calculate_sweep_target(
            allocated_capital=38000,
            strategy_position_value=0,
            sweep_position_value=0,
            sweep_price=0,
        )
        assert result["action"] == "HOLD"


class TestCashNeededForBuy:
    def test_enough_cash(self):
        assert cash_needed_for_buy(buy_value=5000, cash_available=10000) == 0

    def test_shortfall_padded(self):
        """Shortfall is padded 1% for price movement between orders."""
        needed = cash_needed_for_buy(buy_value=10000, cash_available=4000)
        assert needed == pytest.approx(6000 * 1.01)

    def test_negative_cash_treated_as_zero(self):
        needed = cash_needed_for_buy(buy_value=1000, cash_available=-500)
        assert needed == pytest.approx(1000 * 1.01)


class TestAllocatedCapitalWithSweep:
    def test_sweep_etf_not_counted_as_other(self):
        """SGOV is strategy capital — must not shrink allocated_capital."""
        positions = [
            {"symbol": "AAPL", "market_value": 10000},
            {"symbol": "TQQQ", "market_value": 8000},
            {"symbol": "SGOV", "market_value": 20000},
        ]
        result = get_allocated_capital(100000, positions)
        assert result["other_positions_value"] == 10000
        assert result["sweep_position_value"] == 20000
        assert result["allocated_capital"] == 30000  # 30% of 100k, unaffected by SGOV

    def test_dedicated_account_allocation_not_shrunk_by_sweep(self):
        """max_portfolio_pct=1.0: sweeping must not reduce next cycle's allocation."""
        positions = [
            {"symbol": "TQQQ", "market_value": 900},
            {"symbol": "SGOV", "market_value": 1050},
        ]
        result = get_allocated_capital(2000, positions, max_portfolio_pct=1.0)
        assert result["allocated_capital"] == 2000
        # cash_available excludes sweep and strategy positions
        assert result["cash_available"] == pytest.approx(50)

    def test_sweep_disabled_sgov_is_other(self):
        """Without the sweep feature, an SGOV position belongs to someone else."""
        with patch.dict("config.LEVERAGE_CONFIG", {"use_cash_sweep": False}, clear=False):
            positions = [{"symbol": "SGOV", "market_value": 5000}]
            result = get_allocated_capital(100000, positions)
            assert result["other_positions_value"] == 5000
            assert result["sweep_position_value"] == 0


class TestExecuteSweep:
    def _make_client(self, positions=None, sgov_price=100.0):
        client = MagicMock()
        client.get_positions.return_value = positions or []
        client.get_snapshot.return_value = {
            "SGOV": {"latest_trade_price": sgov_price, "daily_bar_close": sgov_price}
        }
        client.submit_market_order.return_value = {"order_id": "test-1"}
        return client

    def test_buys_idle_capital(self):
        client = self._make_client()
        result = execute_sweep(client, allocated_capital=38000, strategy_position_value=17000)
        assert result["executed"] is True
        assert result["action"] == "BUY"
        client.submit_market_order.assert_called_once_with("SGOV", 202, "buy")

    def test_skip_buy_after_presell(self):
        """Never sell and rebuy the sweep in the same run."""
        client = self._make_client()
        result = execute_sweep(
            client, allocated_capital=38000, strategy_position_value=17000, skip_buy=True
        )
        assert result["executed"] is False
        client.submit_market_order.assert_not_called()

    def test_sells_when_overweight(self):
        client = self._make_client(
            positions=[{"symbol": "SGOV", "market_value": 20000, "qty": 200}]
        )
        result = execute_sweep(client, allocated_capital=38000, strategy_position_value=30000)
        assert result["action"] == "SELL"
        client.submit_market_order.assert_called_once_with("SGOV", 128, "sell")

    def test_order_failure_is_nonfatal(self):
        client = self._make_client()
        client.submit_market_order.side_effect = Exception("rejected")
        result = execute_sweep(client, allocated_capital=38000, strategy_position_value=0)
        assert result["executed"] is False
        assert result["action"] == "ERROR"


class TestFreeCashForBuy:
    def test_presells_to_cover_shortfall(self):
        client = MagicMock()
        client.get_positions.return_value = [
            {"symbol": "SGOV", "market_value": 20000, "qty": 200}
        ]
        client.submit_market_order.return_value = {"order_id": "test-2"}
        result = free_cash_for_buy(client, buy_value=10000, cash_available=4000)
        assert result["sold"] is True
        # needed = 6000 * 1.01 = 6060 → 61 shares @ $100
        client.submit_market_order.assert_called_once_with("SGOV", 61, "sell")

    def test_no_sale_when_cash_sufficient(self):
        client = MagicMock()
        result = free_cash_for_buy(client, buy_value=5000, cash_available=10000)
        assert result["sold"] is False
        client.submit_market_order.assert_not_called()

    def test_no_sgov_position(self):
        client = MagicMock()
        client.get_positions.return_value = []
        result = free_cash_for_buy(client, buy_value=5000, cash_available=0)
        assert result["sold"] is False
