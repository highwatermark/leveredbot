"""Tests for order execution logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from strategy.executor import execute_rebalance, force_exit


class TestExecuteRebalance:
    def _mock_client(self, order_result=None, raises=None):
        client = MagicMock()
        if raises:
            client.submit_market_order.side_effect = raises
        else:
            client.submit_market_order.return_value = order_result or {
                "order_id": "test-123",
                "status": "accepted",
                "symbol": "TQQQ",
                "qty": 100,
                "side": "buy",
                "filled_avg_price": None,
                "filled_qty": 0,
            }
        return client

    def test_buy_order(self):
        client = self._mock_client()
        result = execute_rebalance(200, 100, 50.0, client)
        assert result["executed"] is True
        assert result["action"] == "BUY"
        assert result["shares"] == 100
        client.submit_market_order.assert_called_once_with("TQQQ", 100, "buy")

    def test_sell_order(self):
        client = self._mock_client()
        result = execute_rebalance(50, 200, 50.0, client)
        assert result["executed"] is True
        assert result["action"] == "SELL"
        assert result["shares"] == 150
        client.submit_market_order.assert_called_once_with("TQQQ", 150, "sell")

    def test_below_min_trade_value(self):
        """Delta < $100 → no trade."""
        client = self._mock_client()
        result = execute_rebalance(201, 200, 50.0, client)  # $50 delta
        assert result["executed"] is False
        assert result["action"] == "HOLD"
        client.submit_market_order.assert_not_called()

    def test_pdt_blocks_rebalance(self):
        """Not enough day trades → skip non-emergency rebalance."""
        client = self._mock_client()
        result = execute_rebalance(300, 100, 50.0, client, day_trades_remaining=1)
        assert result["executed"] is False
        assert "PDT" in result["reason"]
        client.submit_market_order.assert_not_called()

    def test_pdt_allows_emergency(self):
        """Emergency exit bypasses PDT check."""
        client = self._mock_client()
        result = execute_rebalance(0, 300, 50.0, client, is_emergency=True, day_trades_remaining=0)
        assert result["executed"] is True
        assert result["action"] == "SELL"

    def test_emergency_ignores_min_trade(self):
        """Emergency exit with small delta still executes."""
        client = self._mock_client()
        result = execute_rebalance(0, 1, 50.0, client, is_emergency=True)
        assert result["executed"] is True

    def test_no_change_needed(self):
        client = self._mock_client()
        result = execute_rebalance(200, 200, 50.0, client)
        assert result["executed"] is False
        assert result["action"] == "HOLD"

    def test_order_rejection(self):
        """API error → graceful failure."""
        client = self._mock_client(raises=Exception("Insufficient buying power"))
        result = execute_rebalance(500, 0, 50.0, client)
        assert result["executed"] is False
        assert result["action"] == "ERROR"
        assert "rejected" in result["reason"].lower()


class TestForceExit:
    def test_force_exit_with_position(self):
        client = MagicMock()
        client.get_tqqq_position.return_value = {"qty": 300, "market_value": 15000}
        client.submit_market_order.return_value = {
            "order_id": "exit-456",
            "status": "accepted",
            "symbol": "TQQQ",
            "qty": 300,
            "side": "sell",
            "filled_avg_price": None,
            "filled_qty": 0,
        }
        result = force_exit(client)
        assert result["executed"] is True
        assert result["shares_sold"] == 300
        client.submit_market_order.assert_called_once_with("TQQQ", 300, "sell")

    def test_force_exit_no_position(self):
        client = MagicMock()
        client.get_tqqq_position.return_value = None
        result = force_exit(client)
        assert result["executed"] is False
        assert "No TQQQ" in result["reason"]

    def test_force_exit_api_error(self):
        client = MagicMock()
        client.get_tqqq_position.return_value = {"qty": 200, "market_value": 10000}
        client.submit_market_order.side_effect = Exception("API timeout")
        result = force_exit(client)
        assert result["executed"] is False
        assert "failed" in result["reason"].lower()
