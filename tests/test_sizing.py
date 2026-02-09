"""Tests for position sizing and gate checklist."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.sizing import get_allocated_capital, run_gate_checklist, calculate_target_shares


class TestAllocatedCapital:
    def test_basic_allocation(self, mock_positions):
        result = get_allocated_capital(129000, mock_positions)
        assert result["total_equity"] == 129000
        assert result["other_positions_value"] == pytest.approx(38000, abs=100)
        assert result["tqqq_position_value"] == pytest.approx(10128, abs=100)
        # 30% of 129K = 38.7K, or 129K - 38K = 91K, whichever is less
        assert result["allocated_capital"] <= 129000 * 0.30
        assert result["allocated_capital"] > 0

    def test_no_positions(self):
        result = get_allocated_capital(100000, [])
        assert result["allocated_capital"] == 30000  # 30% of 100K
        assert result["other_positions_value"] == 0

    def test_heavy_other_positions(self):
        """Other positions take most equity → little left for TQQQ."""
        positions = [
            {"symbol": "AAPL", "market_value": 80000},
            {"symbol": "GOOG", "market_value": 15000},
        ]
        result = get_allocated_capital(100000, positions)
        # 100K - 95K = 5K available, but 30% cap = 30K
        assert result["allocated_capital"] == 5000

    def test_zero_equity(self):
        result = get_allocated_capital(0, [])
        assert result["allocated_capital"] == 0

    def test_custom_max_pct(self, mock_positions_no_tqqq):
        result = get_allocated_capital(100000, mock_positions_no_tqqq, max_portfolio_pct=0.50)
        assert result["allocated_capital"] <= 50000


class TestGateChecklist:
    def _make_gate_data(self, **overrides):
        """Create default gate data that passes all gates."""
        np.random.seed(42)
        # Use enough slope so last 30 days have >5% range (not sideways)
        closes = [400 + i * 1.0 + np.random.normal(0, 3) for i in range(300)]

        data = {
            "regime": "STRONG_BULL",
            "qqq_close": 550,
            "sma_50": 520,
            "sma_250": 480,
            "momentum_score": 0.7,
            "realized_vol": 18,
            "vol_regime": "NORMAL",
            "daily_loss_pct": 0.01,
            "qqq_closes": closes,
            "holding_days_losing": 0,
            "is_execution_window": True,
            "allocated_capital": 30000,
            "day_trades_remaining": 5,
            "options_flow_bearish": False,
            "options_flow_ratio": 0.8,
            "trading_days_fetched": 278,
        }
        data.update(overrides)
        return data

    def test_all_gates_pass(self):
        data = self._make_gate_data()
        passed, failed = run_gate_checklist(data)
        assert passed is True
        assert failed == []

    def test_gate_regime_fail(self):
        data = self._make_gate_data(regime="RISK_OFF")
        passed, failed = run_gate_checklist(data)
        assert passed is False
        assert "regime" in failed

    def test_gate_breakdown_fail(self):
        data = self._make_gate_data(regime="BREAKDOWN")
        passed, failed = run_gate_checklist(data)
        assert "regime" in failed

    def test_gate_trend_strength_fail(self):
        # Close barely above both SMAs (less than 2%)
        data = self._make_gate_data(qqq_close=501, sma_50=500, sma_250=500)
        passed, failed = run_gate_checklist(data)
        assert "trend_strength" in failed

    def test_gate_momentum_fail(self):
        data = self._make_gate_data(momentum_score=0.1)
        passed, failed = run_gate_checklist(data)
        assert "momentum" in failed

    def test_gate_vol_extreme_fail(self):
        data = self._make_gate_data(realized_vol=40, vol_regime="EXTREME")
        passed, failed = run_gate_checklist(data)
        assert "vol_extreme" in failed

    def test_gate_daily_loss_fail(self):
        data = self._make_gate_data(daily_loss_pct=0.09)
        passed, failed = run_gate_checklist(data)
        assert "daily_loss" in failed

    def test_gate_execution_window_fail(self):
        data = self._make_gate_data(is_execution_window=False)
        passed, failed = run_gate_checklist(data)
        assert "execution_window" in failed

    def test_gate_capital_fail(self):
        data = self._make_gate_data(allocated_capital=50)
        passed, failed = run_gate_checklist(data)
        assert "capital" in failed

    def test_gate_pdt_fail(self):
        data = self._make_gate_data(day_trades_remaining=1)
        passed, failed = run_gate_checklist(data)
        assert "pdt" in failed

    def test_gate_data_quality_fail(self):
        data = self._make_gate_data(trading_days_fetched=200)
        passed, failed = run_gate_checklist(data)
        assert "data_quality" in failed

    def test_gate_extreme_bearish_flow_fail(self):
        """Extremely bearish flow (ratio > 3x) blocks entry."""
        data = self._make_gate_data(options_flow_bearish=True, options_flow_ratio=3.5)
        passed, failed = run_gate_checklist(data)
        assert "flow_sentiment" in failed

    def test_gate_moderate_bearish_flow_passes(self):
        """Moderate bearish flow (ratio 2-3x) still passes but reduces allocation."""
        data = self._make_gate_data(options_flow_bearish=True, options_flow_ratio=2.5)
        passed, failed = run_gate_checklist(data)
        assert "flow_sentiment" not in failed

    def test_gate_holding_days_fail(self):
        data = self._make_gate_data(holding_days_losing=20)
        passed, failed = run_gate_checklist(data)
        assert "holding_days" in failed

    def test_multiple_gates_fail(self):
        data = self._make_gate_data(regime="RISK_OFF", momentum_score=0.1, realized_vol=40)
        passed, failed = run_gate_checklist(data)
        assert len(failed) >= 3


class TestCalculateTargetShares:
    def test_strong_bull_full_allocation(self):
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": [500 + i * 0.2 for i in range(300)],
            "tqqq_price": 50.64,
            "current_shares": 0,
        }
        result = calculate_target_shares(data)
        # 70% of 38.7K = 27.09K / 50.64 = ~535 shares
        assert result["target_shares"] > 400
        assert result["target_shares"] < 600
        assert result["action"] == "BUY"

    def test_risk_off_exit(self):
        data = {
            "regime": "RISK_OFF",
            "allocated_capital": 38700,
            "momentum_score": 0.5,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 450,
            "sma_50": 500,
            "qqq_closes": [500 - i * 0.1 for i in range(300)],
            "tqqq_price": 30.0,
            "current_shares": 300,
        }
        result = calculate_target_shares(data)
        assert result["target_shares"] == 0
        assert result["action"] == "EXIT"

    def test_hold_when_at_target(self):
        data = {
            "regime": "BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 530,
            "sma_50": 510,
            "qqq_closes": [490 + i * 0.15 for i in range(300)],
            "tqqq_price": 50.0,
            "current_shares": 387,  # 50% of 38.7K / 50 = 387
        }
        result = calculate_target_shares(data)
        assert result["action"] == "HOLD"

    def test_vol_high_halves_allocation(self):
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "HIGH",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": [500 + i * 0.2 for i in range(300)],
            "tqqq_price": 50.0,
            "current_shares": 0,
        }
        result = calculate_target_shares(data)
        # 70% * 0.5 (vol) = 35% of 38.7K = 13.5K / 50 = ~270
        assert result["target_shares"] < 300
        assert "vol=HIGH" in str(result["limiting_factors"])

    def test_vol_extreme_goes_to_cash(self):
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "EXTREME",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": [500 + i * 0.2 for i in range(300)],
            "tqqq_price": 50.0,
            "current_shares": 300,
        }
        result = calculate_target_shares(data)
        assert result["target_shares"] == 0

    def test_bearish_flow_reduces(self):
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 0.75,  # 25% reduction
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": [500 + i * 0.2 for i in range(300)],
            "tqqq_price": 50.0,
            "current_shares": 0,
        }
        result = calculate_target_shares(data)
        # 70% * 0.75 = 52.5% of 38.7K = 20.3K / 50 = ~406
        assert result["target_shares"] < 430
        assert "flow_bearish" in str(result["limiting_factors"])

    def test_low_momentum_reduces_to_minimum(self):
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.2,  # Below 0.3 threshold
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": [500 + i * 0.2 for i in range(300)],
            "tqqq_price": 50.0,
            "current_shares": 0,
        }
        result = calculate_target_shares(data)
        # min_position_pct = 0.10 → 10% of 38.7K = 3870 / 50 = ~77
        assert result["target_shares"] < 100

    def test_negative_shares_capped_at_zero(self):
        data = {
            "regime": "RISK_OFF",
            "allocated_capital": 0,
            "momentum_score": 0,
            "vol_regime": "EXTREME",
            "options_flow_adjustment": 1.0,
            "qqq_close": 400,
            "sma_50": 450,
            "qqq_closes": [500 - i for i in range(300)],
            "tqqq_price": 30.0,
            "current_shares": 0,
        }
        result = calculate_target_shares(data)
        assert result["target_shares"] >= 0
