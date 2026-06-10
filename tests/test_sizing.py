"""Tests for position sizing and gate checklist."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from unittest.mock import patch
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
    @pytest.fixture(autouse=True)
    def _legacy_path(self):
        """Gate checklist tests target the legacy (non-sleeve) path."""
        with patch.dict("config.LEVERAGE_CONFIG", {"use_rule_sleeves": False}, clear=False):
            yield

    def _make_gate_data(self, **overrides):
        """Create default gate data that passes all gates (incl. RSI overbought)."""
        np.random.seed(123)
        # Slope + noise tuned to pass both sideways (>5% range) and RSI (<70)
        closes = [400 + i * 0.7 + np.random.normal(0, 5) for i in range(300)]

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
            "current_shares": 0,
            "knn_direction": "LONG",
            "knn_confidence": 0.65,
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

    def test_gate_rsi_overbought_does_not_block(self):
        """RSI above threshold is handled in sizing, not as a hard gate failure."""
        # Monotonically rising closes → RSI near 100
        closes = list(np.linspace(400, 550, 300))
        data = self._make_gate_data(qqq_closes=closes)
        passed, failed = run_gate_checklist(data)
        assert "rsi_overbought" not in failed

    def test_multiple_gates_fail(self):
        data = self._make_gate_data(regime="RISK_OFF", momentum_score=0.1, realized_vol=40)
        passed, failed = run_gate_checklist(data)
        assert len(failed) >= 3

    def test_gate_model_bearish_blocks_bull_regime_entry(self):
        with patch.dict("config.LEVERAGE_CONFIG", {
            "long_model_short_block_confidence": 0.60,
            "long_model_short_bull_max_position_pct": 0.0,
            "long_model_short_strong_bull_max_position_pct": 0.0,
        }, clear=False):
            data = self._make_gate_data(knn_direction="SHORT", knn_confidence=0.65)
            passed, failed = run_gate_checklist(data)
            assert passed is False
            assert "model_bearish" in failed

    def test_gate_model_neutral_blocks_new_entry(self):
        with patch.dict("config.LEVERAGE_CONFIG", {"long_require_model_support_for_new_entries": True}, clear=False):
            data = self._make_gate_data(knn_direction="FLAT", knn_confidence=0.50, current_shares=0)
            passed, failed = run_gate_checklist(data)
            assert passed is False
            assert "model_neutral" in failed


class TestCalculateTargetShares:
    @pytest.fixture(autouse=True)
    def _legacy_path(self):
        """Sizing tests target the legacy (non-sleeve) math; sleeve engine has its own tests.
        Regime targets pinned to the values these expectations were written against."""
        with patch.dict(
            "config.LEVERAGE_CONFIG",
            {
                "use_rule_sleeves": False,
                "max_position_pct": 0.20,
                "bull_position_pct": 0.15,
                "cautious_position_pct": 0.08,
            },
            clear=False,
        ):
            yield

    def test_strong_bull_full_allocation(self):
        closes = [500 + (i % 9) * 0.4 + ((-1) ** i) * 0.8 for i in range(300)]
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": closes,
            "tqqq_price": 50.64,
            "current_shares": 0,
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        # 20% of 38.7K = 7.74K / 50.64 = ~152 shares
        assert result["target_shares"] > 140
        assert result["target_shares"] < 180
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
            "model_direction": "FLAT",
            "model_confidence": 0.5,
            "model_disagreement": False,
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
            "current_shares": 116,  # 15% of 38.7K / 50 = ~116
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
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
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        # 20% * 0.5 (vol) = 10% of 38.7K = 3.87K / 50 = ~77
        assert result["target_shares"] < 100
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
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
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
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        # 20% * 0.75 = 15% of 38.7K = 5.8K / 50 = ~116
        assert result["target_shares"] < 130
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
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
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

    def test_overbought_flat_book_still_buys_reduced_tqqq(self):
        closes = list(np.linspace(400, 550, 300))
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": closes,
            "tqqq_price": 50.0,
            "current_shares": 0,
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        assert result["action"] == "BUY"
        assert result["target_shares"] > 0
        assert result["target_shares"] < 130
        assert "rsi_overbought_pause_adds" in str(result["limiting_factors"])

    def test_overbought_existing_position_pauses_adds(self):
        closes = list(np.linspace(400, 550, 300))
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": closes,
            "tqqq_price": 50.0,
            "current_shares": 120,
            "model_direction": "LONG",
            "model_confidence": 0.65,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        assert result["target_shares"] == 120
        assert result["action"] == "HOLD"

    def test_model_neutral_flat_book_gets_capped_long(self):
        closes = [500 + (i % 7) * 0.3 + ((-1) ** i) * 0.7 for i in range(300)]
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": closes,
            "tqqq_price": 50.0,
            "current_shares": 0,
            "model_direction": "FLAT",
            "model_confidence": 0.50,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        assert result["target_shares"] > 0
        assert result["target_shares"] <= int((38700 * 0.18) / 50.0)
        assert "model_neutral_cap" in str(result["limiting_factors"])

    def test_model_disagreement_caps_existing_long(self):
        closes = [500 + (i % 7) * 0.3 + ((-1) ** i) * 0.7 for i in range(300)]
        with patch.dict("config.LEVERAGE_CONFIG", {"long_disagreement_max_position_pct": 0.10}, clear=False):
            data = {
                "regime": "STRONG_BULL",
                "allocated_capital": 38700,
                "momentum_score": 0.85,
                "vol_regime": "NORMAL",
                "options_flow_adjustment": 1.0,
                "qqq_close": 550,
                "sma_50": 520,
                "qqq_closes": closes,
                "tqqq_price": 50.0,
                "current_shares": 300,
                "model_direction": "LONG",
                "model_confidence": 0.62,
                "model_disagreement": True,
            }
            result = calculate_target_shares(data)
            assert result["target_shares"] <= int((38700 * 0.10) / 50.0)
            assert result["action"] == "SELL"
            assert "model_disagreement_cap" in str(result["limiting_factors"])

    def test_model_bearish_strong_bull_keeps_reduced_long(self):
        data = {
            "regime": "STRONG_BULL",
            "allocated_capital": 38700,
            "momentum_score": 0.85,
            "vol_regime": "NORMAL",
            "options_flow_adjustment": 1.0,
            "qqq_close": 550,
            "sma_50": 520,
            "qqq_closes": [500 + i * 0.2 for i in range(300)],
            "tqqq_price": 50.0,
            "current_shares": 0,
            "model_direction": "SHORT",
            "model_confidence": 0.72,
            "model_disagreement": False,
        }
        result = calculate_target_shares(data)
        assert result["target_shares"] > 0
        assert result["target_shares"] <= int((38700 * 0.15) / 50.0)
        assert result["action"] == "BUY"
        assert "model_bearish_cap_strong_bull" in str(result["limiting_factors"])
