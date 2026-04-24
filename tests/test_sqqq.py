"""Tests for SQQQ (inverse) trading functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from unittest.mock import patch
from strategy.sizing import (
    get_allocated_capital,
    run_sqqq_gate_checklist,
    calculate_sqqq_target_shares,
)


class TestSqqqGateChecklist:
    def _make_gate_data(self, **overrides):
        """Create default SQQQ gate data that passes all 9 gates."""
        data = {
            "knn_direction": "SHORT",
            "knn_confidence": 0.70,
            "vol_regime": "NORMAL",
            "allocated_capital": 30000,
            "is_execution_window": True,
            "day_trades_remaining": 5,
            "trading_days_fetched": 278,
            "has_tqqq_position": False,
            "regime": "BULL",
        }
        data.update(overrides)
        return data

    def test_all_gates_pass(self):
        data = self._make_gate_data()
        passed, failed = run_sqqq_gate_checklist(data)
        assert passed is True
        assert failed == []

    def test_gate_s1_knn_not_short(self):
        data = self._make_gate_data(knn_direction="LONG")
        passed, failed = run_sqqq_gate_checklist(data)
        assert "knn_not_short" in failed

    def test_gate_s1_knn_flat(self):
        data = self._make_gate_data(knn_direction="FLAT")
        passed, failed = run_sqqq_gate_checklist(data)
        assert "knn_not_short" in failed

    def test_gate_s2_knn_confidence_too_low(self):
        data = self._make_gate_data(knn_confidence=0.50)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "knn_confidence" in failed

    def test_gate_s2_knn_confidence_at_threshold(self):
        """Exactly at threshold (0.55) should pass."""
        data = self._make_gate_data(knn_confidence=0.55)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "knn_confidence" not in failed

    def test_gate_s3_vol_extreme(self):
        data = self._make_gate_data(vol_regime="EXTREME")
        passed, failed = run_sqqq_gate_checklist(data)
        assert "vol_extreme" in failed

    def test_gate_s3_vol_high_passes(self):
        """HIGH vol should still pass (only EXTREME blocks)."""
        data = self._make_gate_data(vol_regime="HIGH")
        passed, failed = run_sqqq_gate_checklist(data)
        assert "vol_extreme" not in failed

    def test_gate_s4_capital(self):
        data = self._make_gate_data(allocated_capital=50)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "capital" in failed

    def test_gate_s5_execution_window(self):
        data = self._make_gate_data(is_execution_window=False)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "execution_window" in failed

    def test_gate_s6_pdt(self):
        data = self._make_gate_data(day_trades_remaining=1)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "pdt" in failed

    def test_gate_s7_data_quality(self):
        data = self._make_gate_data(trading_days_fetched=200)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "data_quality" in failed

    def test_gate_s8_mutual_exclusivity(self):
        """Cannot enter SQQQ when holding TQQQ."""
        data = self._make_gate_data(has_tqqq_position=True)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "mutual_exclusivity" in failed

    def test_gate_s9_regime_risk_off(self):
        data = self._make_gate_data(regime="RISK_OFF")
        passed, failed = run_sqqq_gate_checklist(data)
        assert "regime_risk_off" in failed

    def test_gate_s9_regime_breakdown(self):
        data = self._make_gate_data(regime="BREAKDOWN")
        passed, failed = run_sqqq_gate_checklist(data)
        assert "regime_risk_off" in failed

    def test_multiple_gates_fail(self):
        data = self._make_gate_data(
            knn_direction="FLAT",
            knn_confidence=0.3,
            has_tqqq_position=True,
            regime="RISK_OFF",
        )
        passed, failed = run_sqqq_gate_checklist(data)
        assert len(failed) >= 4


class TestCalculateSqqqTargetShares:
    def _make_sizing_data(self, **overrides):
        data = {
            "knn_direction": "SHORT",
            "knn_confidence": 0.70,
            "vol_regime": "NORMAL",
            "allocated_capital": 30000,
            "sqqq_price": 12.50,
            "current_shares": 0,
        }
        data.update(overrides)
        return data

    def test_sizing_at_min_confidence(self):
        """At 0.55 confidence → 50% of max (40%) = 20% allocation."""
        data = self._make_sizing_data(knn_confidence=0.55)
        result = calculate_sqqq_target_shares(data)
        # 0.40 * 0.50 = 0.20 → 30000 * 0.20 = 6000 / 12.50 = 480 shares
        assert result["target_shares"] == 480
        assert result["symbol"] == "SQQQ"
        assert result["action"] == "BUY"

    def test_sizing_at_mid_confidence(self):
        """At 0.675 confidence → 75% of max (40%) = 30% allocation."""
        data = self._make_sizing_data(knn_confidence=0.675)
        result = calculate_sqqq_target_shares(data)
        # 0.40 * (0.5 + 0.5 * 0.5) = 0.40 * 0.75 = 0.30 → 30000 * 0.30 = 9000 / 12.50 = 720
        assert result["target_shares"] == 720

    def test_sizing_at_max_confidence(self):
        """At 0.80+ confidence → 100% of max (40%) allocation."""
        data = self._make_sizing_data(knn_confidence=0.85)
        result = calculate_sqqq_target_shares(data)
        # 0.40 → 30000 * 0.40 = 12000 / 12.50 = 960
        assert result["target_shares"] == 960

    def test_sizing_below_threshold_no_trend(self):
        """Below min confidence with no trend override → 0 shares."""
        with patch.dict("config.LEVERAGE_CONFIG", {"sqqq_trend_override": False}, clear=False):
            data = self._make_sizing_data(knn_direction="FLAT", knn_confidence=0.50)
            result = calculate_sqqq_target_shares(data)
            assert result["target_shares"] == 0

    def test_sizing_trend_override(self):
        """Non-SHORT k-NN with trend override → conservative sizing."""
        data = self._make_sizing_data(knn_direction="LONG", knn_confidence=0.60)
        result = calculate_sqqq_target_shares(data)
        # trend_override: 0.40 * 0.30 = 0.12 → 30000 * 0.12 = 3600 / 12.50 = 288
        assert result["target_shares"] == 288
        assert "trend_override" in str(result["limiting_factors"])

    def test_vol_high_adjustment(self):
        """HIGH vol halves the allocation."""
        data = self._make_sizing_data(knn_confidence=0.80, vol_regime="HIGH")
        result = calculate_sqqq_target_shares(data)
        # 0.40 * 0.5 (vol adj) = 0.20 → 30000 * 0.20 = 6000 / 12.50 = 480
        assert result["target_shares"] == 480
        assert "vol=HIGH" in str(result["limiting_factors"])

    def test_vol_extreme_zeroes(self):
        """EXTREME vol → 0 shares."""
        data = self._make_sizing_data(knn_confidence=0.80, vol_regime="EXTREME")
        result = calculate_sqqq_target_shares(data)
        assert result["target_shares"] == 0

    def test_exit_action_when_holding(self):
        """When k-NN not SHORT and holding shares with no trend override → EXIT."""
        with patch.dict("config.LEVERAGE_CONFIG", {"sqqq_trend_override": False}, clear=False):
            data = self._make_sizing_data(knn_direction="FLAT", knn_confidence=0.50, current_shares=500)
            result = calculate_sqqq_target_shares(data)
            assert result["target_shares"] == 0
            assert result["action"] == "EXIT"

    def test_hold_when_delta_below_min_trade(self):
        """Small delta → HOLD."""
        data = self._make_sizing_data(knn_confidence=0.675, current_shares=719)
        result = calculate_sqqq_target_shares(data)
        # Target is 720, current is 719, delta = 1 * 12.50 = $12.50 < $100
        assert result["action"] == "HOLD"

    def test_zero_price_no_crash(self):
        data = self._make_sizing_data(sqqq_price=0)
        result = calculate_sqqq_target_shares(data)
        assert result["target_shares"] == 0

    def test_custom_config_values(self):
        """Custom sqqq_max_position_pct and sqqq_min_knn_confidence."""
        with patch.dict("config.LEVERAGE_CONFIG", {
            "sqqq_max_position_pct": 0.50,
            "sqqq_min_knn_confidence": 0.55,
            "sqqq_trend_override": False,
            "min_trade_value": 100,
        }):
            data = self._make_sizing_data(knn_confidence=0.55)
            result = calculate_sqqq_target_shares(data)
            # 0.50 * 0.50 = 0.25 → 30000 * 0.25 = 7500 / 12.50 = 600
            assert result["target_shares"] == 600


class TestSqqqTrendOverrideGate:
    """Test that trend override allows SQQQ entry without k-NN SHORT."""

    def _make_gate_data(self, **overrides):
        data = {
            "knn_direction": "LONG",
            "knn_confidence": 0.60,
            "vol_regime": "NORMAL",
            "allocated_capital": 30000,
            "is_execution_window": True,
            "day_trades_remaining": 5,
            "trading_days_fetched": 278,
            "has_tqqq_position": False,
            "regime": "BULL",
            "pct_above_sma50": -0.05,
            "roc_slow": -0.03,
        }
        data.update(overrides)
        return data

    def test_trend_override_passes_without_knn_short(self):
        """QQQ deeply below SMA-50 with negative momentum → pass via trend override."""
        data = self._make_gate_data()
        passed, failed = run_sqqq_gate_checklist(data)
        assert passed is True
        assert "knn_not_short" not in failed

    def test_trend_override_fails_when_not_bearish_enough(self):
        """QQQ only slightly below SMA-50 → no trend override, fails on k-NN."""
        data = self._make_gate_data(pct_above_sma50=-0.01, roc_slow=-0.03)
        passed, failed = run_sqqq_gate_checklist(data)
        assert passed is False
        assert "knn_not_short" in failed

    def test_trend_override_fails_when_momentum_not_negative(self):
        """QQQ below SMA-50 but momentum positive → no trend override."""
        data = self._make_gate_data(pct_above_sma50=-0.05, roc_slow=0.01)
        passed, failed = run_sqqq_gate_checklist(data)
        assert passed is False

    def test_trend_override_disabled(self):
        """When sqqq_trend_override is False, must rely on k-NN."""
        with patch.dict("config.LEVERAGE_CONFIG", {"sqqq_trend_override": False}, clear=False):
            data = self._make_gate_data()
            passed, failed = run_sqqq_gate_checklist(data)
            assert passed is False
            assert "knn_not_short" in failed

    def test_knn_short_still_works(self):
        """k-NN SHORT path still works even with trend override enabled."""
        data = self._make_gate_data(
            knn_direction="SHORT", knn_confidence=0.65,
            pct_above_sma50=0.02, roc_slow=0.01,  # no trend signal
        )
        passed, failed = run_sqqq_gate_checklist(data)
        assert passed is True


class TestSqqqGateRotationBypass:
    """Test that Gate S8 (mutual_exclusivity) is bypassed during TQQQ→SQQQ rotation."""

    def _make_gate_data(self, **overrides):
        data = {
            "knn_direction": "SHORT",
            "knn_confidence": 0.70,
            "vol_regime": "NORMAL",
            "allocated_capital": 30000,
            "is_execution_window": True,
            "day_trades_remaining": 5,
            "trading_days_fetched": 278,
            "has_tqqq_position": True,
            "regime": "BULL",
            "tqqq_just_exited": False,
        }
        data.update(overrides)
        return data

    def test_gate_s8_blocks_without_rotation(self):
        """Without rotation flag, Gate S8 still blocks when holding TQQQ."""
        data = self._make_gate_data(has_tqqq_position=True, tqqq_just_exited=False)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "mutual_exclusivity" in failed

    def test_gate_s8_bypassed_on_rotation(self):
        """With tqqq_just_exited=True, Gate S8 is bypassed."""
        data = self._make_gate_data(has_tqqq_position=True, tqqq_just_exited=True)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "mutual_exclusivity" not in failed
        assert passed is True

    def test_gate_s8_bypassed_but_other_gates_still_checked(self):
        """Rotation bypass only affects S8, other gates still enforced."""
        data = self._make_gate_data(
            has_tqqq_position=True,
            tqqq_just_exited=True,
            knn_confidence=0.50,  # Below threshold
        )
        passed, failed = run_sqqq_gate_checklist(data)
        assert "mutual_exclusivity" not in failed
        assert "knn_confidence" in failed
        assert passed is False

    def test_no_tqqq_position_no_flag_needed(self):
        """When not holding TQQQ, rotation flag is irrelevant."""
        data = self._make_gate_data(has_tqqq_position=False, tqqq_just_exited=False)
        passed, failed = run_sqqq_gate_checklist(data)
        assert "mutual_exclusivity" not in failed


class TestCapitalAllocationWithSqqq:
    def test_sqqq_not_counted_as_other(self):
        """SQQQ position should not be counted as 'other' — it's strategy capital."""
        positions = [
            {"symbol": "BRK.B", "market_value": 10000},
            {"symbol": "TQQQ", "market_value": 8000},
            {"symbol": "SQQQ", "market_value": 5000},
        ]
        result = get_allocated_capital(100000, positions)
        assert result["other_positions_value"] == 10000
        assert result["tqqq_position_value"] == 8000
        assert result["sqqq_position_value"] == 5000
        assert result["strategy_position_value"] == 13000
        # cash_available = 100000 - 10000 (other) - 13000 (strategy) = 77000
        assert result["cash_available"] == 77000

    def test_sqqq_only_position(self):
        """SQQQ as only position."""
        positions = [
            {"symbol": "SQQQ", "market_value": 5000},
        ]
        result = get_allocated_capital(50000, positions)
        assert result["other_positions_value"] == 0
        assert result["sqqq_position_value"] == 5000
        assert result["tqqq_position_value"] == 0
        assert result["allocated_capital"] == 15000  # 30% of 50K
        assert result["cash_available"] == 45000  # 50K - 0 - 5K

    def test_no_sqqq_backward_compatible(self):
        """Without SQQQ, result still has sqqq fields with 0 values."""
        positions = [
            {"symbol": "TQQQ", "market_value": 10000},
        ]
        result = get_allocated_capital(100000, positions)
        assert result["sqqq_position_value"] == 0
        assert result["strategy_position_value"] == 10000
        assert result["cash_available"] == 90000  # 100K - 0 - 10K
