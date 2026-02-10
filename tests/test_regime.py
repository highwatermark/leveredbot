"""Tests for regime detection state machine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.regime import detect_regime, get_effective_regime, get_regime_target_pct, REGIME_TARGETS


class TestDetectRegime:
    """Test raw regime detection from QQQ price vs SMAs."""

    def test_strong_bull(self):
        """Price above both SMAs + deadzone and golden cross."""
        result = detect_regime(qqq_close=550, sma_50=520, sma_250=480, deadzone_pct=0.02)
        assert result == "STRONG_BULL"

    def test_bull(self):
        """Price above both SMAs but 50-SMA near/below 250-SMA."""
        # Price above sma_50 + deadzone and above sma_250, but sma_50 < sma_250
        result = detect_regime(qqq_close=550, sma_50=520, sma_250=530, deadzone_pct=0.02)
        assert result == "BULL"

    def test_cautious_near_sma50(self):
        """Price above 250-SMA but within deadzone of 50-SMA (binary_mode=False)."""
        # QQQ at 505, SMA50 at 500 (1% above, within 2% deadzone), SMA250 at 480
        result = detect_regime(qqq_close=505, sma_50=500, sma_250=480, deadzone_pct=0.02, binary_mode=False)
        assert result == "CAUTIOUS"

    def test_cautious_below_sma50(self):
        """Price below 50-SMA but above 250-SMA (binary_mode=False)."""
        result = detect_regime(qqq_close=490, sma_50=500, sma_250=480, deadzone_pct=0.02, binary_mode=False)
        assert result == "CAUTIOUS"

    def test_binary_mode_maps_cautious_to_bull(self):
        """In binary mode, CAUTIOUS scenarios return BULL instead."""
        result = detect_regime(qqq_close=490, sma_50=500, sma_250=480, deadzone_pct=0.02, binary_mode=True)
        assert result == "BULL"

    def test_risk_off(self):
        """Price below 250-SMA - deadzone."""
        # SMA250 at 500, deadzone 2% = 490 threshold, price at 485
        result = detect_regime(qqq_close=485, sma_50=510, sma_250=500, deadzone_pct=0.02)
        assert result == "RISK_OFF"

    def test_breakdown(self):
        """Price below 250-SMA - deadzone AND death cross (50 < 250)."""
        result = detect_regime(qqq_close=460, sma_50=470, sma_250=500, deadzone_pct=0.02)
        assert result == "BREAKDOWN"

    def test_deadzone_below_sma250(self):
        """Price in deadzone below 250-SMA (but not below lower band)."""
        # SMA250=500, lower band=490, price=495 (in deadzone)
        result = detect_regime(qqq_close=495, sma_50=510, sma_250=500, deadzone_pct=0.02, binary_mode=False)
        assert result == "CAUTIOUS"

    def test_exact_at_sma50_deadzone(self):
        """Price exactly at 50-SMA + deadzone threshold (binary_mode=False)."""
        # Exactly at the threshold
        result = detect_regime(qqq_close=500 * 1.02, sma_50=500, sma_250=480, deadzone_pct=0.02, binary_mode=False)
        # At boundary, not strictly above
        assert result == "CAUTIOUS"

    def test_default_deadzone(self):
        """Uses config default deadzone when not specified."""
        result = detect_regime(qqq_close=550, sma_50=520, sma_250=480)
        assert result == "STRONG_BULL"

    def test_zero_deadzone(self):
        """Works with zero deadzone."""
        result = detect_regime(qqq_close=501, sma_50=500, sma_250=480, deadzone_pct=0.0)
        assert result == "STRONG_BULL"

    def test_breakdown_takes_priority(self):
        """Breakdown (death cross) overrides risk-off when both conditions met."""
        result = detect_regime(qqq_close=400, sma_50=450, sma_250=500, deadzone_pct=0.02)
        assert result == "BREAKDOWN"

    def test_all_regimes_have_targets(self):
        """Every regime maps to a target allocation."""
        for regime in ["STRONG_BULL", "BULL", "CAUTIOUS", "RISK_OFF", "BREAKDOWN"]:
            pct = get_regime_target_pct(regime)
            assert isinstance(pct, float)
            assert 0 <= pct <= 1.0

    def test_risk_off_target_is_zero(self):
        assert get_regime_target_pct("RISK_OFF") == 0.0

    def test_breakdown_target_is_zero(self):
        assert get_regime_target_pct("BREAKDOWN") == 0.0

    def test_strong_bull_has_highest_target(self):
        assert get_regime_target_pct("STRONG_BULL") > get_regime_target_pct("BULL")
        assert get_regime_target_pct("BULL") > get_regime_target_pct("CAUTIOUS")


class TestEffectiveRegime:
    """Test regime transitions with oscillation protection."""

    def test_cold_start_returns_cautious(self):
        """No previous regime → CAUTIOUS."""
        result = get_effective_regime("STRONG_BULL", previous=None, hold_days=0)
        assert result == "CAUTIOUS"

    def test_same_regime_no_change(self):
        result = get_effective_regime("BULL", previous="BULL", hold_days=5)
        assert result == "BULL"

    def test_risk_off_immediate(self):
        """RISK_OFF always takes effect immediately."""
        result = get_effective_regime("RISK_OFF", previous="BULL", hold_days=0)
        assert result == "RISK_OFF"

    def test_breakdown_immediate(self):
        """BREAKDOWN always takes effect immediately."""
        result = get_effective_regime("BREAKDOWN", previous="STRONG_BULL", hold_days=0)
        assert result == "BREAKDOWN"

    def test_oscillation_protection(self):
        """Non-emergency regime change blocked if hold_days < min."""
        result = get_effective_regime("STRONG_BULL", previous="CAUTIOUS", hold_days=1, min_hold=2)
        assert result == "CAUTIOUS"  # Keeps previous

    def test_regime_change_after_hold_period(self):
        """Regime change allowed after hold period."""
        result = get_effective_regime("STRONG_BULL", previous="CAUTIOUS", hold_days=3, min_hold=2)
        assert result == "STRONG_BULL"

    def test_risk_off_ignores_hold_period(self):
        """RISK_OFF bypasses hold period check."""
        result = get_effective_regime("RISK_OFF", previous="BULL", hold_days=0, min_hold=10)
        assert result == "RISK_OFF"

    def test_bull_to_cautious_requires_hold(self):
        result = get_effective_regime("CAUTIOUS", previous="BULL", hold_days=0, min_hold=2)
        assert result == "BULL"  # Kept due to hold period

    def test_default_min_hold(self):
        """Uses config default when min_hold not specified."""
        result = get_effective_regime("BULL", previous="CAUTIOUS", hold_days=0)
        assert result == "CAUTIOUS"  # Blocked by default min_hold=2
