"""Tests for momentum, volatility, and flow signal calculations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.signals import (
    calculate_momentum,
    calculate_realized_vol,
    classify_vol_regime,
    get_vol_adjustment,
    check_options_flow,
    check_consecutive_down_days,
    check_overextended,
    check_sideways,
)


class TestMomentum:
    def test_positive_momentum(self, bull_market_closes):
        """Bull market should have positive momentum."""
        result = calculate_momentum(bull_market_closes)
        assert result["roc_slow"] > 0
        assert result["score"] > 0.5

    def test_negative_momentum(self, bear_market_closes):
        """Bear market ending should have low/negative momentum."""
        result = calculate_momentum(bear_market_closes)
        assert result["roc_slow"] < 0

    def test_score_bounded_0_1(self, sample_qqq_closes):
        """Score is always between 0 and 1."""
        result = calculate_momentum(sample_qqq_closes)
        assert 0 <= result["score"] <= 1

    def test_insufficient_data(self):
        """Returns zeros with insufficient data."""
        result = calculate_momentum([100, 101, 102])
        assert result["score"] == 0

    def test_flat_market(self):
        """Flat prices give ~0.5 score (neutral)."""
        closes = [100.0] * 30
        result = calculate_momentum(closes)
        assert 0.4 <= result["score"] <= 0.6

    def test_roc_values_present(self, sample_qqq_closes):
        result = calculate_momentum(sample_qqq_closes)
        assert "roc_fast" in result
        assert "roc_slow" in result
        assert "raw_score" in result

    def test_custom_periods(self, sample_qqq_closes):
        result = calculate_momentum(sample_qqq_closes, roc_fast=3, roc_slow=10)
        assert isinstance(result["score"], float)


class TestRealizedVol:
    def test_normal_vol(self, sample_qqq_closes):
        """Typical QQQ vol should be in 10-30% range."""
        vol = calculate_realized_vol(sample_qqq_closes)
        assert 5 < vol < 50

    def test_zero_vol_constant_prices(self):
        """Constant prices → 0 vol."""
        closes = [100.0] * 25
        vol = calculate_realized_vol(closes)
        assert vol == 0.0

    def test_high_vol(self):
        """Highly volatile prices give high vol."""
        np.random.seed(42)
        closes = [100]
        for _ in range(25):
            closes.append(closes[-1] * (1 + np.random.normal(0, 0.05)))
        vol = calculate_realized_vol(closes)
        assert vol > 30

    def test_insufficient_data(self):
        """Returns 0 with insufficient data."""
        vol = calculate_realized_vol([100, 101])
        assert vol == 0.0

    def test_custom_window(self, sample_qqq_closes):
        vol_10 = calculate_realized_vol(sample_qqq_closes, window=10)
        vol_30 = calculate_realized_vol(sample_qqq_closes, window=30)
        # Both should be positive, may differ
        assert vol_10 > 0
        assert vol_30 > 0


class TestVolRegime:
    def test_low(self):
        assert classify_vol_regime(10) == "LOW"

    def test_normal(self):
        assert classify_vol_regime(20) == "NORMAL"

    def test_high(self):
        assert classify_vol_regime(30) == "HIGH"

    def test_extreme(self):
        assert classify_vol_regime(40) == "EXTREME"

    def test_boundary_low_normal(self):
        assert classify_vol_regime(15) == "NORMAL"  # 15 is the threshold

    def test_boundary_normal_high(self):
        assert classify_vol_regime(25) == "HIGH"

    def test_boundary_high_extreme(self):
        assert classify_vol_regime(35) == "EXTREME"

    def test_vol_adjustment_low(self):
        assert get_vol_adjustment("LOW") == 1.0

    def test_vol_adjustment_normal(self):
        assert get_vol_adjustment("NORMAL") == 1.0

    def test_vol_adjustment_high(self):
        assert get_vol_adjustment("HIGH") == 0.5

    def test_vol_adjustment_extreme(self):
        assert get_vol_adjustment("EXTREME") == 0.0


class TestOptionsFlow:
    def test_neutral_flow(self, mock_uw_flow_neutral):
        is_bearish, adj = check_options_flow(mock_uw_flow_neutral)
        assert is_bearish is False
        assert adj == 1.0

    def test_bearish_flow(self, mock_uw_flow_bearish):
        is_bearish, adj = check_options_flow(mock_uw_flow_bearish)
        assert is_bearish is True
        assert adj == 0.75


class TestConsecutiveDown:
    def test_five_down_days(self):
        closes = [100, 99, 98, 97, 96, 95]
        assert check_consecutive_down_days(closes, max_days=5) is True

    def test_four_down_days(self):
        closes = [100, 99, 98, 97, 96]
        assert check_consecutive_down_days(closes, max_days=5) is False

    def test_interrupted_streak(self):
        closes = [100, 99, 98, 99, 97, 96]
        assert check_consecutive_down_days(closes, max_days=5) is False

    def test_insufficient_data(self):
        assert check_consecutive_down_days([100, 99], max_days=5) is False


class TestOverextended:
    def test_overextended(self):
        """Price 20% above SMA50 → overextended."""
        assert check_overextended(600, 500, threshold=0.15) is True

    def test_not_overextended(self):
        """Price 5% above SMA50 → not overextended."""
        assert check_overextended(525, 500, threshold=0.15) is False

    def test_at_threshold(self):
        """Exactly at 15% → not overextended (must be strictly above)."""
        assert check_overextended(575, 500, threshold=0.15) is False

    def test_zero_sma(self):
        """Zero SMA should not crash."""
        assert check_overextended(500, 0) is False


class TestRSI:
    def test_overbought(self):
        """Monotonically rising closes → RSI near 100."""
        closes = list(np.linspace(400, 550, 30))
        from strategy.signals import calculate_rsi
        rsi = calculate_rsi(closes)
        assert rsi > 90

    def test_oversold(self):
        """Monotonically falling closes → RSI near 0."""
        closes = list(np.linspace(550, 400, 30))
        from strategy.signals import calculate_rsi
        rsi = calculate_rsi(closes)
        assert rsi < 10

    def test_neutral(self):
        """Mixed closes → RSI near 50."""
        np.random.seed(42)
        closes = [500 + np.random.normal(0, 5) for _ in range(30)]
        from strategy.signals import calculate_rsi
        rsi = calculate_rsi(closes)
        assert 30 < rsi < 70

    def test_insufficient_data(self):
        """Too few closes → returns neutral 50."""
        from strategy.signals import calculate_rsi
        assert calculate_rsi([100, 101, 102]) == 50.0

    def test_check_rsi_overbought(self):
        """Overbought check with custom threshold."""
        from strategy.signals import check_rsi_overbought
        closes = list(np.linspace(400, 550, 30))
        assert check_rsi_overbought(closes, threshold=70) is True
        # Mixed data should not trigger overbought
        np.random.seed(42)
        mixed = [500 + np.random.normal(0, 5) for _ in range(30)]
        assert check_rsi_overbought(mixed, threshold=70) is False


class TestSideways:
    def test_sideways_market(self, sideways_closes):
        """Tight range over 30 days → sideways."""
        assert check_sideways(sideways_closes, days=30, range_pct=0.05) is True

    def test_trending_market(self, bull_market_closes):
        """Trending market is not sideways."""
        assert check_sideways(bull_market_closes, days=30, range_pct=0.05) is False

    def test_insufficient_data(self):
        assert check_sideways([100, 101], days=30) is False
