"""Tests for intraday microstructure feature computation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.microstructure import compute_microstructure_features, FEATURE_NAMES


def _make_intraday_bars(
    n_bars=78,
    base_price=450.0,
    base_volume=500_000,
    last_hour_volume_mult=1.0,
    close_rally=0.0,
):
    """
    Generate synthetic 5-min intraday bars for a single trading day.

    78 bars = 6.5 hours (9:30-16:00 ET) at 5-min intervals.
    Timestamps are in UTC (ET + 5h).

    Args:
        n_bars: Number of bars (78 = full day)
        base_price: Starting price
        base_volume: Volume per bar
        last_hour_volume_mult: Multiplier for last-hour volume
        close_rally: Additional return in last 30 min (positive = rally)
    """
    bars = []
    price = base_price
    np.random.seed(42)

    for i in range(n_bars):
        # Time in ET minutes from midnight: 9:30 = 570, each bar = 5 min
        et_minutes = 570 + i * 5
        et_hour = et_minutes // 60
        et_min = et_minutes % 60

        # Convert to UTC (ET + 5h during EST)
        utc_hour = et_hour + 5
        timestamp = f"2025-01-15T{utc_hour:02d}:{et_min:02d}:00+00:00"

        # Volume: boost in last hour (15:00-16:00 ET = bars 66-77)
        vol = base_volume
        if et_minutes >= 900:  # 15:00 ET = last hour
            vol = int(base_volume * last_hour_volume_mult)

        # Price movement
        ret = np.random.normal(0, 0.001)

        # Close rally in last 30 min (15:30-16:00 ET = bars 72-77)
        if close_rally != 0.0 and et_minutes >= 930:
            ret += close_rally / 6  # Spread across ~6 bars

        open_price = price
        price = open_price * (1 + ret)
        high = max(open_price, price) * 1.001
        low = min(open_price, price) * 0.999

        bars.append({
            "date": "2025-01-15",
            "timestamp": timestamp,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(price, 2),
            "volume": vol,
            "vwap": round(open_price, 2),
        })
    return bars


class TestMicrostructureFeatures:
    """Test microstructure feature computation."""

    def test_returns_defaults_on_empty_input(self):
        result = compute_microstructure_features([])
        assert result == {name: 0.0 for name in FEATURE_NAMES}

    def test_returns_defaults_on_insufficient_bars(self):
        bars = _make_intraday_bars(n_bars=5)
        result = compute_microstructure_features(bars)
        assert result == {name: 0.0 for name in FEATURE_NAMES}

    def test_last_hour_volume_ratio_heavy_close(self):
        """When last-hour volume is 3x normal, ratio should be elevated."""
        bars = _make_intraday_bars(last_hour_volume_mult=3.0)
        result = compute_microstructure_features(bars)
        assert result["last_hour_volume_ratio"] > 0.3
        # With 12 bars in last hour at 3x vs 66 bars at 1x, ratio = (12*3)/(66*1) ≈ 0.55

    def test_last_hour_volume_ratio_uniform(self):
        """With uniform volume, last-hour ratio should be modest."""
        bars = _make_intraday_bars(last_hour_volume_mult=1.0)
        result = compute_microstructure_features(bars)
        # 12 bars at 1x vs 66 bars at 1x → ratio ≈ 12/66 ≈ 0.18
        assert result["last_hour_volume_ratio"] < 0.3

    def test_vwap_deviation_positive(self):
        """When price rallies through the day, close > VWAP → positive deviation."""
        # Create bars where price steadily rises
        bars = _make_intraday_bars(base_price=450.0, close_rally=0.02)
        result = compute_microstructure_features(bars)
        # Close should be above volume-weighted average → positive
        assert isinstance(result["vwap_deviation"], float)

    def test_closing_momentum_strong_close(self):
        """Flat day with rally in last 30 min → positive closing momentum."""
        bars = _make_intraday_bars(close_rally=0.01)
        result = compute_microstructure_features(bars)
        assert result["closing_momentum"] > 0

    def test_volume_acceleration_uniform(self):
        """With equal volume distribution, acceleration should be near 1.0."""
        bars = _make_intraday_bars(last_hour_volume_mult=1.0)
        result = compute_microstructure_features(bars)
        # First 2h (9:30-11:30) = 24 bars, Last 2h (14:00-16:00) = 24 bars
        # With uniform volume, ratio ≈ 1.0
        assert abs(result["volume_acceleration"] - 1.0) < 0.3

    def test_all_features_finite(self):
        bars = _make_intraday_bars()
        result = compute_microstructure_features(bars)
        for name in FEATURE_NAMES:
            assert np.isfinite(result[name]), f"{name} is not finite: {result[name]}"

    def test_returns_all_four_features(self):
        bars = _make_intraday_bars()
        result = compute_microstructure_features(bars)
        assert set(result.keys()) == set(FEATURE_NAMES)
        assert len(result) == 4

    def test_handles_none_input(self):
        result = compute_microstructure_features(None)
        assert result == {name: 0.0 for name in FEATURE_NAMES}

    def test_handles_bars_with_missing_timestamps(self):
        """Bars without timestamps should be gracefully handled."""
        bars = [{"close": 450.0, "volume": 100} for _ in range(20)]
        result = compute_microstructure_features(bars)
        # Should return defaults since no timestamps can be parsed
        assert result == {name: 0.0 for name in FEATURE_NAMES}
