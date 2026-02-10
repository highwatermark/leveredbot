"""Tests for k-NN signal prediction module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.knn_signal import KNNSignal, FeatureCalculator


def _make_bars(n=400, seed=42, trend=0.001, vol=0.008):
    """Generate synthetic bar data with a mild upward trend."""
    np.random.seed(seed)
    bars = []
    close = 450.0
    for i in range(n):
        ret = np.random.normal(trend, vol)
        open_price = close
        close = open_price * (1 + ret)
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.002)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.002)))
        bars.append({
            "date": f"2024-{1 + i // 30:02d}-{1 + i % 30:02d}",
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": 45_000_000,
        })
    return bars


class TestFeatureCalculator:
    """Test feature vector computation."""

    def test_compute_features_returns_10_elements(self):
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        assert len(features) == 10

    def test_compute_features_insufficient_history(self):
        bars = _make_bars(300)
        # Index 100 needs 200 bars before it — insufficient
        features = FeatureCalculator.compute_features(bars, 100)
        assert features is None

    def test_feature_names_match_count(self):
        names = FeatureCalculator.feature_names()
        assert len(names) == 10

    def test_features_are_finite(self):
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        assert np.all(np.isfinite(features))

    def test_rsi_range(self):
        """RSI feature should be normalized to 0-1."""
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        rsi_val = features[8]  # rsi_14 is index 8
        assert 0 <= rsi_val <= 1.0

    def test_rsi_calculation_overbought(self):
        """Monotonically rising → RSI near 100."""
        closes = list(np.linspace(400, 550, 30))
        rsi = FeatureCalculator.calculate_rsi(closes)
        assert rsi > 90

    def test_rsi_calculation_oversold(self):
        """Monotonically falling → RSI near 0."""
        closes = list(np.linspace(550, 400, 30))
        rsi = FeatureCalculator.calculate_rsi(closes)
        assert rsi < 10


class TestKNNSignal:
    """Test k-NN prediction model."""

    def test_fit_returns_true_with_sufficient_data(self):
        bars = _make_bars(500)
        knn = KNNSignal(n_neighbors=5)
        assert knn.fit_from_bars(bars) is True
        assert knn.is_fitted
        assert knn.training_samples > 200

    def test_fit_returns_false_with_insufficient_data(self):
        bars = _make_bars(100)
        knn = KNNSignal(n_neighbors=5)
        assert knn.fit_from_bars(bars) is False
        assert not knn.is_fitted

    def test_predict_returns_valid_structure(self):
        bars = _make_bars(500)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars)
        result = knn.predict(bars)

        assert "direction" in result
        assert "confidence" in result
        assert "adjustment" in result
        assert "probabilities" in result
        assert result["direction"] in ("LONG", "SHORT", "FLAT")
        assert 0 <= result["confidence"] <= 1.0
        assert 0 < result["adjustment"] <= 1.0
        assert len(result["probabilities"]) == 2

    def test_predict_unfitted_returns_neutral(self):
        bars = _make_bars(300)
        knn = KNNSignal()
        result = knn.predict(bars)
        assert result["direction"] == "FLAT"
        assert result["confidence"] == 0.5
        assert result["adjustment"] == 1.0

    def test_predict_insufficient_features_returns_neutral(self):
        bars = _make_bars(500)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars)
        # Predict with only 50 bars (not enough for feature calc)
        result = knn.predict(bars[:50])
        assert result["direction"] == "FLAT"
        assert result["adjustment"] == 1.0

    def test_save_and_load(self, tmp_path):
        bars = _make_bars(500)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars)
        original_pred = knn.predict(bars)

        model_path = tmp_path / "knn_model.pkl"
        knn.save(model_path)
        assert model_path.exists()

        knn2 = KNNSignal()
        assert knn2.load(model_path) is True
        assert knn2.is_fitted
        loaded_pred = knn2.predict(bars)

        assert original_pred["direction"] == loaded_pred["direction"]
        assert abs(original_pred["confidence"] - loaded_pred["confidence"]) < 0.001

    def test_load_nonexistent_returns_false(self, tmp_path):
        knn = KNNSignal()
        assert knn.load(tmp_path / "no_such_model.pkl") is False
        assert not knn.is_fitted

    def test_get_similar_days(self):
        bars = _make_bars(500)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars)
        similar = knn.get_similar_days(bars, n=3)
        assert len(similar) == 3
        for s in similar:
            assert "date" in s
            assert "distance" in s
            assert "next_day_return" in s
            assert s["distance"] >= 0

    def test_get_similar_days_unfitted_returns_empty(self):
        bars = _make_bars(300)
        knn = KNNSignal()
        assert knn.get_similar_days(bars) == []

    def test_confidence_thresholds(self):
        """Impossible min_confidence → always FLAT with 0.75 adjustment."""
        bars = _make_bars(500)
        knn = KNNSignal(n_neighbors=5, min_confidence=1.01)
        knn.fit_from_bars(bars)
        result = knn.predict(bars)
        # With min_confidence > 1.0, everything must be FLAT
        assert result["direction"] == "FLAT"
        assert result["adjustment"] == 0.75

    def test_deterministic_predictions(self):
        """Same input → same output."""
        bars = _make_bars(500, seed=42)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars)
        r1 = knn.predict(bars)
        r2 = knn.predict(bars)
        assert r1["direction"] == r2["direction"]
        assert r1["confidence"] == r2["confidence"]


class TestConvictionScoring:
    """Test continuous conviction-based adjustment."""

    def test_long_always_returns_1(self):
        assert KNNSignal._conviction_adjustment(0.55, bullish=True) == 1.0
        assert KNNSignal._conviction_adjustment(0.90, bullish=True) == 1.0

    def test_short_low_confidence(self):
        """0.55-0.65 SHORT → 0.75 (25% reduction)."""
        assert KNNSignal._conviction_adjustment(0.55, bullish=False) == 0.75
        assert KNNSignal._conviction_adjustment(0.60, bullish=False) == 0.75
        assert KNNSignal._conviction_adjustment(0.65, bullish=False) == 0.75

    def test_short_high_confidence(self):
        """0.80+ SHORT → 0.40 (60% reduction)."""
        assert KNNSignal._conviction_adjustment(0.80, bullish=False) == 0.40
        assert KNNSignal._conviction_adjustment(0.95, bullish=False) == 0.40

    def test_short_mid_confidence_interpolates(self):
        """0.65-0.80 SHORT → linear from 0.75 to 0.40."""
        adj = KNNSignal._conviction_adjustment(0.725, bullish=False)
        # midpoint: 0.75 - 0.5 * 0.35 = 0.575
        assert abs(adj - 0.575) < 0.01

    def test_short_monotonically_decreasing(self):
        """Higher SHORT confidence → lower adjustment."""
        prev = 1.0
        for conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]:
            adj = KNNSignal._conviction_adjustment(conf, bullish=False)
            assert adj <= prev
            prev = adj
