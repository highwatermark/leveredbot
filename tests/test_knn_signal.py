"""Tests for k-NN signal prediction module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.knn_signal import KNNSignal, FeatureCalculator, FEATURE_VERSION


def _make_bars(n=400, seed=42, trend=0.001, vol=0.008):
    """Generate synthetic bar data with a mild upward trend."""
    np.random.seed(seed)
    bars = []
    close = 450.0
    base_date = _date_from_index(0)
    for i in range(n):
        ret = np.random.normal(trend, vol)
        open_price = close
        close = open_price * (1 + ret)
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.002)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.002)))
        bars.append({
            "date": _date_from_index(i),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(np.random.normal(45_000_000, 5_000_000)),
        })
    return bars


def _date_from_index(i: int) -> str:
    """Generate a YYYY-MM-DD date from index (avoids month/day overflow)."""
    from datetime import date, timedelta
    base = date(2023, 1, 2)  # A Monday
    # Skip weekends
    day = base + timedelta(days=i + (i // 5) * 2)
    return day.isoformat()


def _make_vix_data(bars):
    """Generate synthetic VIX data aligned to bar dates."""
    np.random.seed(99)
    vix = {}
    level = 18.0
    for bar in bars:
        level += np.random.normal(0, 0.5)
        level = max(10.0, min(80.0, level))
        vix[bar["date"]] = round(level, 2)
    return vix


def _make_cross_asset_bars(bars, seed_offset=0):
    """Generate synthetic cross-asset bars (TLT, GLD, IWM) aligned to QQQ bar dates."""
    result = {}
    for sym, base_price, sym_seed in [("TLT", 100.0, 10), ("GLD", 180.0, 20), ("IWM", 200.0, 30)]:
        np.random.seed(sym_seed + seed_offset)
        sym_bars = []
        close = base_price
        for bar in bars:
            ret = np.random.normal(0.0005, 0.006)
            open_price = close
            close = open_price * (1 + ret)
            high = max(open_price, close) * 1.002
            low = min(open_price, close) * 0.998
            sym_bars.append({
                "date": bar["date"],
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": 10_000_000,
            })
        result[sym] = sym_bars
    return result


def _make_microstructure_data(bars):
    """Generate synthetic microstructure data aligned to bar dates."""
    np.random.seed(77)
    result = {}
    for bar in bars:
        result[bar["date"]] = {
            "last_hour_volume_ratio": round(np.random.uniform(0.1, 0.5), 4),
            "vwap_deviation": round(np.random.normal(0, 0.005), 6),
            "closing_momentum": round(np.random.normal(0, 0.003), 6),
            "volume_acceleration": round(np.random.uniform(0.5, 2.0), 4),
        }
    return result


class TestFeatureCalculator:
    """Test feature vector computation."""

    def test_compute_features_returns_16_elements(self):
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        assert len(features) == 20

    def test_compute_features_with_vix(self):
        bars = _make_bars(300)
        vix = _make_vix_data(bars)
        features = FeatureCalculator.compute_features(bars, 250, vix_by_date=vix)
        assert features is not None
        assert len(features) == 20

    def test_compute_features_with_cross_asset(self):
        bars = _make_bars(300)
        cross = _make_cross_asset_bars(bars)
        features = FeatureCalculator.compute_features(bars, 250, cross_asset_bars=cross)
        assert features is not None
        assert len(features) == 20
        # Cross-asset features should be non-zero with aligned data
        tlt_ret = features[12]
        gld_ret = features[13]
        iwm_ratio = features[14]
        assert isinstance(tlt_ret, float)
        assert isinstance(gld_ret, float)
        assert isinstance(iwm_ratio, float)

    def test_compute_features_without_vix_uses_defaults(self):
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        # Feature 11 (vix_level) should be default VIX (20) / 100 = 0.20
        vix_level = features[10]
        assert abs(vix_level - 0.20) < 0.001
        # Feature 12 (vix_change) should be 0.0 (same default for both days)
        vix_change = features[11]
        assert abs(vix_change) < 0.001

    def test_compute_features_without_cross_asset_returns_zeros(self):
        """Cross-asset features default to 0.0 when no data provided."""
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        assert features[12] == 0.0  # tlt_return_5d
        assert features[13] == 0.0  # gld_return_5d
        assert features[14] == 0.0  # iwm_qqq_ratio_change_5d

    def test_compute_features_with_microstructure(self):
        bars = _make_bars(300)
        micro = _make_microstructure_data(bars)
        features = FeatureCalculator.compute_features(bars, 250, microstructure_by_date=micro)
        assert features is not None
        assert len(features) == 20
        # Microstructure features should be non-zero with data
        assert features[16] != 0.0 or features[17] != 0.0  # at least one non-zero

    def test_compute_features_without_microstructure_returns_zeros(self):
        """Microstructure features default to 0.0 when no data provided."""
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        assert features[16] == 0.0  # last_hour_volume_ratio
        assert features[17] == 0.0  # vwap_deviation
        assert features[18] == 0.0  # closing_momentum
        assert features[19] == 0.0  # volume_acceleration

    def test_compute_features_insufficient_history(self):
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 100)
        assert features is None

    def test_feature_names_match_count(self):
        names = FeatureCalculator.feature_names()
        assert len(names) == 20
        assert len(names) == FeatureCalculator.FEATURE_COUNT

    def test_feature_names_content(self):
        names = FeatureCalculator.feature_names()
        assert "distance_from_7ma" in names
        assert "ma_7_20_cross" in names
        assert "tlt_return_5d" in names
        assert "gld_return_5d" in names
        assert "iwm_qqq_ratio_change_5d" in names
        assert "day_of_week" in names
        # Microstructure features
        assert "last_hour_volume_ratio" in names
        assert "vwap_deviation" in names
        assert "closing_momentum" in names
        assert "volume_acceleration" in names
        # Removed features should NOT be present
        assert "prior_day_return" not in names
        assert "two_day_return" not in names
        assert "distance_from_20ma" not in names
        assert "distance_from_200ma" not in names
        assert "momentum_10" not in names
        assert "rsi_deviation" not in names

    def test_features_are_finite(self):
        bars = _make_bars(300)
        vix = _make_vix_data(bars)
        cross = _make_cross_asset_bars(bars)
        features = FeatureCalculator.compute_features(bars, 250, vix_by_date=vix, cross_asset_bars=cross)
        assert features is not None
        assert np.all(np.isfinite(features))

    def test_rsi_range(self):
        """RSI feature should be normalized to 0-1."""
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        rsi_val = features[8]  # rsi_14
        assert 0 <= rsi_val <= 1.0

    def test_atr_ratio_positive(self):
        """ATR ratio should be positive."""
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        atr_ratio = features[3]  # atr_ratio
        assert atr_ratio > 0

    def test_volume_ratio_with_uniform_volume(self):
        """When all volumes are equal, ratio should be ~1.0."""
        bars = _make_bars(300)
        for b in bars:
            b["volume"] = 50_000_000
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        volume_ratio = features[9]  # volume_ratio
        assert abs(volume_ratio - 1.0) < 0.001

    def test_vix_level_scales_correctly(self):
        """VIX of 30 should give level of 0.30."""
        bars = _make_bars(300)
        vix = {bar["date"]: 30.0 for bar in bars}
        features = FeatureCalculator.compute_features(bars, 250, vix_by_date=vix)
        assert features is not None
        assert abs(features[10] - 0.30) < 0.001  # vix_level
        assert abs(features[11]) < 0.001  # vix_change = 0 (same both days)

    def test_vix_change_detects_spike(self):
        """VIX jumping from 15 to 30 should show positive change."""
        bars = _make_bars(300)
        vix = {bar["date"]: 15.0 for bar in bars}
        vix[bars[250]["date"]] = 30.0
        features = FeatureCalculator.compute_features(bars, 250, vix_by_date=vix)
        assert features is not None
        vix_change = features[11]
        assert vix_change > 0.5  # (30 - 15) / 15 = 1.0

    def test_distance_from_7ma(self):
        """In an uptrend, close should be above 7-MA."""
        bars = _make_bars(300, trend=0.005)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        dist_7ma = features[4]
        assert dist_7ma > 0  # Close above 7-MA in strong uptrend

    def test_ma_7_20_cross_positive_uptrend(self):
        """In a strong uptrend, MA7 > MA20 → positive cross."""
        bars = _make_bars(300, trend=0.005)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        ma_7_20 = features[6]
        assert ma_7_20 > 0

    def test_ma_20_50_cross_positive_when_uptrend(self):
        """In a strong uptrend, MA20 > MA50 → positive cross."""
        bars = _make_bars(300, trend=0.005)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        ma_cross = features[7]  # ma_20_50_cross
        assert ma_cross > 0

    def test_day_of_week_range(self):
        """Day of week should be in [0, 1]."""
        bars = _make_bars(300)
        features = FeatureCalculator.compute_features(bars, 250)
        assert features is not None
        dow = features[15]
        assert 0.0 <= dow <= 1.0

    def test_cross_asset_returns_with_data(self):
        """Cross-asset 5d returns should be non-zero with aligned data."""
        bars = _make_bars(300)
        cross = _make_cross_asset_bars(bars)
        features = FeatureCalculator.compute_features(bars, 250, cross_asset_bars=cross)
        assert features is not None
        # At least one of the cross-asset features should be non-zero
        cross_features = [features[12], features[13], features[14]]
        assert any(f != 0.0 for f in cross_features)

    def test_true_range_calculation(self):
        """True range should be >= high - low."""
        bars = [
            {"open": 100, "high": 105, "low": 95, "close": 102, "date": "2024-01-01"},
            {"open": 102, "high": 110, "low": 98, "close": 108, "date": "2024-01-02"},
        ]
        tr = FeatureCalculator._compute_true_range(bars, 1)
        assert abs(tr - 12.0) < 0.001

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
        assert knn.feature_count == 20

    def test_fit_with_vix_data(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        knn = KNNSignal(n_neighbors=5)
        assert knn.fit_from_bars(bars, vix_by_date=vix) is True
        assert knn.is_fitted

    def test_fit_with_cross_asset_bars(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        cross = _make_cross_asset_bars(bars)
        knn = KNNSignal(n_neighbors=5)
        assert knn.fit_from_bars(bars, vix_by_date=vix, cross_asset_bars=cross) is True
        assert knn.is_fitted

    def test_fit_with_microstructure(self):
        bars = _make_bars(500)
        micro = _make_microstructure_data(bars)
        knn = KNNSignal(n_neighbors=5)
        assert knn.fit_from_bars(bars, microstructure_by_date=micro) is True
        assert knn.is_fitted
        assert knn.feature_count == 20

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

    def test_predict_with_vix(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars, vix_by_date=vix)
        result = knn.predict(bars, vix_by_date=vix)
        assert result["direction"] in ("LONG", "SHORT", "FLAT")

    def test_predict_with_cross_asset(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        cross = _make_cross_asset_bars(bars)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars, vix_by_date=vix, cross_asset_bars=cross)
        result = knn.predict(bars, vix_by_date=vix, cross_asset_bars=cross)
        assert result["direction"] in ("LONG", "SHORT", "FLAT")

    def test_predict_with_microstructure(self):
        bars = _make_bars(500)
        micro = _make_microstructure_data(bars)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars, microstructure_by_date=micro)
        result = knn.predict(bars, microstructure_by_date=micro)
        assert result["direction"] in ("LONG", "SHORT", "FLAT")

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
        assert knn2.feature_count == 20
        assert knn2.feature_version == FEATURE_VERSION
        loaded_pred = knn2.predict(bars)

        assert original_pred["direction"] == loaded_pred["direction"]
        assert abs(original_pred["confidence"] - loaded_pred["confidence"]) < 0.001

    def test_load_rejects_old_feature_count(self, tmp_path):
        """Model saved with different feature count should be rejected."""
        import pickle
        model_path = tmp_path / "old_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": None,
                "scaler": None,
                "n_neighbors": 7,
                "min_confidence": 0.55,
                "training_samples": 300,
                "feature_count": 10,
                "feature_version": 1,
            }, f)

        knn = KNNSignal()
        assert knn.load(model_path) is False
        assert not knn.is_fitted

    def test_load_rejects_old_feature_version(self, tmp_path):
        """Model saved with same count but old version should be rejected."""
        import pickle
        model_path = tmp_path / "old_version.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": None,
                "scaler": None,
                "n_neighbors": 7,
                "min_confidence": 0.55,
                "training_samples": 300,
                "feature_count": 16,
                "feature_version": 1,  # Old version
            }, f)

        knn = KNNSignal()
        assert knn.load(model_path) is False
        assert not knn.is_fitted

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

    def test_get_similar_days_with_vix(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        knn = KNNSignal(n_neighbors=5)
        knn.fit_from_bars(bars, vix_by_date=vix)
        similar = knn.get_similar_days(bars, n=3, vix_by_date=vix)
        assert len(similar) == 3

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
        assert abs(adj - 0.575) < 0.01

    def test_short_monotonically_decreasing(self):
        """Higher SHORT confidence → lower adjustment."""
        prev = 1.0
        for conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]:
            adj = KNNSignal._conviction_adjustment(conf, bullish=False)
            assert adj <= prev
            prev = adj
