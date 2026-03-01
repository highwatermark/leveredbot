"""Tests for XGBoost signal prediction module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from strategy.xgb_signal import XGBSignal
from strategy.knn_signal import FeatureCalculator, FEATURE_VERSION


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
            "date": _date_from_index(i),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(np.random.normal(45_000_000, 5_000_000)),
        })
    return bars


def _date_from_index(i: int) -> str:
    from datetime import date, timedelta
    base = date(2023, 1, 2)
    day = base + timedelta(days=i + (i // 5) * 2)
    return day.isoformat()


def _make_vix_data(bars):
    np.random.seed(99)
    vix = {}
    level = 18.0
    for bar in bars:
        level += np.random.normal(0, 0.5)
        level = max(10.0, min(80.0, level))
        vix[bar["date"]] = round(level, 2)
    return vix


def _make_cross_asset_bars(bars):
    result = {}
    for sym, base_price, sym_seed in [("TLT", 100.0, 10), ("GLD", 180.0, 20), ("IWM", 200.0, 30)]:
        np.random.seed(sym_seed)
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


class TestXGBSignal:
    """Test XGBoost prediction model."""

    def test_fit_returns_true_with_sufficient_data(self):
        bars = _make_bars(500)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        assert xgb.fit_from_bars(bars) is True
        assert xgb.is_fitted
        assert xgb.training_samples > 200
        assert xgb.feature_count == 20

    def test_fit_with_vix_and_cross_asset(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        cross = _make_cross_asset_bars(bars)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        assert xgb.fit_from_bars(bars, vix_by_date=vix, cross_asset_bars=cross) is True
        assert xgb.is_fitted

    def test_fit_with_microstructure(self):
        bars = _make_bars(500)
        micro = _make_microstructure_data(bars)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        assert xgb.fit_from_bars(bars, microstructure_by_date=micro) is True
        assert xgb.is_fitted
        assert xgb.feature_count == 20

    def test_fit_returns_false_with_insufficient_data(self):
        bars = _make_bars(100)
        xgb = XGBSignal(n_estimators=50)
        assert xgb.fit_from_bars(bars) is False
        assert not xgb.is_fitted

    def test_predict_returns_valid_structure(self):
        bars = _make_bars(500)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars)
        result = xgb.predict(bars)

        assert "direction" in result
        assert "confidence" in result
        assert "adjustment" in result
        assert "probabilities" in result
        assert result["direction"] in ("LONG", "SHORT", "FLAT")
        assert 0 <= result["confidence"] <= 1.0
        assert 0 < result["adjustment"] <= 1.0
        assert len(result["probabilities"]) == 2

    def test_predict_with_vix_and_cross_asset(self):
        bars = _make_bars(500)
        vix = _make_vix_data(bars)
        cross = _make_cross_asset_bars(bars)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars, vix_by_date=vix, cross_asset_bars=cross)
        result = xgb.predict(bars, vix_by_date=vix, cross_asset_bars=cross)
        assert result["direction"] in ("LONG", "SHORT", "FLAT")

    def test_predict_with_microstructure(self):
        bars = _make_bars(500)
        micro = _make_microstructure_data(bars)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars, microstructure_by_date=micro)
        result = xgb.predict(bars, microstructure_by_date=micro)
        assert result["direction"] in ("LONG", "SHORT", "FLAT")

    def test_predict_unfitted_returns_neutral(self):
        bars = _make_bars(300)
        xgb = XGBSignal()
        result = xgb.predict(bars)
        assert result["direction"] == "FLAT"
        assert result["confidence"] == 0.5
        assert result["adjustment"] == 1.0

    def test_predict_insufficient_features_returns_neutral(self):
        bars = _make_bars(500)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars)
        result = xgb.predict(bars[:50])
        assert result["direction"] == "FLAT"
        assert result["adjustment"] == 1.0

    def test_save_and_load(self, tmp_path):
        bars = _make_bars(500)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars)
        original_pred = xgb.predict(bars)

        model_path = tmp_path / "xgb_model.pkl"
        xgb.save(model_path)
        assert model_path.exists()

        xgb2 = XGBSignal()
        assert xgb2.load(model_path) is True
        assert xgb2.is_fitted
        assert xgb2.feature_count == 20
        assert xgb2.feature_version == FEATURE_VERSION
        loaded_pred = xgb2.predict(bars)

        assert original_pred["direction"] == loaded_pred["direction"]
        assert abs(original_pred["confidence"] - loaded_pred["confidence"]) < 0.001

    def test_load_rejects_old_feature_count(self, tmp_path):
        import pickle
        model_path = tmp_path / "old_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": None,
                "scaler": None,
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_confidence": 0.55,
                "training_samples": 300,
                "feature_count": 10,
                "feature_version": 1,
            }, f)

        xgb = XGBSignal()
        assert xgb.load(model_path) is False
        assert not xgb.is_fitted

    def test_load_rejects_old_feature_version(self, tmp_path):
        import pickle
        model_path = tmp_path / "old_version.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": None,
                "scaler": None,
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_confidence": 0.55,
                "training_samples": 300,
                "feature_count": 16,
                "feature_version": 1,
            }, f)

        xgb = XGBSignal()
        assert xgb.load(model_path) is False
        assert not xgb.is_fitted

    def test_load_nonexistent_returns_false(self, tmp_path):
        xgb = XGBSignal()
        assert xgb.load(tmp_path / "no_such_model.pkl") is False
        assert not xgb.is_fitted

    def test_get_feature_importance(self):
        bars = _make_bars(500)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars)
        importances = xgb.get_feature_importance()
        assert len(importances) == 20
        # Should be sorted by importance (descending)
        for i in range(len(importances) - 1):
            assert importances[i][1] >= importances[i + 1][1]
        # Each entry is (name, importance)
        names = [name for name, _ in importances]
        assert "intraday_return" in names

    def test_get_feature_importance_unfitted_returns_empty(self):
        xgb = XGBSignal()
        assert xgb.get_feature_importance() == []

    def test_deterministic_predictions(self):
        """Same input → same output."""
        bars = _make_bars(500, seed=42)
        xgb = XGBSignal(n_estimators=50, max_depth=3)
        xgb.fit_from_bars(bars)
        r1 = xgb.predict(bars)
        r2 = xgb.predict(bars)
        assert r1["direction"] == r2["direction"]
        assert r1["confidence"] == r2["confidence"]

    def test_confidence_thresholds(self):
        """Impossible min_confidence → always FLAT."""
        bars = _make_bars(500)
        xgb = XGBSignal(n_estimators=50, max_depth=3, min_confidence=1.01)
        xgb.fit_from_bars(bars)
        result = xgb.predict(bars)
        assert result["direction"] == "FLAT"
        assert result["adjustment"] == 0.75
