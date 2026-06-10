"""Tests for expectancy-based action ranking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from strategy.expectancy_signal import ExpectancySignal, determine_best_action, ACTION_LABELS
from strategy.knn_signal import FeatureCalculator


def _date_from_index(i: int) -> str:
    from datetime import date, timedelta
    base = date(2023, 1, 2)
    day = base + timedelta(days=i + (i // 5) * 2)
    return day.isoformat()


def _make_bars(n=450, seed=42, trend=0.001, vol=0.01, start=100.0):
    np.random.seed(seed)
    bars = []
    close = start
    for i in range(n):
        ret = np.random.normal(trend, vol)
        open_price = close
        close = max(1.0, open_price * (1 + ret))
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


def _make_cross_asset_bars(bars):
    result = {}
    for sym, base, seed in [("TLT", 100.0, 11), ("GLD", 180.0, 22), ("IWM", 200.0, 33)]:
        result[sym] = _make_bars(len(bars), seed=seed, trend=0.0005, vol=0.006, start=base)
        for i, bar in enumerate(result[sym]):
            bar["date"] = bars[i]["date"]
    return result


def _make_vix_data(bars):
    np.random.seed(99)
    vix = {}
    level = 18.0
    for bar in bars:
        level = max(10.0, min(60.0, level + np.random.normal(0, 0.5)))
        vix[bar["date"]] = round(level, 2)
    return vix


def _make_microstructure_data(bars):
    np.random.seed(77)
    return {
        bar["date"]: {
            "last_hour_volume_ratio": round(np.random.uniform(0.1, 0.5), 4),
            "vwap_deviation": round(np.random.normal(0, 0.005), 6),
            "closing_momentum": round(np.random.normal(0, 0.003), 6),
            "volume_acceleration": round(np.random.uniform(0.5, 2.0), 4),
        }
        for bar in bars
    }


class TestDetermineBestAction:
    def test_returns_valid_action_and_details(self):
        qqq = _make_bars(450, seed=1, trend=0.0015, start=450.0)
        tqqq = _make_bars(450, seed=2, trend=0.003, vol=0.02, start=60.0)
        sqqq = _make_bars(450, seed=3, trend=-0.002, vol=0.02, start=20.0)
        qqq_by = {b["date"]: b for b in qqq}
        tqqq_by = {b["date"]: b for b in tqqq}
        sqqq_by = {b["date"]: b for b in sqqq}
        common = sorted(set(qqq_by) & set(tqqq_by) & set(sqqq_by))

        label, details = determine_best_action(300, common, qqq_by, tqqq_by, sqqq_by)
        assert label in ACTION_LABELS
        assert set(details.keys()) == {"CASH", "TQQQ", "SQQQ"}


class TestExpectancySignal:
    def test_fit_and_predict(self):
        qqq = _make_bars(520, seed=10, trend=0.0012, start=450.0)
        tqqq = _make_bars(520, seed=20, trend=0.003, vol=0.02, start=60.0)
        sqqq = _make_bars(520, seed=30, trend=-0.002, vol=0.02, start=20.0)
        vix = _make_vix_data(qqq)
        cross = _make_cross_asset_bars(qqq)
        micro = _make_microstructure_data(qqq)

        model = ExpectancySignal(n_estimators=40, max_depth=3, learning_rate=0.1)
        assert model.fit_from_aligned_bars(qqq, tqqq, sqqq, vix_by_date=vix, cross_asset_bars=cross, microstructure_by_date=micro) is True
        assert model.is_fitted
        assert model.feature_count == FeatureCalculator.FEATURE_COUNT

        result = model.predict(qqq, vix_by_date=vix, cross_asset_bars=cross, microstructure_by_date=micro)
        assert result["action"] in ACTION_LABELS
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["probabilities"].keys()) == set(ACTION_LABELS)
