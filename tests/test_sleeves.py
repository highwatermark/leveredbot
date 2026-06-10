"""Tests for the rule-based sleeve allocator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.sleeves import evaluate_rule_sleeves


def _base_data(**overrides):
    closes = [400 + i * 0.6 for i in range(260)]
    data = {
        "regime": "STRONG_BULL",
        "qqq_close": closes[-1],
        "sma_50": sum(closes[-50:]) / 50,
        "sma_250": sum(closes[-250:]) / 250,
        "qqq_closes": closes,
        "vol_regime": "NORMAL",
        "daily_loss_pct": 0.0,
        "model_direction": "LONG",
        "model_confidence": 0.70,
        "model_disagreement": False,
    }
    data.update(overrides)
    return data


class TestSleeves:
    def test_strong_bull_activates_trend_and_breakout_sleeves(self):
        data = _base_data()
        result = evaluate_rule_sleeves(data)
        names = [s.name for s in result["bull_sleeves"]]
        assert "trend_core" in names
        assert result["bull_target_pct"] > 0
        assert result["bear_target_pct"] == 0

    def test_overbought_applies_overlay(self):
        closes = [400 + i * 1.5 for i in range(260)]
        data = _base_data(
            qqq_closes=closes,
            qqq_close=closes[-1],
            sma_50=sum(closes[-50:]) / 50,
            sma_250=sum(closes[-250:]) / 250,
        )
        result = evaluate_rule_sleeves(data)
        assert "overbought_cooldown" in result["overlays"]

    def test_breakdown_activates_bearish_sleeves(self):
        closes = [500 - i * 0.7 for i in range(260)]
        data = _base_data(
            regime="BREAKDOWN",
            qqq_closes=closes,
            qqq_close=closes[-1],
            sma_50=sum(closes[-50:]) / 50,
            sma_250=sum(closes[-250:]) / 250,
            daily_loss_pct=0.02,
        )
        result = evaluate_rule_sleeves(data)
        bear_names = [s.name for s in result["bear_sleeves"]]
        assert "breakdown_short" in bear_names
        assert result["bear_target_pct"] > 0
        assert result["bull_target_pct"] == 0

    def test_model_short_only_reduces_bull_not_zero(self):
        data = _base_data(model_direction="SHORT", model_confidence=0.75)
        result = evaluate_rule_sleeves(data)
        assert result["bull_target_pct"] > 0
        assert "model_short_reduce" in result["overlays"]
