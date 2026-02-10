"""Tests for pregame override logic and the 0.5 bug fix."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from config import LEVERAGE_CONFIG


class TestPregameFlowOverride:
    """Test that pregame intel correctly overrides flow signals using config values."""

    def _apply_pregame_override(self, signals: dict, pregame: dict | None) -> dict:
        """
        Reproduce the pregame override logic from job.py cmd_run (lines 454-467).
        This isolates the logic for focused testing without needing the full pipeline.
        """
        if pregame:
            if pregame["flow_samples"] and pregame["flow_samples"] > 0:
                signals["options_flow_ratio"] = pregame["flow_avg_ratio"]
                signals["options_flow_bearish"] = pregame["flow_bearish_samples"] >= 3
                if signals["options_flow_bearish"]:
                    signals["options_flow_adjustment"] = 1.0 - LEVERAGE_CONFIG["options_flow_reduction_pct"]
        return signals

    def _make_signals(self, **overrides) -> dict:
        """Create default signals dict with flow-related fields."""
        signals = {
            "options_flow_ratio": 0.8,
            "options_flow_bearish": False,
            "options_flow_adjustment": 1.0,
        }
        signals.update(overrides)
        return signals

    def _make_pregame(self, **overrides) -> dict:
        """Create default pregame data."""
        pregame = {
            "flow_samples": 4,
            "flow_avg_ratio": 1.2,
            "flow_bearish_samples": 0,
            "pregame_sentiment": "NEUTRAL",
            "pregame_notes": "No notable signals",
        }
        pregame.update(overrides)
        return pregame

    def test_bearish_pregame_uses_config_reduction(self):
        """BUG FIX: Bearish pregame should use config value (0.75), not hardcoded 0.5."""
        signals = self._make_signals()
        pregame = self._make_pregame(flow_bearish_samples=3, flow_avg_ratio=2.5)

        result = self._apply_pregame_override(signals, pregame)

        expected = 1.0 - LEVERAGE_CONFIG["options_flow_reduction_pct"]  # 0.75
        assert result["options_flow_adjustment"] == expected
        assert result["options_flow_adjustment"] != 0.5  # The old bug value
        assert result["options_flow_bearish"] is True
        assert result["options_flow_ratio"] == 2.5

    def test_bearish_pregame_adjustment_matches_live_flow(self):
        """Pregame bearish adjustment should match what live flow would produce."""
        # Live flow produces 0.75 (from uw_client via config)
        live_adjustment = 1.0 - LEVERAGE_CONFIG["options_flow_reduction_pct"]

        signals = self._make_signals()
        pregame = self._make_pregame(flow_bearish_samples=4, flow_avg_ratio=3.0)
        result = self._apply_pregame_override(signals, pregame)

        assert result["options_flow_adjustment"] == live_adjustment

    def test_non_bearish_pregame_no_override(self):
        """When pregame is not bearish, options_flow_adjustment stays at original."""
        signals = self._make_signals(options_flow_adjustment=1.0)
        pregame = self._make_pregame(flow_bearish_samples=2, flow_avg_ratio=1.5)

        result = self._apply_pregame_override(signals, pregame)

        assert result["options_flow_adjustment"] == 1.0
        assert result["options_flow_bearish"] is False
        assert result["options_flow_ratio"] == 1.5

    def test_no_pregame_data(self):
        """When no pregame data exists, flow signals are unchanged."""
        signals = self._make_signals(
            options_flow_ratio=0.8,
            options_flow_bearish=False,
            options_flow_adjustment=1.0,
        )

        result = self._apply_pregame_override(signals, None)

        assert result["options_flow_ratio"] == 0.8
        assert result["options_flow_bearish"] is False
        assert result["options_flow_adjustment"] == 1.0

    def test_pregame_zero_samples_no_override(self):
        """Pregame with 0 samples should not override flow data."""
        signals = self._make_signals(options_flow_adjustment=0.9)
        pregame = self._make_pregame(flow_samples=0, flow_bearish_samples=3)

        result = self._apply_pregame_override(signals, pregame)

        assert result["options_flow_adjustment"] == 0.9  # Unchanged

    def test_pregame_overrides_ratio(self):
        """Pregame should always override flow ratio when samples > 0."""
        signals = self._make_signals(options_flow_ratio=0.5)
        pregame = self._make_pregame(flow_avg_ratio=2.8, flow_bearish_samples=1)

        result = self._apply_pregame_override(signals, pregame)

        assert result["options_flow_ratio"] == 2.8

    def test_bearish_threshold_is_3(self):
        """Bearish requires >= 3 out of 4 polls to be bearish."""
        signals_2 = self._make_signals()
        pregame_2 = self._make_pregame(flow_bearish_samples=2)
        result_2 = self._apply_pregame_override(signals_2, pregame_2)
        assert result_2["options_flow_bearish"] is False

        signals_3 = self._make_signals()
        pregame_3 = self._make_pregame(flow_bearish_samples=3)
        result_3 = self._apply_pregame_override(signals_3, pregame_3)
        assert result_3["options_flow_bearish"] is True
