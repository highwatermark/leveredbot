"""Tests for effective model arbitration between k-NN and XGBoost."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from job import _resolve_model_signal


class TestModelPolicy:
    def test_agreement_preserves_direction(self):
        result = _resolve_model_signal("LONG", 0.71, 1.0, "LONG", 0.68, 1.0)
        assert result["direction"] == "LONG"
        assert result["source"] == "agreement"
        assert result["disagreement"] is False

    def test_single_directional_model_reduces_conviction(self):
        with patch.dict("config.LEVERAGE_CONFIG", {"model_disagreement_adjustment": 0.75}, clear=False):
            result = _resolve_model_signal("SHORT", 0.72, 0.6, "FLAT", 0.5, 1.0)
            assert result["direction"] == "SHORT"
            assert result["source"] == "knn_only"
            assert result["disagreement"] is True
            assert result["adjustment"] == 0.45

    def test_opposite_directions_neutralize_signal(self):
        result = _resolve_model_signal("LONG", 0.70, 1.0, "SHORT", 0.74, 0.55)
        assert result["direction"] == "FLAT"
        assert result["disagreement"] is True
        assert result["source"].startswith("conflict_")
