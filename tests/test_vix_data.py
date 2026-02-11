"""Tests for VIX data fetcher with local cache."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch


class TestVixCache:
    """Test cache load/save functionality."""

    def test_load_empty_cache(self, tmp_path):
        from strategy.vix_data import _load_cache, VIX_CACHE_PATH
        with patch("strategy.vix_data.VIX_CACHE_PATH", tmp_path / "vix_cache.json"):
            result = _load_cache()
            assert result == {}

    def test_load_existing_cache(self, tmp_path):
        from strategy.vix_data import _load_cache, _save_cache
        cache_path = tmp_path / "vix_cache.json"
        data = {"2025-01-01": 18.5, "2025-01-02": 19.0}
        with open(cache_path, "w") as f:
            json.dump(data, f)

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path):
            result = _load_cache()
            assert result == data

    def test_save_and_reload(self, tmp_path):
        from strategy.vix_data import _load_cache, _save_cache
        cache_path = tmp_path / "vix_cache.json"
        data = {"2025-01-01": 18.5, "2025-01-02": 19.0, "2025-01-03": 17.2}

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path):
            _save_cache(data)
            assert cache_path.exists()
            result = _load_cache()
            assert result == data

    def test_load_corrupted_cache(self, tmp_path):
        from strategy.vix_data import _load_cache
        cache_path = tmp_path / "vix_cache.json"
        cache_path.write_text("not valid json{{{")

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path):
            result = _load_cache()
            assert result == {}


class TestGetVixData:
    """Test the main get_vix_data function."""

    def test_returns_dict(self, tmp_path):
        """With mocked fetch, should return dict of date→float."""
        from strategy.vix_data import get_vix_data
        cache_path = tmp_path / "vix_cache.json"

        fake_data = {"2025-01-06": 20.5, "2025-01-07": 21.0}

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path), \
             patch("strategy.vix_data._fetch_vix", return_value=fake_data):
            result = get_vix_data()
            assert isinstance(result, dict)
            assert "2025-01-06" in result
            assert result["2025-01-06"] == 20.5

    def test_appends_to_existing_cache(self, tmp_path):
        """New data should be merged with existing cache."""
        from strategy.vix_data import get_vix_data
        cache_path = tmp_path / "vix_cache.json"

        # Pre-populate cache
        existing = {"2025-01-01": 18.0, "2025-01-02": 19.0}
        with open(cache_path, "w") as f:
            json.dump(existing, f)

        new_data = {"2025-01-03": 20.0, "2025-01-06": 21.0}

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path), \
             patch("strategy.vix_data._fetch_vix", return_value=new_data):
            result = get_vix_data()
            assert len(result) == 4
            assert result["2025-01-01"] == 18.0  # Existing preserved
            assert result["2025-01-06"] == 21.0  # New added

    def test_returns_cache_when_up_to_date(self, tmp_path):
        """If last cached date is today, should not fetch."""
        from strategy.vix_data import get_vix_data
        from datetime import datetime
        import pytz

        cache_path = tmp_path / "vix_cache.json"
        today = datetime.now(pytz.timezone("America/New_York")).date().isoformat()

        existing = {today: 22.0}
        with open(cache_path, "w") as f:
            json.dump(existing, f)

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path), \
             patch("strategy.vix_data._fetch_vix") as mock_fetch:
            result = get_vix_data()
            mock_fetch.assert_not_called()
            assert result[today] == 22.0

    def test_handles_fetch_failure(self, tmp_path):
        """If fetch fails, should return existing cache."""
        from strategy.vix_data import get_vix_data
        cache_path = tmp_path / "vix_cache.json"

        existing = {"2025-01-01": 18.0}
        with open(cache_path, "w") as f:
            json.dump(existing, f)

        with patch("strategy.vix_data.VIX_CACHE_PATH", cache_path), \
             patch("strategy.vix_data._fetch_vix", return_value={}):
            result = get_vix_data()
            assert result == existing
