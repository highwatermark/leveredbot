"""
Shared fixtures for leveraged ETF tests.

Provides in-memory DB, mock data generators, and common fixtures.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import init_tables


@pytest.fixture
def db_conn():
    """In-memory SQLite database with all tables created."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_qqq_closes():
    """
    Generate realistic QQQ daily closes for testing.
    Simulates ~300 trading days with an uptrend + corrections.
    """
    np.random.seed(42)
    # Start at 400, trend up with noise
    base = 400
    prices = [base]
    for i in range(350):
        daily_return = np.random.normal(0.0004, 0.012)  # Slight upward bias
        prices.append(prices[-1] * (1 + daily_return))
    return prices


@pytest.fixture
def bull_market_closes():
    """QQQ closes in a strong bull market (price well above both SMAs)."""
    # Start low and trend strongly up over 300 days
    np.random.seed(100)
    prices = [400]
    for _ in range(300):
        prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.008)))
    return prices


@pytest.fixture
def bear_market_closes():
    """QQQ closes in a bear market (price below 250-SMA)."""
    np.random.seed(200)
    # Bull for 250 days then sharp decline
    prices = [500]
    for i in range(250):
        prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.01)))
    for i in range(60):
        prices.append(prices[-1] * (1 + np.random.normal(-0.005, 0.015)))
    return prices


@pytest.fixture
def sideways_closes():
    """QQQ closes in a sideways/range-bound market."""
    np.random.seed(300)
    # Oscillate around 500 with tiny range
    prices = [500]
    for _ in range(300):
        prices.append(500 + np.random.normal(0, 3))
    return prices


@pytest.fixture
def mock_account():
    """Mock Alpaca account data."""
    return {
        "equity": 129000.0,
        "cash": 91000.0,
        "buying_power": 91000.0,
        "daytrade_count": 0,
        "pattern_day_trader": True,
    }


@pytest.fixture
def mock_positions():
    """Mock positions with existing momentum-agent positions + TQQQ."""
    return [
        {"symbol": "BRK.B", "qty": 25, "market_value": 12700.0, "avg_entry_price": 480.0, "current_price": 508.0, "unrealized_pl": 700.0, "unrealized_plpc": 0.058},
        {"symbol": "TSM", "qty": 60, "market_value": 13000.0, "avg_entry_price": 195.0, "current_price": 216.67, "unrealized_pl": 1300.0, "unrealized_plpc": 0.111},
        {"symbol": "VMC", "qty": 40, "market_value": 12300.0, "avg_entry_price": 290.0, "current_price": 307.5, "unrealized_pl": 700.0, "unrealized_plpc": 0.060},
        {"symbol": "TQQQ", "qty": 200, "market_value": 10128.0, "avg_entry_price": 48.0, "current_price": 50.64, "unrealized_pl": 528.0, "unrealized_plpc": 0.055},
    ]


@pytest.fixture
def mock_positions_no_tqqq():
    """Mock positions without TQQQ (for new entry scenarios)."""
    return [
        {"symbol": "BRK.B", "qty": 25, "market_value": 12700.0, "avg_entry_price": 480.0, "current_price": 508.0, "unrealized_pl": 700.0, "unrealized_plpc": 0.058},
        {"symbol": "TSM", "qty": 60, "market_value": 13000.0, "avg_entry_price": 195.0, "current_price": 216.67, "unrealized_pl": 1300.0, "unrealized_plpc": 0.111},
    ]


@pytest.fixture
def mock_calendar_normal():
    """Normal trading day calendar."""
    return {"date": "2025-01-15", "open": "09:30", "close": "16:00", "is_half_day": False}


@pytest.fixture
def mock_calendar_halfday():
    """Half-day trading calendar."""
    return {"date": "2025-11-28", "open": "09:30", "close": "13:00", "is_half_day": True}


@pytest.fixture
def mock_snapshot():
    """Mock snapshot data for QQQ and TQQQ."""
    return {
        "QQQ": {
            "latest_trade_price": 520.50,
            "latest_trade_time": "2025-01-15T15:45:00-05:00",
            "daily_bar_close": 519.80,
            "daily_bar_open": 518.20,
            "daily_bar_high": 521.30,
            "daily_bar_low": 517.50,
            "daily_bar_volume": 45000000,
            "prev_daily_bar_close": 517.20,
        },
        "TQQQ": {
            "latest_trade_price": 50.64,
            "latest_trade_time": "2025-01-15T15:45:00-05:00",
            "daily_bar_close": 50.50,
            "daily_bar_open": 49.80,
            "daily_bar_high": 50.80,
            "daily_bar_low": 49.60,
            "daily_bar_volume": 120000000,
            "prev_daily_bar_close": 49.90,
        },
    }


@pytest.fixture
def mock_uw_flow_neutral():
    """Neutral options flow data."""
    return {
        "put_premium": 5000000,
        "call_premium": 6000000,
        "ratio": 0.83,
        "is_bearish": False,
        "adjustment_factor": 1.0,
        "alert_count": 45,
        "error": None,
    }


@pytest.fixture
def mock_uw_flow_bearish():
    """Bearish options flow data."""
    return {
        "put_premium": 12000000,
        "call_premium": 4000000,
        "ratio": 3.0,
        "is_bearish": True,
        "adjustment_factor": 0.75,
        "alert_count": 70,
        "error": None,
    }


# Fixture file paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)
