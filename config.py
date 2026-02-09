"""
Configuration for the leveraged ETF strategy.

All thresholds, regime parameters, and execution settings live here.
Loads API keys from .env file (shared with momentum-agent).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this repo first, fall back to momentum-agent's .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path.home() / "momentum-agent" / ".env")

# API Keys
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID", "")

UW_API_KEY = os.getenv("UW_API_KEY", "")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "leveraged_etf.db"

LEVERAGE_CONFIG = {
    # Instruments
    "bull_etf": "TQQQ",
    "bear_etf": "SQQQ",  # reference only, NOT traded
    "underlying": "QQQ",

    # Capital Allocation
    "max_portfolio_pct": 0.30,      # Max 30% of total account equity
    "max_position_pct": 0.70,       # Max 70% of allocated capital in TQQQ
    "min_position_pct": 0.10,       # Minimum position if Risk-On

    # Regime Detection (on QQQ)
    "sma_fast": 50,
    "sma_slow": 250,
    "sma_deadzone_pct": 0.02,       # 2% band around SMA to prevent whipsaws

    # Historical Data
    "history_calendar_days": 400,   # ~280 trading days, enough for 250-bar SMA

    # Momentum / Rate of Change
    "roc_period": 20,
    "roc_fast": 5,

    # Volatility Guardrails — REALIZED VOLATILITY ONLY (no VIX/VIXY)
    "vol_low_threshold": 15,
    "vol_normal_threshold": 25,
    "vol_high_threshold": 35,

    "max_daily_loss_pct": 0.08,     # 8% intraday drop threshold

    # Expected Value Guardrails
    "min_trend_strength": 0.02,     # QQQ must be 2%+ above SMA to enter
    "min_momentum_score": 0.3,
    "mean_reversion_threshold": 0.15,   # >15% above 50-SMA = overextended
    "consecutive_down_days_max": 5,

    # Volatility Decay Protection
    "max_holding_days_losing": 15,
    "sideways_detection_days": 30,
    "sideways_range_pct": 0.05,

    # Options Flow Sentiment (Unusual Whales)
    "use_options_flow_gate": True,
    "options_flow_bearish_ratio": 2.0,
    "options_flow_reduction_pct": 0.25,
    "options_flow_lookback_hours": 4,

    # Execution
    "execution_time_normal": "15:50",
    "execution_time_halfday": "12:45",
    "order_type": "market",
    "min_trade_value": 100,

    # PDT Safety
    "min_day_trades_for_rebalance": 2,

    # Regime oscillation protection
    "min_regime_hold_days": 2,
}
