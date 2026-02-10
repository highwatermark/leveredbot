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
    "bear_etf": "SQQQ",  # traded when use_sqqq_trading=True
    "underlying": "QQQ",

    # Capital Allocation
    "max_portfolio_pct": 0.30,      # Max 30% of total account equity
    "max_position_pct": 0.70,       # Max 70% of allocated capital in TQQQ
    "min_position_pct": 0.10,       # Minimum position if Risk-On

    # Regime Detection (on QQQ)
    "sma_fast": 50,
    "sma_slow": 250,
    "sma_deadzone_pct": 0.05,       # 5% band around SMA to reduce whipsaws (was 2%)

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
    "min_regime_hold_days": 10,       # Hold regime for 10 days before switching (was 2)

    # Expansion Tier 1 toggles
    "use_binary_mode": True,          # Eliminate CAUTIOUS regime (maps to BULL)
    "rsi_overbought_threshold": 70,   # RSI-14 above this blocks new buys

    # k-NN signal overlay
    "use_knn_signal": True,           # Enable k-NN prediction
    "knn_report_only": True,          # Report-only mode (no sizing impact)
    "knn_neighbors": 7,               # k in k-NN
    "knn_min_confidence": 0.55,       # Below this → FLAT signal
    "knn_disagreement_confidence": 0.60,  # k-NN SHORT + regime BULL at this confidence → gate blocks
    "knn_model_path": "data/knn_model.pkl",

    # SQQQ (inverse) trading
    "use_sqqq_trading": False,            # Enable SQQQ entries on k-NN SHORT signals
    "sqqq_min_knn_confidence": 0.60,      # Minimum k-NN confidence for SQQQ entry
    "sqqq_max_position_pct": 0.40,        # Max 40% of allocated capital in SQQQ

    # Position Manager — windowed intraday management
    "pm_enabled": True,
    "pm_stop_loss_pct": 0.08,               # 8% hard stop from entry
    "pm_trailing_stop_pct": 0.06,           # 6% trailing stop from high watermark
    "pm_gap_down_exit_pct": 0.04,           # 4% overnight gap triggers exit
    "pm_regime_emergency_pct": 0.03,        # 3% below SMA-250 triggers intraday exit
    "pm_vol_spike_exit_pct": 0.50,          # 50% vol increase triggers exit
    "pm_daily_loss_limit_pct": 0.05,        # 5% account drawdown triggers exit
    "pm_max_hold_days_losing": 15,          # Max days to hold a losing position
    "pm_min_day_trades_reserve": 2,         # Reserve 2 day trades for EOD
    "pm_profit_taking_enabled": True,
    "pm_profit_tiers": [
        {"threshold_pct": 8.0, "sell_fraction": 0.25},
        {"threshold_pct": 15.0, "sell_fraction": 0.25},
        {"threshold_pct": 25.0, "sell_fraction": 0.25},
    ],
}
