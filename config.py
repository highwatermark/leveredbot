"""
Configuration for the leveraged ETF strategy.

All thresholds, regime parameters, and execution settings live here.
Loads API keys from this repo's local .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this repo only.
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

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
    "max_position_pct": 0.20,       # Strong-bull target as % of allocated capital
    "bull_position_pct": 0.15,      # Bull target as % of allocated capital
    "cautious_position_pct": 0.08,  # Partial-risk target when trend is mixed
    "min_position_pct": 0.10,       # Minimum position if Risk-On

    # Regime Detection (on QQQ)
    "sma_fast": 50,
    "sma_slow": 250,
    "sma_deadzone_pct": 0.05,       # 5% band around SMA to reduce whipsaws (was 2%)

    # Historical Data
    "history_calendar_days": 800,   # ~560 trading days, enough for 250-bar SMA + k-NN training (needs 400+)

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

    # Prediction model overlay
    "prediction_model": "both",       # "knn", "xgb", or "both"
    "use_knn_signal": True,           # Enable prediction model
    "use_xgb_signal": True,           # Best-effort secondary model (falls back neutral if unavailable)
    "model_primary": "knn",           # Primary model when arbitration is needed
    "model_disagreement_action": "reduce",  # "reduce" or "flat"
    "model_disagreement_adjustment": 0.82,  # Lighter haircut so disagreement reduces size without killing participation
    "long_require_model_support_for_new_entries": False,
    "long_model_short_block_confidence": 0.64,
    "long_model_short_bull_max_position_pct": 0.08,   # In BULL, tactical bearishness trims to a starter long, not cash
    "long_model_short_strong_bull_max_position_pct": 0.15,  # STRONG_BULL should stay invested unless the regime actually breaks
    "long_neutral_max_position_pct": 0.18,   # Max sleeve allocation when tactical model is flat
    "long_disagreement_max_position_pct": 0.22,  # Max sleeve allocation when models disagree
    "knn_report_only": False,         # Model overlay now affects sizing through the effective model
    "knn_neighbors": 7,               # k in k-NN
    "knn_min_confidence": 0.55,       # Below this → FLAT signal
    "knn_disagreement_confidence": 0.60,  # k-NN SHORT + regime BULL at this confidence → gate blocks
    "knn_model_path": "data/knn_model.pkl",
    # XGBoost hyperparameters
    "xgb_n_estimators": 200,
    "xgb_max_depth": 4,
    "xgb_learning_rate": 0.05,
    "xgb_min_confidence": 0.55,
    "xgb_model_path": "data/xgb_model.pkl",
    "expectancy_model_path": "data/expectancy_model.pkl",
    "expectancy_max_hold_days": 5,
    "expectancy_cash_buffer_pct": 0.0025,  # Require 25 bps edge over cash to take risk

    # Combined UW flow (TQQQ + SQQQ)
    "use_combined_flow": True,

    # Microstructure features (intraday-derived) — disabled, hurt accuracy in testing
    "use_microstructure": False,

    # Rule-based sleeve engine
    "use_rule_sleeves": True,
    "sleeve_overbought_rsi": 80,
    "sleeve_overbought_cooldown_mult": 0.70,
    "sleeve_model_flat_mult": 0.85,
    # 2026-06-10 live validation: k-NN SHORT is anti-predictive at every horizon
    # (38% @ 1d, 29% @ 10d; QQQ avg +3.5% in 10d after SHORT). Do not trim longs on it.
    "sleeve_model_short_mult": 1.0,
    "sleeve_model_disagreement_mult": 0.90,
    "sleeve_cash_buffer_pct": 0.0025,
    # Bull sleeves (target pct of allocated capital; capped by regime target)
    "sleeve_trend_core_pct": 0.12,
    "sleeve_breakout_pct": 0.05,
    "sleeve_pullback_pct": 0.04,
    "sleeve_mean_reversion_pct": 0.04,
    "sleeve_cautious_bull_pct": 0.03,
    # Bear sleeves (target pct of allocated capital; capped by sqqq_max_position_pct)
    "sleeve_breakdown_short_pct": 0.12,
    "sleeve_bear_rally_pct": 0.05,
    "sleeve_panic_short_pct": 0.06,

    # SQQQ (inverse) trading
    "regime_authority": "controlling",    # Structural regime controls default side selection
    # 2026-06-10: SQQQ benched. Live validation showed every SQQQ entry was driven by
    # anti-predictive k-NN SHORT calls; longer holds lose more (5d avg -4.9%, 10d -9.6%).
    # Re-enable only after a walk-forward-validated short model exists.
    "use_sqqq_trading": False,
    "inverse_allowed_regimes": ("RISK_OFF", "BREAKDOWN"),
    "allow_inverse_in_bull": False,       # Bull regime should reduce/flat, not invert
    "allow_inverse_in_strong_bull": False,
    "allow_tqqq_to_sqqq_rotation": False, # Disable direct bull->inverse flips by default
    "overbought_action": "pause_adds",    # "pause_adds", "trim", or "block_new_entries"
    "overbought_reduction_pct": 0.75,     # Applied to TQQQ target sizing when RSI is overbought
    "sqqq_min_knn_confidence": 0.55,      # Minimum k-NN confidence for SQQQ entry
    "sqqq_min_bearish_model_votes": 2,    # Require broad bearish confirmation unless trend override is active
    "sqqq_trend_override": True,          # Allow SQQQ entry on bearish trend without k-NN SHORT
    "sqqq_trend_sma50_threshold": -0.03,  # QQQ must be >3% below SMA-50 for trend override
    "sqqq_trend_roc_threshold": -0.02,    # ROC-20 must be < -2% for trend override
    "sqqq_trend_position_pct": 0.30,      # Fraction of sqqq_max_position_pct for trend entries
    "sqqq_max_position_pct": 0.40,        # Max 40% of allocated capital in SQQQ

    # Stale position protection
    "stale_position_max_days": 5,       # Force exit after N consecutive gate-fail days while holding

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
