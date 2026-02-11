"""
Signal calculations for momentum, realized volatility, and options flow.

All volatility is computed from QQQ realized returns — no VIX/VIXY.
Options flow is a secondary overlay from Unusual Whales (best-effort).
"""

import numpy as np

from config import LEVERAGE_CONFIG


def calculate_momentum(
    closes: list[float],
    roc_fast: int | None = None,
    roc_slow: int | None = None,
) -> dict:
    """
    Calculate blended momentum score from rate-of-change values.

    Blends 5-day ROC (40% weight) and 20-day ROC (60% weight) into
    a 0-1 normalized score using historical percentile ranking.
    Score of 0.3+ is the minimum to hold positions.

    Returns:
        {
            "roc_fast": float,    # 5-day rate of change
            "roc_slow": float,    # 20-day rate of change
            "raw_score": float,   # Blended raw value
            "score": float,       # Normalized 0-1 (percentile rank)
        }
    """
    if roc_fast is None:
        roc_fast = LEVERAGE_CONFIG["roc_fast"]
    if roc_slow is None:
        roc_slow = LEVERAGE_CONFIG["roc_period"]

    if len(closes) < roc_slow + 1:
        return {"roc_fast": 0, "roc_slow": 0, "raw_score": 0, "score": 0}

    current = closes[-1]
    roc_fast_val = (current - closes[-roc_fast - 1]) / closes[-roc_fast - 1]
    roc_slow_val = (current - closes[-roc_slow - 1]) / closes[-roc_slow - 1]

    # Blend: 60% slow + 40% fast
    raw = roc_slow_val * 0.6 + roc_fast_val * 0.4

    # Normalize via historical percentile: compute blended ROC for each
    # available historical point and rank current value among them.
    lookback = min(len(closes), 252)  # Use up to 1 year of history
    if lookback > roc_slow + 1:
        historical_raws = []
        for i in range(roc_slow + 1, lookback):
            idx = len(closes) - lookback + i
            c = closes[idx]
            c_fast = closes[idx - roc_fast]
            c_slow = closes[idx - roc_slow]
            rf = (c - c_fast) / c_fast if c_fast > 0 else 0
            rs = (c - c_slow) / c_slow if c_slow > 0 else 0
            historical_raws.append(rs * 0.6 + rf * 0.4)
        # Percentile rank: fraction of historical values <= current
        below = sum(1 for h in historical_raws if h <= raw)
        normalized = below / len(historical_raws)
    else:
        # Fallback to fixed bounds if insufficient history
        normalized = (raw + 0.10) / 0.20
        normalized = max(0.0, min(1.0, normalized))

    return {
        "roc_fast": round(roc_fast_val, 4),
        "roc_slow": round(roc_slow_val, 4),
        "raw_score": round(raw, 4),
        "score": round(normalized, 4),
    }


def calculate_realized_vol(closes: list[float], window: int = 20) -> float:
    """
    Calculate annualized 20-day realized volatility from QQQ closes.

    This replaces all VIX/VIXY approaches. Realized vol from QQQ returns
    is self-contained and requires no external data sources.

    Method:
        1. Compute daily log returns
        2. Rolling standard deviation over window days
        3. Annualize: stdev * sqrt(252) * 100

    Returns:
        Annualized realized volatility as percentage (e.g., 18.2)
    """
    if len(closes) < window + 1:
        return 0.0

    log_returns = np.diff(np.log(closes[-(window + 1):]))
    vol = float(np.std(log_returns, ddof=1) * np.sqrt(252) * 100)
    return round(vol, 2)


def classify_vol_regime(vol: float) -> str:
    """
    Classify realized volatility into regimes.

    LOW (<15): full allocation OK
    NORMAL (15-25): standard allocation
    HIGH (25-35): reduce allocation 50%
    EXTREME (>35): go to 100% cash
    """
    if vol < LEVERAGE_CONFIG["vol_low_threshold"]:
        return "LOW"
    elif vol < LEVERAGE_CONFIG["vol_normal_threshold"]:
        return "NORMAL"
    elif vol < LEVERAGE_CONFIG["vol_high_threshold"]:
        return "HIGH"
    else:
        return "EXTREME"


def get_vol_adjustment(vol_regime: str) -> float:
    """
    Get allocation multiplier based on volatility regime.

    LOW: 1.0 (no reduction)
    NORMAL: 1.0 (no reduction)
    HIGH: 0.5 (halve allocation)
    EXTREME: 0.0 (go to cash)
    """
    return {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.5, "EXTREME": 0.0}.get(vol_regime, 1.0)


def check_options_flow(uw_flow_data: dict) -> tuple[bool, float]:
    """
    Evaluate options flow sentiment from Unusual Whales data.

    Args:
        uw_flow_data: Dict from uw_client.get_tqqq_flow()

    Returns:
        (is_bearish, adjustment_factor) — e.g. (True, 0.75)
    """
    return uw_flow_data.get("is_bearish", False), uw_flow_data.get("adjustment_factor", 1.0)


def calculate_rsi(closes: list[float], period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index) from closing prices.

    Uses the standard Wilder smoothing method (exponential moving average of
    gains and losses).

    Args:
        closes: List of closing prices (needs at least period + 1 values)
        period: RSI lookback period (default 14)

    Returns:
        RSI value between 0 and 100. Returns 50.0 (neutral) if insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def check_rsi_overbought(
    closes: list[float],
    threshold: float | None = None,
    period: int = 14,
) -> bool:
    """
    Check if RSI is above the overbought threshold.

    Returns True if RSI >= threshold (meaning we should not buy).
    """
    if threshold is None:
        threshold = LEVERAGE_CONFIG.get("rsi_overbought_threshold", 70)
    return calculate_rsi(closes, period) >= threshold


def check_consecutive_down_days(closes: list[float], max_days: int | None = None) -> bool:
    """
    Check if QQQ has had too many consecutive down days.

    Returns True if consecutive down days >= max_days (meaning we should reduce).
    """
    if max_days is None:
        max_days = LEVERAGE_CONFIG["consecutive_down_days_max"]

    if len(closes) < max_days + 1:
        return False

    recent = closes[-(max_days + 1):]
    consecutive = 0
    for i in range(1, len(recent)):
        if recent[i] < recent[i - 1]:
            consecutive += 1
        else:
            consecutive = 0

    return consecutive >= max_days


def check_overextended(
    close: float, sma_50: float, threshold: float | None = None
) -> bool:
    """
    Check if QQQ is overextended (too far above 50-day SMA).

    Returns True if close is more than threshold% above SMA-50.
    """
    if threshold is None:
        threshold = LEVERAGE_CONFIG["mean_reversion_threshold"]

    if sma_50 <= 0:
        return False

    pct_above = (close - sma_50) / sma_50
    return pct_above > threshold


def check_sideways(
    closes: list[float],
    days: int | None = None,
    range_pct: float | None = None,
) -> bool:
    """
    Check if QQQ is in a sideways range (low volatility chop).

    Returns True if the high-low range over the period is less than range_pct.
    """
    if days is None:
        days = LEVERAGE_CONFIG["sideways_detection_days"]
    if range_pct is None:
        range_pct = LEVERAGE_CONFIG["sideways_range_pct"]

    if len(closes) < days:
        return False

    recent = closes[-days:]
    high = max(recent)
    low = min(recent)

    if low <= 0:
        return False

    actual_range = (high - low) / low
    return actual_range < range_pct
