"""
Regime detection state machine for the leveraged ETF strategy.

Detects 5 regimes based on QQQ price relative to its 50-day and 250-day SMAs:
- STRONG_BULL: price above both SMAs + deadzone, golden cross
- BULL: price above both SMAs, 50-SMA may be near/below 250-SMA
- CAUTIOUS: price above 250-SMA but near or below 50-SMA
- RISK_OFF: price below 250-SMA - deadzone
- BREAKDOWN: price below 250-SMA and death cross (50-SMA < 250-SMA)

Includes oscillation protection: regime must hold for min_regime_hold_days
before switching (except RISK_OFF/BREAKDOWN which take effect immediately).
"""

from config import LEVERAGE_CONFIG


# Regime -> target allocation as percentage of allocated capital
REGIME_TARGETS = {
    "STRONG_BULL": LEVERAGE_CONFIG["max_position_pct"],   # 0.70
    "BULL": 0.50,
    "CAUTIOUS": 0.25,
    "RISK_OFF": 0.0,
    "BREAKDOWN": 0.0,
}


def detect_regime(
    qqq_close: float,
    sma_50: float,
    sma_250: float,
    deadzone_pct: float | None = None,
    binary_mode: bool | None = None,
) -> str:
    """
    Determine the raw regime state based on QQQ price vs SMAs.

    Args:
        qqq_close: Current QQQ closing price
        sma_50: 50-day simple moving average of QQQ
        sma_250: 250-day simple moving average of QQQ
        deadzone_pct: Band around SMA to prevent whipsaws (default from config)
        binary_mode: If True, CAUTIOUS maps to BULL (default from config)

    Returns:
        One of: STRONG_BULL, BULL, CAUTIOUS, RISK_OFF, BREAKDOWN
        (CAUTIOUS only returned when binary_mode is False)
    """
    if deadzone_pct is None:
        deadzone_pct = LEVERAGE_CONFIG["sma_deadzone_pct"]
    if binary_mode is None:
        binary_mode = LEVERAGE_CONFIG.get("use_binary_mode", False)

    sma_50_upper = sma_50 * (1 + deadzone_pct)
    sma_250_upper = sma_250 * (1 + deadzone_pct)
    sma_250_lower = sma_250 * (1 - deadzone_pct)

    # BREAKDOWN: below 250-SMA and death cross
    if qqq_close < sma_250_lower and sma_50 < sma_250:
        return "BREAKDOWN"

    # RISK_OFF: below 250-SMA - deadzone
    if qqq_close < sma_250_lower:
        return "RISK_OFF"

    # STRONG_BULL: above both SMAs + deadzone and golden cross
    if qqq_close > sma_50_upper and qqq_close > sma_250_upper and sma_50 > sma_250:
        return "STRONG_BULL"

    # BULL: above both SMAs (at least above 250-SMA + deadzone and above 50-SMA + deadzone)
    if qqq_close > sma_50_upper and qqq_close > sma_250:
        return "BULL"

    # CAUTIOUS: above 250-SMA but near or below 50-SMA
    # In binary mode, this maps to BULL to avoid 25% partial positions
    cautious = "BULL" if binary_mode else "CAUTIOUS"

    if qqq_close > sma_250:
        return cautious

    # Between sma_250_lower and sma_250 — in the deadzone below 250-SMA
    # Treat as CAUTIOUS/BULL (don't flip to RISK_OFF in the deadzone)
    return cautious


def get_effective_regime(
    detected: str,
    previous: str | None,
    hold_days: int,
    min_hold: int | None = None,
) -> str:
    """
    Apply oscillation protection to the detected regime.

    - Cold start (no previous regime): return CAUTIOUS
    - RISK_OFF/BREAKDOWN always take effect immediately
    - Other regime changes require min_hold days before switching

    Args:
        detected: Raw detected regime from detect_regime()
        previous: Previous regime (None on cold start)
        hold_days: Days since last regime change
        min_hold: Minimum days to hold a regime (default from config)

    Returns:
        Effective regime to use for sizing
    """
    if min_hold is None:
        min_hold = LEVERAGE_CONFIG["min_regime_hold_days"]

    # Cold start — no history
    if previous is None:
        return "CAUTIOUS"

    # No change
    if detected == previous:
        return detected

    # RISK_OFF and BREAKDOWN always take effect immediately
    if detected in ("RISK_OFF", "BREAKDOWN"):
        return detected

    # Oscillation protection: if previous regime hasn't been held long enough,
    # keep the previous regime
    if hold_days < min_hold:
        return previous

    return detected


def get_regime_target_pct(regime: str) -> float:
    """Map regime to target allocation percentage of allocated capital."""
    return REGIME_TARGETS.get(regime, 0.0)
