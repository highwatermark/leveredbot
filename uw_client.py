"""
Unusual Whales API wrapper for options flow sentiment.

Best-effort: if UW API is unavailable, returns neutral sentiment.
Supports TQQQ (bull), SQQQ (bear), and combined flow analysis.
"""

import logging
from datetime import datetime, timedelta

import httpx
import pytz

from config import UW_API_KEY, LEVERAGE_CONFIG

logger = logging.getLogger(__name__)

UW_BASE_URL = "https://api.unusualwhales.com"
TIMEOUT = 15  # seconds


def _neutral_result(error: str | None = None) -> dict:
    """Return a neutral sentiment dict."""
    return {
        "put_premium": 0,
        "call_premium": 0,
        "ratio": 1.0,
        "is_bearish": False,
        "adjustment_factor": 1.0,
        "alert_count": 0,
        "error": error,
    }


def _get_flow_for_symbol(symbol: str, lookback_hours: int | None = None) -> dict:
    """
    Get options flow sentiment for a single symbol.

    Queries UW flow alerts over the lookback period,
    sums put and call premiums, and returns aggregated data.

    Returns:
        {
            "put_premium": float,
            "call_premium": float,
            "ratio": float,  # put/call ratio
            "is_bearish": bool,
            "adjustment_factor": float,  # 1.0 or 0.75
            "alert_count": int,
            "error": str | None,
        }
    """
    if lookback_hours is None:
        lookback_hours = LEVERAGE_CONFIG["options_flow_lookback_hours"]

    if not UW_API_KEY:
        return _neutral_result("No UW API key configured")

    try:
        headers = {
            "Authorization": f"Bearer {UW_API_KEY}",
            "Accept": "application/json",
        }

        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(
                f"{UW_BASE_URL}/api/stock/{symbol}/flow-alerts",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        alerts = data.get("data", [])
        if not alerts:
            return _neutral_result(f"No flow alerts returned for {symbol}")

        # Filter to lookback window (use UTC for consistent comparison with API timestamps)
        cutoff = datetime.now(pytz.UTC) - timedelta(hours=lookback_hours)
        put_premium = 0.0
        call_premium = 0.0
        count = 0

        for alert in alerts:
            # Parse alert timestamp
            alert_time = alert.get("created_at") or alert.get("timestamp", "")
            if alert_time:
                try:
                    ts = datetime.fromisoformat(alert_time.replace("Z", "+00:00"))
                    # Ensure ts is timezone-aware for comparison
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=pytz.UTC)
                    if ts < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            premium = float(alert.get("total_premium", 0) or 0)
            option_type = (alert.get("put_call") or alert.get("type", "")).upper()

            if option_type in ("P", "PUT"):
                put_premium += premium
            elif option_type in ("C", "CALL"):
                call_premium += premium
            count += 1

        # Calculate ratio
        if call_premium > 0:
            ratio = put_premium / call_premium
        elif put_premium > 0:
            ratio = 10.0  # All puts, no calls — very bearish
        else:
            ratio = 1.0  # No data

        bearish_threshold = LEVERAGE_CONFIG["options_flow_bearish_ratio"]
        is_bearish = ratio > bearish_threshold
        reduction = LEVERAGE_CONFIG["options_flow_reduction_pct"]
        adjustment = 1.0 - reduction if is_bearish else 1.0

        return {
            "put_premium": put_premium,
            "call_premium": call_premium,
            "ratio": round(ratio, 2),
            "is_bearish": is_bearish,
            "adjustment_factor": adjustment,
            "alert_count": count,
            "error": None,
        }

    except Exception as e:
        logger.warning(f"UW API error for {symbol} (non-fatal): {e}")
        return _neutral_result(str(e))


def get_tqqq_flow(lookback_hours: int | None = None) -> dict:
    """Get TQQQ (bull ETF) options flow sentiment."""
    return _get_flow_for_symbol(LEVERAGE_CONFIG["bull_etf"], lookback_hours)


def get_sqqq_flow(lookback_hours: int | None = None) -> dict:
    """Get SQQQ (bear ETF) options flow sentiment."""
    return _get_flow_for_symbol(LEVERAGE_CONFIG["bear_etf"], lookback_hours)


def get_combined_flow(lookback_hours: int | None = None) -> dict:
    """
    Get combined TQQQ+SQQQ flow sentiment.

    Logic:
      bullish = tqqq_call_premium + sqqq_put_premium
      bearish = tqqq_put_premium + sqqq_call_premium
      combined_ratio = bearish / bullish

    Returns backward-compatible keys so check_options_flow() works unchanged.
    Falls back to TQQQ-only if SQQQ fetch fails.
    """
    tqqq = get_tqqq_flow(lookback_hours)

    try:
        sqqq = get_sqqq_flow(lookback_hours)
    except Exception as e:
        logger.warning(f"SQQQ flow fetch failed, using TQQQ-only: {e}")
        return tqqq

    if sqqq.get("error") and tqqq.get("error"):
        # Both failed — return neutral
        return _neutral_result("Both TQQQ and SQQQ flow fetch failed")

    if sqqq.get("error"):
        # SQQQ failed — fall back to TQQQ only
        return tqqq

    # Combined sentiment:
    # Bullish signals = people buying TQQQ calls + buying SQQQ puts (betting on up)
    # Bearish signals = people buying TQQQ puts + buying SQQQ calls (betting on down)
    bullish_premium = tqqq.get("call_premium", 0) + sqqq.get("put_premium", 0)
    bearish_premium = tqqq.get("put_premium", 0) + sqqq.get("call_premium", 0)

    if bullish_premium > 0:
        combined_ratio = bearish_premium / bullish_premium
    elif bearish_premium > 0:
        combined_ratio = 10.0
    else:
        combined_ratio = 1.0

    bearish_threshold = LEVERAGE_CONFIG["options_flow_bearish_ratio"]
    is_bearish = combined_ratio > bearish_threshold
    reduction = LEVERAGE_CONFIG["options_flow_reduction_pct"]
    adjustment = 1.0 - reduction if is_bearish else 1.0

    total_alerts = tqqq.get("alert_count", 0) + sqqq.get("alert_count", 0)

    return {
        "put_premium": tqqq.get("put_premium", 0) + sqqq.get("call_premium", 0),  # bearish side
        "call_premium": tqqq.get("call_premium", 0) + sqqq.get("put_premium", 0),  # bullish side
        "ratio": round(combined_ratio, 2),
        "is_bearish": is_bearish,
        "adjustment_factor": adjustment,
        "alert_count": total_alerts,
        "error": None,
        "source": "combined",
        "tqqq_ratio": tqqq.get("ratio", 1.0),
        "sqqq_ratio": sqqq.get("ratio", 1.0),
    }
