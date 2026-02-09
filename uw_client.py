"""
Unusual Whales API wrapper for TQQQ options flow sentiment.

Best-effort: if UW API is unavailable, returns neutral sentiment.
TQQQ has ~70+ flow alerts per day — we aggregate put vs call premium
to gauge institutional sentiment as a secondary overlay.
"""

import logging
from datetime import datetime, timedelta

import httpx

from config import UW_API_KEY, LEVERAGE_CONFIG

logger = logging.getLogger(__name__)

UW_BASE_URL = "https://api.unusualwhales.com"
TIMEOUT = 15  # seconds


def get_tqqq_flow(lookback_hours: int | None = None) -> dict:
    """
    Get TQQQ options flow sentiment.

    Queries UW flow alerts for TQQQ over the lookback period,
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

    neutral = {
        "put_premium": 0,
        "call_premium": 0,
        "ratio": 1.0,
        "is_bearish": False,
        "adjustment_factor": 1.0,
        "alert_count": 0,
        "error": None,
    }

    if not UW_API_KEY:
        neutral["error"] = "No UW API key configured"
        return neutral

    try:
        headers = {
            "Authorization": f"Bearer {UW_API_KEY}",
            "Accept": "application/json",
        }

        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(
                f"{UW_BASE_URL}/api/stock/{LEVERAGE_CONFIG['bull_etf']}/flow-alerts",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        alerts = data.get("data", [])
        if not alerts:
            neutral["error"] = "No flow alerts returned"
            return neutral

        # Filter to lookback window
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        put_premium = 0.0
        call_premium = 0.0
        count = 0

        for alert in alerts:
            # Parse alert timestamp
            alert_time = alert.get("created_at") or alert.get("timestamp", "")
            if alert_time:
                try:
                    ts = datetime.fromisoformat(alert_time.replace("Z", "+00:00"))
                    if ts.replace(tzinfo=None) < cutoff:
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
        logger.warning(f"UW API error (non-fatal): {e}")
        neutral["error"] = str(e)
        return neutral
