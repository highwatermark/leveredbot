"""
Intraday microstructure features computed from 5-minute bars.

Pure computation module — no I/O. Takes one day's intraday bars,
returns 4 features capturing informed flow clustering, VWAP deviation,
closing patterns, and volume acceleration.

Features:
  1. last_hour_volume_ratio  — volume(3-4pm) / volume(rest of day)
  2. vwap_deviation          — (close - VWAP) / VWAP
  3. closing_momentum        — last_30min_return - day_return
  4. volume_acceleration     — volume(last 2h) / volume(first 2h)

Time boundaries (ET): open 9:30, first-2h-end 11:30, last-2h-start 14:00,
last-hour 15:00, last-30min 15:30, close 16:00.

Bars have UTC timestamps — converted to ET before classifying.
Returns {feature: 0.0} defaults on any error or insufficient data (<10 bars).
"""

import logging
from datetime import datetime

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# Time boundaries in minutes from midnight ET
_OPEN = 9 * 60 + 30       # 9:30 AM
_FIRST_2H_END = 11 * 60 + 30  # 11:30 AM
_LAST_2H_START = 14 * 60  # 2:00 PM
_LAST_HOUR = 15 * 60      # 3:00 PM
_LAST_30MIN = 15 * 60 + 30  # 3:30 PM
_CLOSE = 16 * 60          # 4:00 PM

FEATURE_NAMES = [
    "last_hour_volume_ratio",
    "vwap_deviation",
    "closing_momentum",
    "volume_acceleration",
]

_DEFAULTS = {name: 0.0 for name in FEATURE_NAMES}


def _to_et_minutes(timestamp_str: str) -> int | None:
    """Convert ISO timestamp string to minutes-from-midnight in ET."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        dt_et = dt.astimezone(ET)
        return dt_et.hour * 60 + dt_et.minute
    except (ValueError, TypeError):
        return None


def compute_microstructure_features(intraday_bars: list[dict]) -> dict[str, float]:
    """
    Compute 4 microstructure features from a single day's intraday bars.

    Args:
        intraday_bars: List of bar dicts with keys: timestamp, open, high, low,
                       close, volume. Timestamps are ISO format (typically UTC).

    Returns:
        Dict with 4 feature values. Returns all 0.0 on error or insufficient data.
    """
    try:
        if not intraday_bars or len(intraday_bars) < 10:
            return dict(_DEFAULTS)

        # Classify each bar by time bucket
        first_2h_volume = 0
        last_2h_volume = 0
        last_hour_volume = 0
        rest_volume = 0
        total_volume = 0
        vwap_numerator = 0.0
        vwap_denominator = 0

        # Track first and last bars, and last-30min boundary
        all_closes = []
        last_30min_closes = []
        classified_count = 0

        for bar in intraday_bars:
            ts = bar.get("timestamp", "")
            minutes = _to_et_minutes(ts)
            if minutes is None:
                continue

            vol = bar.get("volume", 0) or 0
            close = bar.get("close", 0.0)
            total_volume += vol
            vwap_numerator += close * vol
            vwap_denominator += vol
            all_closes.append(close)
            classified_count += 1

            # Time bucket classification
            if _OPEN <= minutes < _FIRST_2H_END:
                first_2h_volume += vol
            if minutes >= _LAST_2H_START and minutes < _CLOSE:
                last_2h_volume += vol
            if minutes >= _LAST_HOUR and minutes < _CLOSE:
                last_hour_volume += vol
            else:
                rest_volume += vol
            if minutes >= _LAST_30MIN and minutes < _CLOSE:
                last_30min_closes.append(close)

        if classified_count < 10 or total_volume == 0 or not all_closes:
            return dict(_DEFAULTS)

        # Feature 1: last_hour_volume_ratio
        if rest_volume > 0:
            last_hour_volume_ratio = last_hour_volume / rest_volume
        else:
            last_hour_volume_ratio = 0.0

        # Feature 2: vwap_deviation — (close - VWAP) / VWAP
        vwap = vwap_numerator / vwap_denominator if vwap_denominator > 0 else 0.0
        day_close = all_closes[-1]
        if vwap > 0:
            vwap_deviation = (day_close - vwap) / vwap
        else:
            vwap_deviation = 0.0

        # Feature 3: closing_momentum — last_30min_return - day_return
        day_open = all_closes[0]
        day_return = (day_close - day_open) / day_open if day_open > 0 else 0.0

        if last_30min_closes and len(last_30min_closes) >= 2:
            last_30min_open = last_30min_closes[0]
            last_30min_close = last_30min_closes[-1]
            last_30min_return = (last_30min_close - last_30min_open) / last_30min_open if last_30min_open > 0 else 0.0
        else:
            last_30min_return = 0.0

        closing_momentum = last_30min_return - day_return

        # Feature 4: volume_acceleration — volume(last 2h) / volume(first 2h)
        if first_2h_volume > 0:
            volume_acceleration = last_2h_volume / first_2h_volume
        else:
            volume_acceleration = 0.0

        return {
            "last_hour_volume_ratio": last_hour_volume_ratio,
            "vwap_deviation": vwap_deviation,
            "closing_momentum": closing_momentum,
            "volume_acceleration": volume_acceleration,
        }

    except Exception as e:
        logger.warning(f"Microstructure feature computation failed: {e}")
        return dict(_DEFAULTS)
