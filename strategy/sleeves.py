"""Rule-based sleeve allocator for QQQ-driven TQQQ/SQQQ trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import LEVERAGE_CONFIG
from strategy.regime import get_regime_target_pct


@dataclass
class SleeveSignal:
    name: str
    side: str
    target_pct: float
    reason: str


def _sma(values: list[float], period: int) -> float:
    if len(values) < period:
        return float(values[-1]) if values else 0.0
    return float(np.mean(values[-period:]))


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    arr = np.array(closes[-(period + 1):], dtype=float)
    chg = np.diff(arr)
    gains = np.where(chg > 0, chg, 0.0)
    losses = np.where(chg < 0, -chg, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def evaluate_rule_sleeves(data: dict) -> dict:
    """Return active TQQQ/SQQQ sleeves and capped target allocations."""
    closes = list(data.get("qqq_closes", []))
    if len(closes) < 60:
        return {
            "bull_target_pct": 0.0,
            "bear_target_pct": 0.0,
            "bull_sleeves": [],
            "bear_sleeves": [],
            "overlays": ["insufficient_history"],
            "stats": {},
        }

    qqq_close = float(data.get("qqq_close", closes[-1]))
    sma_10 = _sma(closes, 10)
    sma_20 = _sma(closes, 20)
    sma_50 = float(data.get("sma_50", _sma(closes, 50)))
    sma_250 = float(data.get("sma_250", _sma(closes, 250)))
    regime = data.get("regime", "RISK_OFF")
    roc_5 = (qqq_close / closes[-6] - 1.0) if len(closes) >= 6 else 0.0
    roc_20 = (qqq_close / closes[-21] - 1.0) if len(closes) >= 21 else 0.0
    recent_high_20 = max(closes[-20:])
    recent_low_20 = min(closes[-20:])
    rsi_14 = _rsi(closes, 14)
    dist_sma20 = (qqq_close / sma_20 - 1.0) if sma_20 > 0 else 0.0
    dist_sma50 = (qqq_close / sma_50 - 1.0) if sma_50 > 0 else 0.0
    slope_20 = (sma_20 / _sma(closes[:-1], 20) - 1.0) if len(closes) > 21 and _sma(closes[:-1], 20) > 0 else 0.0
    vol_regime = data.get("vol_regime", "NORMAL")
    daily_loss_pct = float(data.get("daily_loss_pct", 0.0))
    model_direction = data.get("model_direction", "FLAT")
    model_confidence = float(data.get("model_confidence", 0.5))
    model_disagreement = bool(data.get("model_disagreement", False))

    bull_sleeves: list[SleeveSignal] = []
    bear_sleeves: list[SleeveSignal] = []
    overlays: list[str] = []

    bullish_regime = regime in ("STRONG_BULL", "BULL", "CAUTIOUS")
    bearish_regime = regime in ("RISK_OFF", "BREAKDOWN")
    regime_target = get_regime_target_pct(regime)

    if bullish_regime and qqq_close > sma_50 and sma_50 > sma_250 and roc_20 > 0:
        bull_sleeves.append(
            SleeveSignal(
                "trend_core",
                "TQQQ",
                LEVERAGE_CONFIG.get("sleeve_trend_core_pct", 0.12),
                "close>sma50>sma250 and 20d momentum positive",
            )
        )

    if bullish_regime and qqq_close >= recent_high_20 * 0.995 and roc_5 > 0.01 and slope_20 >= 0:
        bull_sleeves.append(
            SleeveSignal(
                "breakout_continuation",
                "TQQQ",
                LEVERAGE_CONFIG.get("sleeve_breakout_pct", 0.05),
                "near 20d high with positive short-term momentum",
            )
        )

    if bullish_regime and qqq_close > sma_50 and 0.0 <= dist_sma20 <= 0.015 and roc_20 > 0 and roc_5 >= -0.01:
        bull_sleeves.append(
            SleeveSignal(
                "pullback_reentry",
                "TQQQ",
                LEVERAGE_CONFIG.get("sleeve_pullback_pct", 0.04),
                "bull trend with controlled pullback toward sma20",
            )
        )

    if bullish_regime and qqq_close > sma_250 and rsi_14 <= 45 and roc_20 > 0 and roc_5 < 0:
        bull_sleeves.append(
            SleeveSignal(
                "bull_mean_reversion",
                "TQQQ",
                LEVERAGE_CONFIG.get("sleeve_mean_reversion_pct", 0.04),
                "oversold pullback inside larger uptrend",
            )
        )

    if regime == "CAUTIOUS" and qqq_close > sma_250 and roc_20 >= 0:
        bull_sleeves.append(
            SleeveSignal(
                "cautious_bull",
                "TQQQ",
                LEVERAGE_CONFIG.get("sleeve_cautious_bull_pct", 0.03),
                "above sma250 but trend strength mixed",
            )
        )

    if bearish_regime and qqq_close < sma_50 and sma_50 < sma_250 and roc_20 < -0.03:
        bear_sleeves.append(
            SleeveSignal(
                "breakdown_short",
                "SQQQ",
                LEVERAGE_CONFIG.get("sleeve_breakdown_short_pct", 0.12),
                "close<sma50<sma250 and 20d momentum negative",
            )
        )

    if bearish_regime and qqq_close < sma_50 and roc_20 < -0.02 and roc_5 > 0 and qqq_close < sma_20:
        bear_sleeves.append(
            SleeveSignal(
                "bear_rally_fade",
                "SQQQ",
                LEVERAGE_CONFIG.get("sleeve_bear_rally_pct", 0.05),
                "bearish regime with weak countertrend bounce",
            )
        )

    if bearish_regime and (daily_loss_pct >= 0.015 or vol_regime in ("HIGH", "EXTREME")) and roc_5 < -0.02:
        bear_sleeves.append(
            SleeveSignal(
                "panic_acceleration",
                "SQQQ",
                LEVERAGE_CONFIG.get("sleeve_panic_short_pct", 0.06),
                "downside acceleration with expanding stress",
            )
        )

    bull_target_pct = sum(s.target_pct for s in bull_sleeves)
    bear_target_pct = sum(s.target_pct for s in bear_sleeves)

    if bull_target_pct > 0:
        bull_target_pct = min(bull_target_pct, regime_target)
    if bear_target_pct > 0:
        bear_target_pct = min(bear_target_pct, LEVERAGE_CONFIG.get("sqqq_max_position_pct", 0.40))

    if rsi_14 >= LEVERAGE_CONFIG.get("sleeve_overbought_rsi", 80) or dist_sma20 >= 0.05:
        bull_target_pct *= LEVERAGE_CONFIG.get("sleeve_overbought_cooldown_mult", 0.70)
        overlays.append("overbought_cooldown")

    if model_direction == "FLAT" and bull_target_pct > 0:
        bull_target_pct *= LEVERAGE_CONFIG.get("sleeve_model_flat_mult", 0.85)
        overlays.append("model_flat_reduce")
    elif model_direction == "SHORT" and model_confidence >= LEVERAGE_CONFIG.get("long_model_short_block_confidence", 0.64) and bull_target_pct > 0:
        bull_target_pct *= LEVERAGE_CONFIG.get("sleeve_model_short_mult", 0.60)
        overlays.append("model_short_reduce")

    if model_disagreement and bull_target_pct > 0:
        bull_target_pct *= LEVERAGE_CONFIG.get("sleeve_model_disagreement_mult", 0.90)
        overlays.append("model_disagreement_reduce")

    if bull_target_pct < LEVERAGE_CONFIG.get("sleeve_cash_buffer_pct", 0.0025):
        bull_target_pct = 0.0
    if bear_target_pct < LEVERAGE_CONFIG.get("sleeve_cash_buffer_pct", 0.0025):
        bear_target_pct = 0.0

    return {
        "bull_target_pct": round(float(bull_target_pct), 4),
        "bear_target_pct": round(float(bear_target_pct), 4),
        "bull_sleeves": bull_sleeves,
        "bear_sleeves": bear_sleeves,
        "overlays": overlays,
        "stats": {
            "sma_10": round(sma_10, 2),
            "sma_20": round(sma_20, 2),
            "rsi_14": round(rsi_14, 2),
            "roc_5": round(roc_5, 4),
            "roc_20": round(roc_20, 4),
            "dist_sma20": round(dist_sma20, 4),
            "dist_sma50": round(dist_sma50, 4),
            "recent_high_20": round(recent_high_20, 2),
            "recent_low_20": round(recent_low_20, 2),
        },
    }
