"""
Position sizing and entry gate checklist.

Contains the single `get_allocated_capital()` function for DRY capital isolation,
the 14-gate entry checklist, and the full target position calculator.
"""

from datetime import date

from config import LEVERAGE_CONFIG
from strategy.regime import get_regime_target_pct
from strategy.signals import (
    calculate_momentum,
    calculate_realized_vol,
    classify_vol_regime,
    get_vol_adjustment,
    check_options_flow,
    check_consecutive_down_days,
    check_overextended,
    check_sideways,
    check_rsi_overbought,
)


def get_allocated_capital(
    equity: float,
    positions: list[dict],
    max_portfolio_pct: float | None = None,
) -> dict:
    """
    Calculate capital allocated to the leveraged ETF strategy.

    Isolates capital by subtracting non-TQQQ position values from total equity.
    Single function, used everywhere (DRY).

    Args:
        equity: Total account equity
        positions: List of position dicts (from alpaca_client.get_positions)
        max_portfolio_pct: Max fraction of equity for this strategy

    Returns:
        {
            "total_equity": float,
            "other_positions_value": float,
            "tqqq_position_value": float,
            "allocated_capital": float,
            "cash_available": float,
        }
    """
    if max_portfolio_pct is None:
        max_portfolio_pct = LEVERAGE_CONFIG["max_portfolio_pct"]

    bull_etf = LEVERAGE_CONFIG["bull_etf"]
    bear_etf = LEVERAGE_CONFIG["bear_etf"]
    other_value = 0.0
    tqqq_value = 0.0
    sqqq_value = 0.0

    for pos in positions:
        sym = pos["symbol"]
        if sym == bull_etf:
            tqqq_value = abs(pos.get("market_value", 0))
        elif sym == bear_etf:
            sqqq_value = abs(pos.get("market_value", 0))
        else:
            other_value += abs(pos.get("market_value", 0))

    strategy_value = tqqq_value + sqqq_value
    max_allocation = equity * max_portfolio_pct
    available = min(max_allocation, equity - other_value)
    available = max(0, available)

    return {
        "total_equity": equity,
        "other_positions_value": other_value,
        "tqqq_position_value": tqqq_value,
        "sqqq_position_value": sqqq_value,
        "strategy_position_value": strategy_value,
        "allocated_capital": available,
        "cash_available": equity - other_value - strategy_value,
    }


def run_gate_checklist(data: dict) -> tuple[bool, list[str]]:
    """
    Run the 14-gate entry checklist. ALL gates must pass for a buy order.

    Args:
        data: Dict containing all required signals and state:
            - regime: str
            - qqq_close: float
            - sma_50: float
            - sma_250: float
            - momentum_score: float
            - realized_vol: float
            - vol_regime: str
            - daily_loss_pct: float (intraday QQQ loss)
            - qqq_closes: list[float] (for sideways/consecutive checks)
            - holding_days_losing: int
            - is_execution_window: bool
            - allocated_capital: float
            - day_trades_remaining: int
            - options_flow_bearish: bool
            - trading_days_fetched: int

    Returns:
        (all_passed: bool, list_of_failed_gate_names: list[str])
    """
    failed = []

    # Gate 1: Regime
    regime = data.get("regime", "RISK_OFF")
    if regime in ("RISK_OFF", "BREAKDOWN"):
        failed.append("regime")

    # Gate 2: Trend strength
    qqq_close = data.get("qqq_close", 0)
    sma_50 = data.get("sma_50", 0)
    sma_250 = data.get("sma_250", 0)
    min_strength = LEVERAGE_CONFIG["min_trend_strength"]
    # QQQ must be at least 2% above the relevant SMA
    if sma_50 > 0 and sma_250 > 0:
        pct_above_50 = (qqq_close - sma_50) / sma_50
        pct_above_250 = (qqq_close - sma_250) / sma_250
        if pct_above_50 < min_strength and pct_above_250 < min_strength:
            failed.append("trend_strength")

    # Gate 3: Momentum
    momentum = data.get("momentum_score", 0)
    if momentum < LEVERAGE_CONFIG["min_momentum_score"]:
        failed.append("momentum")

    # Gate 4: Volatility not HIGH
    vol = data.get("realized_vol", 0)
    if vol >= LEVERAGE_CONFIG["vol_high_threshold"]:
        failed.append("vol_extreme")

    # Gate 5: Daily loss
    daily_loss = data.get("daily_loss_pct", 0)
    if daily_loss >= LEVERAGE_CONFIG["max_daily_loss_pct"]:
        failed.append("daily_loss")

    # Gate 6: Not sideways
    closes = data.get("qqq_closes", [])
    if check_sideways(closes):
        failed.append("sideways")

    # Gate 7: Holding period (losing position)
    holding_losing = data.get("holding_days_losing", 0)
    if holding_losing >= LEVERAGE_CONFIG["max_holding_days_losing"]:
        failed.append("holding_days")

    # Gate 8: Not overextended
    if check_overextended(qqq_close, sma_50):
        failed.append("overextended")

    # Gate 9: Consecutive down days
    if check_consecutive_down_days(closes):
        failed.append("consecutive_down")

    # Gate 10: Execution window
    if not data.get("is_execution_window", False):
        failed.append("execution_window")

    # Gate 11: Capital available
    allocated = data.get("allocated_capital", 0)
    if allocated < LEVERAGE_CONFIG["min_trade_value"]:
        failed.append("capital")

    # Gate 12: PDT
    day_trades = data.get("day_trades_remaining", 0)
    if day_trades < LEVERAGE_CONFIG["min_day_trades_for_rebalance"]:
        failed.append("pdt")

    # Gate 13: Options flow sentiment
    if data.get("options_flow_bearish", False):
        # Flow bearishness doesn't block entry, it reduces allocation.
        # But extreme bearishness (ratio > 3x) blocks entirely.
        flow_ratio = data.get("options_flow_ratio", 1.0)
        if flow_ratio > 3.0:
            failed.append("flow_sentiment")

    # Gate 14: Data quality
    trading_days = data.get("trading_days_fetched", 0)
    if trading_days < 250:
        failed.append("data_quality")

    # Gate 15: RSI overbought
    if check_rsi_overbought(closes):
        failed.append("rsi_overbought")

    # Gate 16: k-NN disagreement (regime bullish but k-NN predicts SHORT with high confidence)
    if not LEVERAGE_CONFIG.get("knn_report_only", True):
        knn_dir = data.get("knn_direction", "FLAT")
        knn_conf = data.get("knn_confidence", 0.5)
        disagree_threshold = LEVERAGE_CONFIG.get("knn_disagreement_confidence", 0.60)
        if regime in ("STRONG_BULL", "BULL") and knn_dir == "SHORT" and knn_conf >= disagree_threshold:
            failed.append("knn_disagreement")

    return (len(failed) == 0, failed)


def calculate_target_shares(data: dict) -> dict:
    """
    Full position sizing pipeline.

    Flow:
        1. Regime target % of allocated capital
        2. Momentum scaling
        3. Volatility adjustment (halve if HIGH, zero if EXTREME)
        4. Options flow adjustment (reduce 25% if bearish)
        5. Overextension reduction
        6. Consecutive down day reduction
        7. Calculate share count

    Args:
        data: Dict with regime, allocated_capital, momentum_score, vol_regime,
              options_flow_adjustment, qqq_close, sma_50, qqq_closes,
              tqqq_price, current_shares

    Returns:
        {
            "target_shares": int,
            "target_value": float,
            "current_shares": int,
            "delta_shares": int,
            "delta_value": float,
            "action": str,  # BUY, SELL, HOLD, EXIT
            "limiting_factors": list[str],
            "target_allocation_pct": float,
        }
    """
    regime = data.get("regime", "RISK_OFF")
    allocated = data.get("allocated_capital", 0)
    momentum = data.get("momentum_score", 0)
    vol_regime = data.get("vol_regime", "NORMAL")
    flow_adj = data.get("options_flow_adjustment", 1.0)
    qqq_close = data.get("qqq_close", 0)
    sma_50 = data.get("sma_50", 0)
    closes = data.get("qqq_closes", [])
    tqqq_price = data.get("tqqq_price", 1)
    current_shares = data.get("current_shares", 0)

    limiting_factors = []

    # Step 1: Regime target
    regime_pct = get_regime_target_pct(regime)
    target_pct = regime_pct

    if regime_pct == 0:
        limiting_factors.append(f"regime={regime}")
        target_value = 0.0
        target_shares = 0
    else:
        # Step 2: Momentum scaling
        min_mom = LEVERAGE_CONFIG["min_momentum_score"]
        if momentum < min_mom:
            target_pct = LEVERAGE_CONFIG["min_position_pct"]
            limiting_factors.append(f"low_momentum={momentum:.2f}")
        elif momentum > 0.8:
            pass  # full regime allocation
        else:
            # Linear interpolation between min_position_pct and regime_pct
            min_pct = LEVERAGE_CONFIG["min_position_pct"]
            scale = (momentum - min_mom) / (0.8 - min_mom)
            target_pct = min_pct + (regime_pct - min_pct) * scale

        # Step 3: Vol adjustment
        vol_adj = get_vol_adjustment(vol_regime)
        if vol_adj < 1.0:
            target_pct *= vol_adj
            limiting_factors.append(f"vol={vol_regime}")

        # Step 4: Options flow adjustment
        if flow_adj < 1.0:
            target_pct *= flow_adj
            limiting_factors.append(f"flow_bearish_adj={flow_adj}")

        # Step 5: Overextension reduction
        if check_overextended(qqq_close, sma_50):
            target_pct *= 0.5
            limiting_factors.append("overextended")

        # Step 6: Consecutive down days
        if check_consecutive_down_days(closes):
            target_pct = min(target_pct, LEVERAGE_CONFIG["min_position_pct"])
            limiting_factors.append("consecutive_down")

        # Step 7: k-NN adjustment (if enabled and not report-only)
        knn_adj = data.get("knn_adjustment", 1.0)
        if not LEVERAGE_CONFIG.get("knn_report_only", True) and knn_adj < 1.0:
            target_pct *= knn_adj
            limiting_factors.append(f"knn_adj={knn_adj}")

        # Calculate target value and shares
        target_value = allocated * target_pct
        target_shares = max(0, int(target_value / tqqq_price)) if tqqq_price > 0 else 0

    target_value = target_shares * tqqq_price
    delta_shares = target_shares - current_shares
    delta_value = delta_shares * tqqq_price

    # Determine action
    min_trade = LEVERAGE_CONFIG["min_trade_value"]
    if regime in ("RISK_OFF", "BREAKDOWN") and current_shares > 0:
        action = "EXIT"
    elif abs(delta_value) < min_trade:
        action = "HOLD"
        delta_shares = 0
        delta_value = 0
    elif delta_shares > 0:
        action = "BUY"
    elif delta_shares < 0:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "target_shares": target_shares,
        "target_value": target_value,
        "current_shares": current_shares,
        "delta_shares": delta_shares,
        "delta_value": delta_value,
        "action": action,
        "limiting_factors": limiting_factors,
        "target_allocation_pct": round(target_pct, 4),
    }


def run_sqqq_gate_checklist(data: dict) -> tuple[bool, list[str]]:
    """
    Run the 9-gate SQQQ entry checklist. ALL gates must pass for SQQQ entry.

    Args:
        data: Dict with knn_direction, knn_confidence, vol_regime, allocated_capital,
              is_execution_window, day_trades_remaining, trading_days_fetched,
              has_tqqq_position, regime

    Returns:
        (all_passed: bool, list_of_failed_gate_names: list[str])
    """
    failed = []

    # Gate S1: k-NN must predict SHORT
    knn_dir = data.get("knn_direction", "FLAT")
    if knn_dir != "SHORT":
        failed.append("knn_not_short")

    # Gate S2: k-NN confidence above SQQQ threshold
    knn_conf = data.get("knn_confidence", 0.5)
    min_conf = LEVERAGE_CONFIG.get("sqqq_min_knn_confidence", 0.60)
    if knn_conf < min_conf:
        failed.append("knn_confidence")

    # Gate S3: Volatility not EXTREME
    vol_regime = data.get("vol_regime", "NORMAL")
    if vol_regime == "EXTREME":
        failed.append("vol_extreme")

    # Gate S4: Capital available
    allocated = data.get("allocated_capital", 0)
    if allocated < LEVERAGE_CONFIG["min_trade_value"]:
        failed.append("capital")

    # Gate S5: Execution window
    if not data.get("is_execution_window", False):
        failed.append("execution_window")

    # Gate S6: PDT
    day_trades = data.get("day_trades_remaining", 0)
    if day_trades < LEVERAGE_CONFIG["min_day_trades_for_rebalance"]:
        failed.append("pdt")

    # Gate S7: Data quality
    trading_days = data.get("trading_days_fetched", 0)
    if trading_days < 250:
        failed.append("data_quality")

    # Gate S8: Mutual exclusivity — no TQQQ position
    if data.get("has_tqqq_position", False):
        failed.append("mutual_exclusivity")

    # Gate S9: Not in RISK_OFF or BREAKDOWN regime
    regime = data.get("regime", "RISK_OFF")
    if regime in ("RISK_OFF", "BREAKDOWN"):
        failed.append("regime_risk_off")

    return (len(failed) == 0, failed)


def calculate_sqqq_target_shares(data: dict) -> dict:
    """
    SQQQ position sizing — confidence-driven, simpler than TQQQ.

    Steps:
        1. Linear scale: 0.60 conf → 50% of max, 0.80+ → 100% of max
        2. Vol adjustment (same as TQQQ)
        3. Calculate shares from target_value / sqqq_price

    Args:
        data: Dict with knn_confidence, vol_regime, allocated_capital,
              sqqq_price, current_shares

    Returns:
        Same dict shape as calculate_target_shares() + "symbol": "SQQQ"
    """
    knn_conf = data.get("knn_confidence", 0.5)
    vol_regime = data.get("vol_regime", "NORMAL")
    allocated = data.get("allocated_capital", 0)
    sqqq_price = data.get("sqqq_price", 1)
    current_shares = data.get("current_shares", 0)

    max_pct = LEVERAGE_CONFIG.get("sqqq_max_position_pct", 0.40)
    min_conf = LEVERAGE_CONFIG.get("sqqq_min_knn_confidence", 0.60)
    limiting_factors = []

    # Step 1: Confidence-driven sizing
    # Linear: min_conf → 50% of max_pct, 0.80+ → 100% of max_pct
    if knn_conf >= 0.80:
        target_pct = max_pct
    elif knn_conf >= min_conf:
        scale = (knn_conf - min_conf) / (0.80 - min_conf)
        target_pct = max_pct * (0.5 + 0.5 * scale)
        limiting_factors.append(f"knn_conf={knn_conf:.2f}")
    else:
        target_pct = 0.0
        limiting_factors.append(f"knn_conf_too_low={knn_conf:.2f}")

    # Step 2: Vol adjustment
    vol_adj = get_vol_adjustment(vol_regime)
    if vol_adj < 1.0:
        target_pct *= vol_adj
        limiting_factors.append(f"vol={vol_regime}")

    # Step 3: Calculate shares
    target_value = allocated * target_pct
    target_shares = max(0, int(target_value / sqqq_price)) if sqqq_price > 0 else 0
    target_value = target_shares * sqqq_price

    delta_shares = target_shares - current_shares
    delta_value = delta_shares * sqqq_price

    # Determine action
    min_trade = LEVERAGE_CONFIG["min_trade_value"]
    if target_shares == 0 and current_shares > 0:
        action = "EXIT"
    elif abs(delta_value) < min_trade:
        action = "HOLD"
        delta_shares = 0
        delta_value = 0
    elif delta_shares > 0:
        action = "BUY"
    elif delta_shares < 0:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "symbol": "SQQQ",
        "target_shares": target_shares,
        "target_value": target_value,
        "current_shares": current_shares,
        "delta_shares": delta_shares,
        "delta_value": delta_value,
        "action": action,
        "limiting_factors": limiting_factors,
        "target_allocation_pct": round(target_pct, 4),
    }
