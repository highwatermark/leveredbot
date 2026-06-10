"""
Cash-yield sweep: idle allocated capital parked in a T-bill ETF (SGOV).

The strategy's average TQQQ deployment is well under its allocation; the
remainder earned nothing. The sweep buys SGOV with idle allocated capital at
the main run and sells it whenever the strategy needs the room. SGOV is
treated as strategy capital (cash-equivalent), never as an "other" position.

PDT safety: the sweep trades at most once per direction per day at the main
run, and SGOV positions are held overnight, so no same-day round trips occur.
"""

import logging

from config import LEVERAGE_CONFIG

logger = logging.getLogger(__name__)


def calculate_sweep_target(
    allocated_capital: float,
    strategy_position_value: float,
    sweep_position_value: float,
    sweep_price: float,
) -> dict:
    """
    Compute the SGOV target from idle allocated capital.

    idle = allocated_capital - strategy positions - cash buffer
    (buffer absorbs fees/slippage so TQQQ buys never bounce on insufficient cash)

    Returns:
        {"action": "BUY"|"SELL"|"HOLD", "target_shares": int,
         "delta_shares": int, "delta_value": float, "reason": str}
    """
    hold = {"action": "HOLD", "target_shares": 0, "delta_shares": 0,
            "delta_value": 0.0, "reason": ""}

    if not LEVERAGE_CONFIG.get("use_cash_sweep", False):
        return {**hold, "reason": "sweep disabled"}
    if sweep_price <= 0:
        return {**hold, "reason": "no sweep price"}

    buffer = allocated_capital * LEVERAGE_CONFIG.get("sweep_buffer_pct", 0.02)
    idle = max(0.0, allocated_capital - strategy_position_value - buffer)

    target_shares = int(idle / sweep_price)
    current_shares = int(sweep_position_value / sweep_price) if sweep_price > 0 else 0
    delta_shares = target_shares - current_shares
    delta_value = delta_shares * sweep_price

    min_trade = LEVERAGE_CONFIG.get("sweep_min_trade_value", 250)
    if abs(delta_value) < min_trade:
        return {**hold, "target_shares": target_shares,
                "reason": f"sweep delta ${abs(delta_value):.0f} below ${min_trade}"}

    action = "BUY" if delta_shares > 0 else "SELL"
    return {
        "action": action,
        "target_shares": target_shares,
        "delta_shares": delta_shares,
        "delta_value": delta_value,
        "reason": f"sweep {action} {abs(delta_shares)} shares (idle ${idle:,.0f})",
    }


def cash_needed_for_buy(buy_value: float, cash_available: float) -> float:
    """
    Raw cash shortfall for a pending strategy buy, padded 1% for price drift
    between the SGOV sell and the TQQQ buy. 0 when cash already covers it.
    """
    shortfall = buy_value - max(0.0, cash_available)
    if shortfall <= 0:
        return 0.0
    return shortfall * 1.01


def execute_sweep(
    alpaca_client,
    allocated_capital: float,
    strategy_position_value: float,
    skip_buy: bool = False,
) -> dict:
    """
    Bring the SGOV position to target. Called at the end of the main run.

    Args:
        skip_buy: True when SGOV was already sold this run to fund a TQQQ buy —
                  never sell and rebuy the sweep in the same run.
    """
    sweep_etf = LEVERAGE_CONFIG.get("sweep_etf", "SGOV")
    noop = {"executed": False, "action": "HOLD", "shares": 0, "reason": ""}

    if not LEVERAGE_CONFIG.get("use_cash_sweep", False):
        return {**noop, "reason": "sweep disabled"}

    sweep_value = 0.0
    for p in alpaca_client.get_positions():
        if p["symbol"] == sweep_etf:
            sweep_value = abs(p.get("market_value", 0))
            break

    snapshots = alpaca_client.get_snapshot([sweep_etf]) or {}
    snap = snapshots.get(sweep_etf) or {}
    sweep_price = snap.get("latest_trade_price") or snap.get("daily_bar_close") or 0

    target = calculate_sweep_target(
        allocated_capital, strategy_position_value, sweep_value, sweep_price
    )
    if target["action"] == "HOLD":
        return {**noop, "reason": target["reason"]}
    if target["action"] == "BUY" and skip_buy:
        return {**noop, "reason": "sweep buy skipped (sold sweep earlier this run)"}

    side = "buy" if target["action"] == "BUY" else "sell"
    qty = abs(target["delta_shares"])
    try:
        order = alpaca_client.submit_market_order(sweep_etf, qty, side)
        logger.info(f"Cash sweep: {side} {qty} {sweep_etf} @ ~${sweep_price:.2f}")
        return {"executed": True, "action": target["action"], "shares": qty,
                "order": order, "reason": target["reason"]}
    except Exception as e:
        logger.error(f"Cash sweep order failed: {e}")
        return {**noop, "action": "ERROR", "reason": f"sweep order rejected: {e}"}


def free_cash_for_buy(alpaca_client, buy_value: float, cash_available: float) -> dict:
    """
    Sell enough SGOV before a strategy buy so the market order can't bounce.
    Returns {"sold": bool, "value": float, "reason": str}.
    """
    sweep_etf = LEVERAGE_CONFIG.get("sweep_etf", "SGOV")
    if not LEVERAGE_CONFIG.get("use_cash_sweep", False):
        return {"sold": False, "value": 0.0, "reason": "sweep disabled"}

    needed = cash_needed_for_buy(buy_value, cash_available)
    if needed <= 0:
        return {"sold": False, "value": 0.0, "reason": "cash sufficient"}

    sweep_qty = 0
    sweep_price = 0.0
    for p in alpaca_client.get_positions():
        if p["symbol"] == sweep_etf:
            sweep_qty = int(p["qty"])
            mv = abs(p.get("market_value", 0))
            sweep_price = mv / sweep_qty if sweep_qty else 0.0
            break
    if sweep_qty <= 0 or sweep_price <= 0:
        return {"sold": False, "value": 0.0, "reason": f"no {sweep_etf} to liquidate"}

    qty = min(sweep_qty, int(needed / sweep_price) + 1)
    try:
        alpaca_client.submit_market_order(sweep_etf, qty, "sell")
        logger.info(f"Cash sweep: pre-sold {qty} {sweep_etf} to fund ${buy_value:,.0f} buy")
        return {"sold": True, "value": qty * sweep_price,
                "reason": f"pre-sold {qty} {sweep_etf}"}
    except Exception as e:
        logger.error(f"Sweep pre-sell failed: {e}")
        return {"sold": False, "value": 0.0, "reason": f"pre-sell rejected: {e}"}
