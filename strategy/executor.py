"""
Order execution for the leveraged ETF strategy.

Handles rebalance orders (buy/sell TQQQ), force exits, and PDT checks.
Exit trades (RISK_OFF/BREAKDOWN) always execute regardless of PDT.
"""

import logging

from config import LEVERAGE_CONFIG

logger = logging.getLogger(__name__)


def execute_rebalance(
    target_shares: int,
    current_shares: int,
    price: float,
    alpaca_client,
    is_emergency: bool = False,
    day_trades_remaining: int = 99,
    symbol: str | None = None,
) -> dict:
    """
    Execute a rebalance order.

    Calculates delta, checks PDT and min trade value, submits market order.

    Args:
        target_shares: Desired number of TQQQ shares
        current_shares: Currently held TQQQ shares
        price: Current TQQQ price (for value calculations)
        alpaca_client: Module with submit_market_order function
        is_emergency: If True, skip PDT check (regime shift to RISK_OFF)
        day_trades_remaining: Alpaca daytrade_count

    Returns:
        {
            "executed": bool,
            "action": str,
            "shares": int,
            "value": float,
            "order": dict | None,
            "reason": str,
        }
    """
    delta = target_shares - current_shares
    value = abs(delta * price)

    # Check min trade value
    if value < LEVERAGE_CONFIG["min_trade_value"] and not is_emergency:
        return {
            "executed": False,
            "action": "HOLD",
            "shares": 0,
            "value": 0,
            "order": None,
            "reason": f"Delta value ${value:.0f} below minimum ${LEVERAGE_CONFIG['min_trade_value']}",
        }

    # PDT check — skip for emergency exits
    if not is_emergency and day_trades_remaining < LEVERAGE_CONFIG["min_day_trades_for_rebalance"]:
        return {
            "executed": False,
            "action": "HOLD",
            "shares": 0,
            "value": 0,
            "order": None,
            "reason": f"PDT: only {day_trades_remaining} day trades remaining",
        }

    if delta == 0:
        return {
            "executed": False,
            "action": "HOLD",
            "shares": 0,
            "value": 0,
            "order": None,
            "reason": "Already at target",
        }

    side = "buy" if delta > 0 else "sell"
    qty = abs(delta)
    trade_symbol = symbol or LEVERAGE_CONFIG["bull_etf"]

    try:
        order = alpaca_client.submit_market_order(
            trade_symbol, qty, side
        )
        return {
            "executed": True,
            "action": side.upper(),
            "shares": qty,
            "value": value,
            "order": order,
            "reason": f"Submitted {side} {qty} shares @ ~${price:.2f}",
        }
    except Exception as e:
        logger.error(f"Order submission failed: {e}")
        return {
            "executed": False,
            "action": "ERROR",
            "shares": 0,
            "value": 0,
            "order": None,
            "reason": f"Order rejected: {e}",
        }


def force_exit(alpaca_client, symbol: str | None = None) -> dict:
    """
    Emergency sell all shares of a symbol immediately. Always executes regardless of PDT.

    Args:
        alpaca_client: Module with get_positions/submit_market_order
        symbol: Symbol to exit (default: bull_etf/TQQQ)

    Returns:
        {
            "executed": bool,
            "shares_sold": int,
            "order": dict | None,
            "reason": str,
        }
    """
    target_symbol = symbol or LEVERAGE_CONFIG["bull_etf"]

    # Find position by symbol
    position = None
    for p in alpaca_client.get_positions():
        if p["symbol"] == target_symbol:
            position = p
            break

    if not position:
        return {
            "executed": False,
            "shares_sold": 0,
            "order": None,
            "reason": f"No {target_symbol} position to exit",
        }

    qty = position["qty"]
    try:
        order = alpaca_client.submit_market_order(
            target_symbol, qty, "sell"
        )
        return {
            "executed": True,
            "shares_sold": qty,
            "order": order,
            "reason": f"Force sold {qty} shares {target_symbol}",
        }
    except Exception as e:
        logger.error(f"Force exit failed: {e}")
        return {
            "executed": False,
            "shares_sold": 0,
            "order": None,
            "reason": f"Force exit failed: {e}",
        }
