"""
Thin wrapper around alpaca-py for the leveraged ETF strategy.

Handles retries and provides clean dict interfaces for:
- Account info (equity, cash, PDT count)
- Positions (all + TQQQ-specific)
- Market data (snapshots, historical bars, calendar)
- Order submission
"""

import time
import logging
from datetime import datetime, date, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, LEVERAGE_CONFIG

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 30


def _get_trading_client() -> TradingClient:
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def _get_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _retry(fn, description: str = "API call"):
    """Retry a function up to MAX_RETRIES times with RETRY_DELAY between attempts."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            logger.warning(f"{description} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise last_error


def get_account() -> dict:
    """Get account info: equity, cash, PDT count, day trade count."""
    def _call():
        client = _get_trading_client()
        acct = client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "daytrade_count": int(acct.daytrade_count),
            "pattern_day_trader": acct.pattern_day_trader,
        }
    return _retry(_call, "get_account")


def get_positions() -> list[dict]:
    """Get all open positions."""
    def _call():
        client = _get_trading_client()
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]
    return _retry(_call, "get_positions")


def get_tqqq_position() -> dict | None:
    """Get current TQQQ position, or None if not held."""
    positions = get_positions()
    for p in positions:
        if p["symbol"] == LEVERAGE_CONFIG["bull_etf"]:
            return p
    return None


def get_snapshot(symbols: list[str]) -> dict:
    """Get multi-symbol snapshot (latest quote, trade, daily bar)."""
    def _call():
        client = _get_data_client()
        request = StockSnapshotRequest(symbol_or_symbols=symbols)
        snapshots = client.get_stock_snapshot(request)
        result = {}
        for sym, snap in snapshots.items():
            result[sym] = {
                "latest_trade_price": float(snap.latest_trade.price) if snap.latest_trade else None,
                "latest_trade_time": snap.latest_trade.timestamp.isoformat() if snap.latest_trade else None,
                "daily_bar_close": float(snap.daily_bar.close) if snap.daily_bar else None,
                "daily_bar_open": float(snap.daily_bar.open) if snap.daily_bar else None,
                "daily_bar_high": float(snap.daily_bar.high) if snap.daily_bar else None,
                "daily_bar_low": float(snap.daily_bar.low) if snap.daily_bar else None,
                "daily_bar_volume": int(snap.daily_bar.volume) if snap.daily_bar else None,
                "prev_daily_bar_close": float(snap.previous_daily_bar.close) if snap.previous_daily_bar else None,
            }
        return result
    return _retry(_call, "get_snapshot")


def get_bars(symbol: str, start: str, end: str) -> list[dict]:
    """
    Get historical daily bars.

    Args:
        symbol: e.g. "QQQ"
        start: ISO date string e.g. "2023-01-01"
        end: ISO date string e.g. "2025-01-01"

    Returns:
        List of dicts with date, open, high, low, close, volume
    """
    def _call():
        client = _get_data_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
        )
        bars = client.get_stock_bars(request)
        result = []
        bar_set = bars[symbol] if symbol in bars else []
        for bar in bar_set:
            result.append({
                "date": bar.timestamp.date().isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })
        return result
    return _retry(_call, f"get_bars({symbol})")


def fetch_bars_for_cache(symbol: str, start: str, end: str) -> list[dict]:
    """Wrapper for get_bars compatible with cache.get_bars_with_cache."""
    return get_bars(symbol, start, end)


def get_calendar(target_date: str) -> dict | None:
    """
    Get market calendar for a specific date.

    Returns dict with open/close times, or None if market is closed.
    """
    def _call():
        client = _get_trading_client()
        cal = client.get_calendar(
            filters={"start": target_date, "end": target_date}
        )
        if not cal:
            return None
        day = cal[0]
        return {
            "date": str(day.date) if hasattr(day, "date") else target_date,
            "open": str(day.open),
            "close": str(day.close),
            "is_half_day": str(day.close) != "16:00",
        }
    return _retry(_call, "get_calendar")


def submit_market_order(symbol: str, qty: int, side: str) -> dict:
    """
    Submit a market order.

    Args:
        symbol: e.g. "TQQQ"
        qty: number of shares (positive)
        side: "buy" or "sell"

    Returns:
        Dict with order_id, status, filled_qty, filled_avg_price
    """
    def _call():
        client = _get_trading_client()
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(request)
        return {
            "order_id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": int(order.qty) if order.qty else qty,
            "side": side,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
        }
    return _retry(_call, f"submit_order({symbol} {side} {qty})")
