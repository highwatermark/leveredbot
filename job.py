#!/usr/bin/env python3
"""
CLI entry point for the leveraged ETF strategy.

Usage:
    python job.py run               # Execute daily strategy
    python job.py run --halfday-check  # Only run if today is a half day
    python job.py pregame           # Pre-execution intel gathering (3:30 PM)
    python job.py status            # Show current state without trading
    python job.py backtest          # Run historical backtest
    python job.py force_exit        # Emergency exit all TQQQ
"""

import sys
import logging
import json
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from config import LEVERAGE_CONFIG, DATA_DIR, LOG_DIR

ET = pytz.timezone("America/New_York")

# Configure logging — write to file directly; cron captures stdout separately
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "leveraged_etf.log"),
    ],
)
logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")


def _fetch_all_data(use_cache: bool = True) -> dict:
    """Fetch all market data needed for the strategy."""
    import alpaca_client
    from db.cache import get_bars_with_cache
    from db.models import get_connection

    logger.info("Fetching market data...")

    # Account and positions
    account = alpaca_client.get_account()
    positions = alpaca_client.get_positions()
    tqqq_pos = None
    for p in positions:
        if p["symbol"] == LEVERAGE_CONFIG["bull_etf"]:
            tqqq_pos = p

    # Calendar (half-day detection)
    today_str = datetime.now(ET).date().isoformat()
    calendar = alpaca_client.get_calendar(today_str)
    is_half_day = calendar.get("is_half_day", False) if calendar else False

    # Snapshots
    try:
        snapshots = alpaca_client.get_snapshot(
            [LEVERAGE_CONFIG["underlying"], LEVERAGE_CONFIG["bull_etf"]]
        )
    except Exception as e:
        logger.warning(f"Snapshot fetch failed: {e}")
        snapshots = {}

    # Historical bars
    cal_days = LEVERAGE_CONFIG["history_calendar_days"]
    if use_cache:
        conn = get_connection()
        qqq_bars = get_bars_with_cache(
            LEVERAGE_CONFIG["underlying"], cal_days,
            alpaca_client.fetch_bars_for_cache, conn
        )
        conn.close()
    else:
        start = (datetime.now(ET).date() - timedelta(days=cal_days)).isoformat()
        end = datetime.now(ET).date().isoformat()
        qqq_bars = alpaca_client.get_bars(LEVERAGE_CONFIG["underlying"], start, end)

    qqq_closes = [b["close"] for b in qqq_bars]

    # TQQQ price from snapshot or position
    tqqq_price = None
    if LEVERAGE_CONFIG["bull_etf"] in snapshots:
        tqqq_price = snapshots[LEVERAGE_CONFIG["bull_etf"]].get("latest_trade_price")
    if tqqq_price is None and tqqq_pos:
        tqqq_price = tqqq_pos["current_price"]

    # QQQ current price from snapshot or last bar
    qqq_current = None
    if LEVERAGE_CONFIG["underlying"] in snapshots:
        qqq_current = snapshots[LEVERAGE_CONFIG["underlying"]].get("latest_trade_price")
    if qqq_current is None and qqq_closes:
        qqq_current = qqq_closes[-1]

    # QQQ intraday loss
    daily_loss_pct = 0.0
    if LEVERAGE_CONFIG["underlying"] in snapshots:
        snap = snapshots[LEVERAGE_CONFIG["underlying"]]
        prev = snap.get("prev_daily_bar_close")
        curr = snap.get("latest_trade_price")
        if prev and curr and prev > 0:
            change = (curr - prev) / prev
            if change < 0:
                daily_loss_pct = abs(change)

    return {
        "account": account,
        "positions": positions,
        "tqqq_position": tqqq_pos,
        "calendar": calendar,
        "is_half_day": is_half_day,
        "snapshots": snapshots,
        "qqq_bars": qqq_bars,
        "qqq_closes": qqq_closes,
        "qqq_current": qqq_current,
        "tqqq_price": tqqq_price,
        "daily_loss_pct": daily_loss_pct,
        "trading_days_fetched": len(qqq_bars),
    }


def _compute_signals(data: dict) -> dict:
    """Compute all strategy signals from market data."""
    from strategy.signals import (
        calculate_momentum, calculate_realized_vol,
        classify_vol_regime, get_vol_adjustment, check_options_flow,
    )
    from strategy.regime import detect_regime, get_effective_regime
    from strategy.sizing import get_allocated_capital
    from db.models import get_last_regime, get_regime_duration_days, get_consecutive_losing_days
    import uw_client

    closes = data["qqq_closes"]
    qqq_close = data["qqq_current"] or (closes[-1] if closes else 0)

    # SMAs
    sma_50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else 0
    sma_250 = float(np.mean(closes[-250:])) if len(closes) >= 250 else 0

    # Momentum
    mom = calculate_momentum(closes)

    # Realized vol
    vol = calculate_realized_vol(closes)
    vol_regime = classify_vol_regime(vol)
    vol_adj = get_vol_adjustment(vol_regime)

    # Options flow
    flow = uw_client.get_tqqq_flow()
    is_bearish, flow_adj = check_options_flow(flow)

    # Regime
    raw_regime = detect_regime(qqq_close, sma_50, sma_250)
    prev_regime = get_last_regime()
    hold_days = get_regime_duration_days()
    effective_regime = get_effective_regime(raw_regime, prev_regime, hold_days)

    # Capital
    account = data["account"]
    positions = data["positions"]
    capital = get_allocated_capital(account["equity"], positions)

    # Percentages above SMAs
    pct_above_sma50 = ((qqq_close / sma_50) - 1) * 100 if sma_50 > 0 else 0
    pct_above_sma250 = ((qqq_close / sma_250) - 1) * 100 if sma_250 > 0 else 0

    current_shares = data["tqqq_position"]["qty"] if data["tqqq_position"] else 0

    return {
        "qqq_close": qqq_close,
        "sma_50": round(sma_50, 2),
        "sma_250": round(sma_250, 2),
        "pct_above_sma50": round(pct_above_sma50, 2),
        "pct_above_sma250": round(pct_above_sma250, 2),
        "momentum": mom,
        "momentum_score": mom["score"],
        "realized_vol": vol,
        "vol_regime": vol_regime,
        "vol_adjustment": vol_adj,
        "flow": flow,
        "options_flow_bearish": is_bearish,
        "options_flow_adjustment": flow_adj,
        "options_flow_ratio": flow.get("ratio", 1.0),
        "raw_regime": raw_regime,
        "effective_regime": effective_regime,
        "previous_regime": prev_regime,
        "regime_hold_days": hold_days,
        "regime_changed": effective_regime != prev_regime if prev_regime else False,
        "capital": capital,
        "allocated_capital": capital["allocated_capital"],
        "current_shares": current_shares,
        "tqqq_price": data["tqqq_price"] or 0,
        "daily_loss_pct": data["daily_loss_pct"],
        "qqq_closes": closes,
        "trading_days_fetched": data["trading_days_fetched"],
        "day_trades_remaining": account["daytrade_count"],
        "account_equity": account["equity"],
        "cash_balance": account["cash"],
        "consecutive_losing_days": get_consecutive_losing_days(),
    }


def cmd_pregame():
    """
    Pre-execution intelligence gathering. Runs at 3:30 PM EST.

    Polls UW flow sentiment 4 times over 20 minutes, tracks QQQ intraday
    movement, checks volume, and writes a summary for the 3:50 PM run.
    """
    import time
    import alpaca_client
    import uw_client
    from db.models import init_tables, save_pregame
    import notifications

    logger.info(f"{'='*50}")
    logger.info(f"Leveraged ETF Strategy - PREGAME - {_timestamp()}")
    logger.info(f"{'='*50}")

    init_tables()

    # Check market is open
    today_str = datetime.now(ET).date().isoformat()
    calendar = alpaca_client.get_calendar(today_str)
    if calendar is None:
        logger.info("Market closed today, skipping pregame")
        return

    # Get snapshots for intraday data
    snapshots = alpaca_client.get_snapshot(
        [LEVERAGE_CONFIG["underlying"], LEVERAGE_CONFIG["bull_etf"]]
    )

    qqq_snap = snapshots.get("QQQ", {})
    tqqq_snap = snapshots.get("TQQQ", {})

    qqq_open = qqq_snap.get("daily_bar_open", 0)
    qqq_current = qqq_snap.get("latest_trade_price", 0)
    qqq_high = qqq_snap.get("daily_bar_high", 0)
    qqq_low = qqq_snap.get("daily_bar_low", 0)
    qqq_volume = qqq_snap.get("daily_bar_volume", 0)

    tqqq_open = tqqq_snap.get("daily_bar_open", 0)
    tqqq_current = tqqq_snap.get("latest_trade_price", 0)

    # Intraday calculations
    qqq_intraday_pct = ((qqq_current - qqq_open) / qqq_open * 100) if qqq_open else 0
    tqqq_intraday_pct = ((tqqq_current - tqqq_open) / tqqq_open * 100) if tqqq_open else 0
    qqq_range_pct = ((qqq_high - qqq_low) / qqq_low * 100) if qqq_low else 0

    # Poll UW flow multiple times (4 polls, 5 min apart = 20 min coverage)
    flow_samples = []
    poll_count = 4
    poll_interval = 300  # 5 minutes

    logger.info(f"Starting {poll_count} flow polls (every {poll_interval//60} min)...")

    for i in range(poll_count):
        if i > 0:
            time.sleep(poll_interval)

        flow = uw_client.get_tqqq_flow()
        flow_samples.append(flow)
        logger.info(
            f"  Poll {i+1}/{poll_count}: P/C ratio={flow['ratio']:.2f} "
            f"puts=${flow['put_premium']:,.0f} calls=${flow['call_premium']:,.0f} "
            f"alerts={flow['alert_count']}"
        )

    # Aggregate flow data
    total_put = sum(f["put_premium"] for f in flow_samples)
    total_call = sum(f["call_premium"] for f in flow_samples)
    avg_ratio = (total_put / total_call) if total_call > 0 else (10.0 if total_put > 0 else 1.0)
    bearish_count = sum(1 for f in flow_samples if f["is_bearish"])

    # Determine flow trend
    if len(flow_samples) >= 2:
        ratios = [f["ratio"] for f in flow_samples]
        if ratios[-1] > ratios[0] * 1.3:
            flow_trend = "INCREASINGLY_BEARISH"
        elif ratios[-1] < ratios[0] * 0.7:
            flow_trend = "INCREASINGLY_BULLISH"
        else:
            flow_trend = "STABLE"
    else:
        flow_trend = "INSUFFICIENT_DATA"

    # Estimate relative volume (vs 20-day average from daily bars)
    # Use cached bars if available
    qqq_avg_volume = 0
    try:
        from db.cache import get_cached_bars
        cached = get_cached_bars("QQQ", 20)
        if cached:
            qqq_avg_volume = int(sum(b["volume"] for b in cached) / len(cached))
    except Exception:
        pass

    relative_volume = (qqq_volume / qqq_avg_volume) if qqq_avg_volume > 0 else 0

    # Late-day check: get a fresh snapshot after polling
    try:
        fresh_snap = alpaca_client.get_snapshot([LEVERAGE_CONFIG["underlying"]])
        qqq_latest = fresh_snap.get("QQQ", {}).get("latest_trade_price", qqq_current)
        qqq_last_hour_pct = ((qqq_latest - qqq_current) / qqq_current * 100) if qqq_current else 0
        selling_into_close = qqq_latest < qqq_current and qqq_intraday_pct > 0
    except Exception:
        qqq_latest = qqq_current
        qqq_last_hour_pct = 0
        selling_into_close = False

    # Overall sentiment assessment
    notes = []
    if abs(qqq_intraday_pct) > 2:
        notes.append(f"Large QQQ move: {qqq_intraday_pct:+.1f}%")
    if relative_volume > 1.5:
        notes.append(f"High volume: {relative_volume:.1f}x avg")
    if bearish_count >= 3:
        notes.append(f"Persistent bearish flow ({bearish_count}/{poll_count} polls)")
    if flow_trend == "INCREASINGLY_BEARISH":
        notes.append("Flow deteriorating into close")
    if selling_into_close:
        notes.append("Selling into close (gave up intraday gains)")
    if qqq_range_pct > 3:
        notes.append(f"Wide intraday range: {qqq_range_pct:.1f}%")

    if bearish_count >= 3 or (selling_into_close and avg_ratio > 1.5):
        sentiment = "BEARISH"
    elif bearish_count == 0 and qqq_intraday_pct > 0.5 and flow_trend != "INCREASINGLY_BEARISH":
        sentiment = "BULLISH"
    else:
        sentiment = "NEUTRAL"

    # Save to DB
    pregame_data = {
        "date": today_str,
        "timestamp": _timestamp(),
        "flow_samples": poll_count,
        "flow_put_premium_total": round(total_put, 2),
        "flow_call_premium_total": round(total_call, 2),
        "flow_avg_ratio": round(avg_ratio, 2),
        "flow_trend": flow_trend,
        "flow_bearish_samples": bearish_count,
        "qqq_open": qqq_open,
        "qqq_current": qqq_latest,
        "qqq_intraday_pct": round(qqq_intraday_pct, 2),
        "qqq_intraday_high": qqq_high,
        "qqq_intraday_low": qqq_low,
        "qqq_intraday_range_pct": round(qqq_range_pct, 2),
        "tqqq_open": tqqq_open,
        "tqqq_current": tqqq_current,
        "tqqq_intraday_pct": round(tqqq_intraday_pct, 2),
        "qqq_volume": qqq_volume,
        "qqq_avg_volume": qqq_avg_volume,
        "qqq_relative_volume": round(relative_volume, 2),
        "qqq_last_hour_pct": round(qqq_last_hour_pct, 2),
        "selling_into_close": 1 if selling_into_close else 0,
        "pregame_sentiment": sentiment,
        "pregame_notes": "; ".join(notes) if notes else "No notable signals",
    }
    save_pregame(pregame_data)

    # Send Telegram summary
    text = (
        f"\U0001f3af Pre-Game Intel ({today_str})\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
        f"Sentiment: {sentiment}\n\n"
        f"QQQ Today: {qqq_intraday_pct:+.2f}% (${qqq_latest:.2f})\n"
        f"  Range: ${qqq_low:.2f} - ${qqq_high:.2f} ({qqq_range_pct:.1f}%)\n"
        f"  Volume: {qqq_volume:,} ({relative_volume:.1f}x avg)\n\n"
        f"TQQQ Today: {tqqq_intraday_pct:+.2f}% (${tqqq_current:.2f})\n\n"
        f"Flow ({poll_count} polls over 20 min):\n"
        f"  Avg P/C ratio: {avg_ratio:.2f}\n"
        f"  Trend: {flow_trend}\n"
        f"  Bearish polls: {bearish_count}/{poll_count}\n"
    )
    if notes:
        text += f"\nFlags:\n"
        for n in notes:
            text += f"  \u26a0\ufe0f {n}\n"
    text += f"\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
    text += f"Execution in ~20 min"

    notifications._send_message(text, parse_mode="")

    logger.info(f"Pregame complete: {sentiment}")
    logger.info(f"Notes: {'; '.join(notes) if notes else 'None'}")


def _already_traded_today() -> bool:
    """Check if we already executed a trade today (prevents double-buy from halfday + main run)."""
    from db.models import get_connection
    today_str = datetime.now(ET).date().isoformat()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM decisions WHERE date = ? AND order_action IN ('BUY', 'SELL')",
            [today_str],
        ).fetchone()
        return row["cnt"] > 0 if row else False
    finally:
        conn.close()


def cmd_run(halfday_check: bool = False):
    """Execute the daily strategy pipeline."""
    from strategy.sizing import run_gate_checklist, calculate_target_shares
    from strategy.executor import execute_rebalance
    from db.models import init_tables, log_daily_decision, log_regime_change, log_daily_performance, get_today_pregame
    import alpaca_client
    import notifications

    logger.info(f"{'='*50}")
    logger.info(f"Leveraged ETF Strategy - RUN - {_timestamp()}")
    logger.info(f"{'='*50}")

    init_tables()

    # Fetch data
    data = _fetch_all_data()

    # Half-day check: if --halfday-check and not a half day, exit
    if halfday_check and not data["is_half_day"]:
        logger.info("Not a half day, exiting (--halfday-check mode)")
        return
    if data["is_half_day"]:
        logger.info("Half day detected!")
        notifications.send_halfday_alert()

    # Check market is open
    if data["calendar"] is None:
        logger.info("Market closed today, exiting")
        return

    # Verify data quality
    if data["trading_days_fetched"] < 250:
        msg = f"Only {data['trading_days_fetched']} trading days, need 250 for SMA"
        logger.error(msg)
        notifications.send_error("Insufficient Data", msg)
        return

    # Compute signals
    signals = _compute_signals(data)

    # Check for pregame intel
    pregame = get_today_pregame()
    if pregame:
        logger.info(f"Pregame intel: sentiment={pregame['pregame_sentiment']}, "
                     f"flow_ratio={pregame['flow_avg_ratio']:.2f}, "
                     f"notes={pregame['pregame_notes']}")
        # Override flow data with pregame aggregated data if available
        if pregame["flow_samples"] and pregame["flow_samples"] > 0:
            signals["options_flow_ratio"] = pregame["flow_avg_ratio"]
            signals["options_flow_bearish"] = pregame["flow_bearish_samples"] >= 3
            if signals["options_flow_bearish"]:
                signals["options_flow_adjustment"] = 0.5
    else:
        logger.info("No pregame intel available for today")

    # Run gate checklist
    gate_data = {
        "regime": signals["effective_regime"],
        "qqq_close": signals["qqq_close"],
        "sma_50": signals["sma_50"],
        "sma_250": signals["sma_250"],
        "momentum_score": signals["momentum_score"],
        "realized_vol": signals["realized_vol"],
        "vol_regime": signals["vol_regime"],
        "daily_loss_pct": signals["daily_loss_pct"],
        "qqq_closes": signals["qqq_closes"],
        "holding_days_losing": signals["consecutive_losing_days"],
        "is_execution_window": True,  # We're running in the cron window
        "allocated_capital": signals["allocated_capital"],
        "day_trades_remaining": signals["day_trades_remaining"],
        "options_flow_bearish": signals["options_flow_bearish"],
        "options_flow_ratio": signals["options_flow_ratio"],
        "trading_days_fetched": signals["trading_days_fetched"],
    }
    gates_passed, gates_failed = run_gate_checklist(gate_data)

    # Calculate target position
    sizing_data = {
        "regime": signals["effective_regime"],
        "allocated_capital": signals["allocated_capital"],
        "momentum_score": signals["momentum_score"],
        "vol_regime": signals["vol_regime"],
        "options_flow_adjustment": signals["options_flow_adjustment"],
        "qqq_close": signals["qqq_close"],
        "sma_50": signals["sma_50"],
        "qqq_closes": signals["qqq_closes"],
        "tqqq_price": signals["tqqq_price"],
        "current_shares": signals["current_shares"],
    }
    target = calculate_target_shares(sizing_data)

    logger.info(f"Regime: {signals['effective_regime']} | Momentum: {signals['momentum_score']:.2f}")
    logger.info(f"Vol: {signals['realized_vol']:.1f}% ({signals['vol_regime']})")
    logger.info(f"Gates: {'PASS' if gates_passed else 'FAIL'} ({gates_failed})")
    logger.info(f"Target: {target['target_shares']} shares ({target['action']})")

    # Execute
    is_emergency = signals["effective_regime"] in ("RISK_OFF", "BREAKDOWN")
    should_trade = gates_passed or is_emergency

    # Dedup: skip if we already traded today (prevents halfday + main run double-buy)
    if should_trade and not is_emergency and _already_traded_today():
        logger.info("Already traded today, skipping execution (dedup)")
        should_trade = False

    execution_result = {"executed": False, "action": "HOLD", "order": None, "reason": "Gates failed"}

    if should_trade and target["action"] != "HOLD":
        execution_result = execute_rebalance(
            target["target_shares"],
            signals["current_shares"],
            signals["tqqq_price"],
            alpaca_client,
            is_emergency=is_emergency,
            day_trades_remaining=signals["day_trades_remaining"],
        )
        logger.info(f"Execution: {execution_result['reason']}")
    elif target["action"] == "HOLD":
        execution_result = {"executed": False, "action": "HOLD", "order": None, "reason": "At target"}

    # Log regime change
    if signals["regime_changed"] and signals["previous_regime"]:
        regime_data = {
            "date": datetime.now(ET).date().isoformat(),
            "qqq_close": signals["qqq_close"],
            "qqq_sma_50": signals["sma_50"],
            "qqq_sma_250": signals["sma_250"],
            "trigger_reason": f"Detected: {signals['raw_regime']}",
        }
        log_regime_change(signals["previous_regime"], signals["effective_regime"], regime_data)

        # Send regime alert
        action_desc = execution_result.get("reason", target["action"])
        if is_emergency and signals["current_shares"] > 0:
            action_desc = f"Selling all {signals['current_shares']} shares TQQQ"
        alert_data = {
            **regime_data,
            "realized_vol_20d": signals["realized_vol"],
            "vol_regime": signals["vol_regime"],
            "options_flow_ratio": signals["options_flow_ratio"],
            "options_flow_bearish": signals["options_flow_bearish"],
            "action_description": action_desc,
        }
        notifications.send_regime_alert(signals["previous_regime"], signals["effective_regime"], alert_data)

    # Log decision
    order = execution_result.get("order") or {}
    today_et = datetime.now(ET).date().isoformat()
    decision = {
        "date": today_et,
        "timestamp": _timestamp(),
        "qqq_close": signals["qqq_close"],
        "qqq_sma_50": signals["sma_50"],
        "qqq_sma_250": signals["sma_250"],
        "qqq_pct_above_sma50": signals["pct_above_sma50"],
        "qqq_pct_above_sma250": signals["pct_above_sma250"],
        "qqq_roc_5": signals["momentum"]["roc_fast"],
        "qqq_roc_20": signals["momentum"]["roc_slow"],
        "realized_vol_20d": signals["realized_vol"],
        "vol_regime": signals["vol_regime"],
        "options_flow_put_premium": signals["flow"].get("put_premium", 0),
        "options_flow_call_premium": signals["flow"].get("call_premium", 0),
        "options_flow_ratio": signals["options_flow_ratio"],
        "options_flow_bearish": 1 if signals["options_flow_bearish"] else 0,
        "options_flow_adjustment": signals["options_flow_adjustment"],
        "regime": signals["effective_regime"],
        "regime_changed": 1 if signals["regime_changed"] else 0,
        "previous_regime": signals["previous_regime"],
        "momentum_score": signals["momentum_score"],
        "momentum_factor": signals["momentum"]["raw_score"],
        "gates_passed": 1 if gates_passed else 0,
        "gates_failed": gates_failed,
        "target_allocation_pct": target["target_allocation_pct"],
        "target_dollar_value": target["target_value"],
        "target_shares": target["target_shares"],
        "current_shares": signals["current_shares"],
        "order_action": execution_result.get("action", target["action"]),
        "order_shares": execution_result.get("shares", 0),
        "order_value": execution_result.get("value", 0),
        "order_id": order.get("order_id"),
        "fill_price": order.get("filled_avg_price"),
        "fill_time": None,
        "execution_window": "HALFDAY" if data["is_half_day"] else "NORMAL",
        "account_equity": signals["account_equity"],
        "allocated_capital": signals["allocated_capital"],
        "tqqq_position_value": data["tqqq_position"]["market_value"] if data["tqqq_position"] else 0,
        "tqqq_pnl_pct": data["tqqq_position"]["unrealized_plpc"] * 100 if data["tqqq_position"] else 0,
        "other_positions_value": signals["capital"]["other_positions_value"],
        "cash_balance": signals["cash_balance"],
        "day_trades_remaining": signals["day_trades_remaining"],
        "trading_days_fetched": signals["trading_days_fetched"],
        "is_half_day": 1 if data["is_half_day"] else 0,
    }
    log_daily_decision(decision)

    # Log performance
    tqqq_val = data["tqqq_position"]["market_value"] if data["tqqq_position"] else 0
    tqqq_pnl = data["tqqq_position"]["unrealized_pl"] if data["tqqq_position"] else 0
    perf = {
        "date": today_et,
        "tqqq_shares": signals["current_shares"],
        "tqqq_avg_cost": data["tqqq_position"]["avg_entry_price"] if data["tqqq_position"] else 0,
        "tqqq_current_price": signals["tqqq_price"],
        "tqqq_position_value": tqqq_val,
        "tqqq_pnl_day": tqqq_pnl,
        "tqqq_pnl_total": tqqq_pnl,
        "tqqq_pnl_pct": data["tqqq_position"]["unrealized_plpc"] * 100 if data["tqqq_position"] else 0,
        "regime": signals["effective_regime"],
        "allocated_capital": signals["allocated_capital"],
        "realized_vol": signals["realized_vol"],
        "benchmark_qqq_pct": 0,  # Computed over time
        "strategy_total_return_pct": 0,
    }
    log_daily_performance(perf)

    # Send daily report
    report_data = {
        "regime": signals["effective_regime"],
        "regime_days": signals["regime_hold_days"],
        "qqq_close": signals["qqq_close"],
        "qqq_sma_50": signals["sma_50"],
        "qqq_sma_250": signals["sma_250"],
        "qqq_pct_above_sma50": signals["pct_above_sma50"],
        "qqq_pct_above_sma250": signals["pct_above_sma250"],
        "momentum_score": signals["momentum_score"],
        "realized_vol_20d": signals["realized_vol"],
        "vol_regime": signals["vol_regime"],
        "options_flow_ratio": signals["options_flow_ratio"],
        "options_flow_bearish": signals["options_flow_bearish"],
        "trading_days_fetched": signals["trading_days_fetched"],
        "gates_passed": 14 - len(gates_failed),
        "gates_failed_list": gates_failed,
        "day_trades_remaining": signals["day_trades_remaining"],
        "order_action": execution_result.get("action", target["action"]),
        "order_shares": execution_result.get("shares", abs(target["delta_shares"])),
        "target_dollar_value": target["target_value"],
        "allocated_capital": signals["allocated_capital"],
        "current_shares": signals["current_shares"],
        "tqqq_position_value": tqqq_val,
        "tqqq_pnl_pct": perf["tqqq_pnl_pct"],
    }
    if pregame:
        report_data["pregame_sentiment"] = pregame["pregame_sentiment"]
        report_data["pregame_notes"] = pregame["pregame_notes"]
    notifications.send_daily_report(report_data)

    logger.info("Daily run complete")


def cmd_status():
    """Show current state without trading."""
    from strategy.signals import (
        calculate_momentum, calculate_realized_vol,
        classify_vol_regime,
    )
    from strategy.regime import detect_regime, get_effective_regime
    from strategy.sizing import get_allocated_capital, run_gate_checklist, calculate_target_shares
    from db.models import init_tables, get_last_regime, get_regime_duration_days
    import uw_client

    init_tables()
    data = _fetch_all_data()
    signals = _compute_signals(data)

    print(f"\n{'='*50}")
    print(f"  Leveraged ETF Strategy - STATUS")
    print(f"  {_timestamp()}")
    print(f"{'='*50}\n")

    # Regime
    print(f"Regime: {signals['effective_regime']} (raw: {signals['raw_regime']})")
    print(f"  Previous: {signals['previous_regime'] or 'N/A'}")
    print(f"  Hold days: {signals['regime_hold_days']}")

    # QQQ
    print(f"\nQQQ: ${signals['qqq_close']:.2f}")
    print(f"  SMA-50:  ${signals['sma_50']:.2f} ({signals['pct_above_sma50']:+.1f}%)")
    print(f"  SMA-250: ${signals['sma_250']:.2f} ({signals['pct_above_sma250']:+.1f}%)")
    print(f"  Trading days loaded: {signals['trading_days_fetched']}")

    # Momentum
    mom = signals["momentum"]
    print(f"\nMomentum: {signals['momentum_score']:.2f}")
    print(f"  ROC-5: {mom['roc_fast']:+.4f}")
    print(f"  ROC-20: {mom['roc_slow']:+.4f}")

    # Vol
    print(f"\nRealized Vol: {signals['realized_vol']:.1f}% ({signals['vol_regime']})")

    # Flow
    flow = signals["flow"]
    print(f"\nOptions Flow: P/C ratio = {flow['ratio']:.2f} ({'Bearish' if flow['is_bearish'] else 'Neutral'})")
    print(f"  Put premium: ${flow['put_premium']:,.0f}")
    print(f"  Call premium: ${flow['call_premium']:,.0f}")
    print(f"  Alerts: {flow['alert_count']}")
    if flow.get("error"):
        print(f"  Warning: {flow['error']}")

    # Position
    tqqq = data["tqqq_position"]
    if tqqq:
        print(f"\nTQQQ Position: {tqqq['qty']} shares @ ${tqqq['avg_entry_price']:.2f}")
        print(f"  Market value: ${tqqq['market_value']:,.2f}")
        print(f"  P/L: ${tqqq['unrealized_pl']:,.2f} ({tqqq['unrealized_plpc']*100:+.1f}%)")
    else:
        print(f"\nTQQQ Position: None (100% cash)")

    # Capital
    cap = signals["capital"]
    print(f"\nCapital Allocation:")
    print(f"  Total equity: ${cap['total_equity']:,.2f}")
    print(f"  Other positions: ${cap['other_positions_value']:,.2f}")
    print(f"  Allocated (30% cap): ${cap['allocated_capital']:,.2f}")
    print(f"  Cash available: ${cap['cash_available']:,.2f}")

    # PDT
    print(f"\nPDT: {signals['day_trades_remaining']} day trades used")

    # Gate checklist
    gate_data = {
        "regime": signals["effective_regime"],
        "qqq_close": signals["qqq_close"],
        "sma_50": signals["sma_50"],
        "sma_250": signals["sma_250"],
        "momentum_score": signals["momentum_score"],
        "realized_vol": signals["realized_vol"],
        "vol_regime": signals["vol_regime"],
        "daily_loss_pct": signals["daily_loss_pct"],
        "qqq_closes": signals["qqq_closes"],
        "holding_days_losing": signals["consecutive_losing_days"],
        "is_execution_window": True,
        "allocated_capital": signals["allocated_capital"],
        "day_trades_remaining": signals["day_trades_remaining"],
        "options_flow_bearish": signals["options_flow_bearish"],
        "options_flow_ratio": signals.get("options_flow_ratio", 1.0),
        "trading_days_fetched": signals["trading_days_fetched"],
    }
    gates_passed, gates_failed = run_gate_checklist(gate_data)

    print(f"\nGate Checklist: {'PASS' if gates_passed else 'FAIL'}")
    if gates_failed:
        for g in gates_failed:
            print(f"  FAIL: {g}")
    else:
        print(f"  All 14 gates passed")

    # What-if
    sizing_data = {
        "regime": signals["effective_regime"],
        "allocated_capital": signals["allocated_capital"],
        "momentum_score": signals["momentum_score"],
        "vol_regime": signals["vol_regime"],
        "options_flow_adjustment": signals["options_flow_adjustment"],
        "qqq_close": signals["qqq_close"],
        "sma_50": signals["sma_50"],
        "qqq_closes": signals["qqq_closes"],
        "tqqq_price": signals["tqqq_price"],
        "current_shares": signals["current_shares"],
    }
    target = calculate_target_shares(sizing_data)

    print(f"\nWhat-if (no execution):")
    print(f"  Target: {target['target_shares']} shares (${target['target_value']:,.0f})")
    print(f"  Current: {target['current_shares']} shares")
    print(f"  Delta: {target['delta_shares']:+d} shares (${target['delta_value']:+,.0f})")
    print(f"  Action: {target['action']}")
    if target["limiting_factors"]:
        print(f"  Limiting: {', '.join(target['limiting_factors'])}")

    print(f"\nHalf day: {'Yes' if data['is_half_day'] else 'No'}")
    print(f"{'='*50}\n")


def cmd_backtest():
    """Run historical backtest simulation."""
    import alpaca_client
    from db.models import get_connection, init_tables
    from db.cache import get_bars_with_cache
    from strategy.regime import detect_regime, get_regime_target_pct
    from strategy.signals import calculate_momentum, calculate_realized_vol, classify_vol_regime, get_vol_adjustment
    import notifications

    logger.info("Starting backtest...")
    init_tables()

    # Fetch historical data (2+ years)
    end = datetime.now(ET).date()
    start = end - timedelta(days=900)  # ~2.5 years
    conn = get_connection()

    qqq_bars = get_bars_with_cache(
        "QQQ", 900, alpaca_client.fetch_bars_for_cache, conn
    )
    tqqq_bars = get_bars_with_cache(
        "TQQQ", 900, alpaca_client.fetch_bars_for_cache, conn
    )

    logger.info(f"QQQ bars: {len(qqq_bars)}, TQQQ bars: {len(tqqq_bars)}")

    if len(qqq_bars) < 280:
        logger.error(f"Insufficient QQQ data: {len(qqq_bars)} bars (need 280+)")
        return

    # TQQQ split check
    for bar in tqqq_bars:
        if bar["date"] < "2022-01-01" and bar["close"] < 5:
            logger.warning("TQQQ pre-2022 prices appear unadjusted for reverse split!")
            break

    # Align dates
    qqq_by_date = {b["date"]: b for b in qqq_bars}
    tqqq_by_date = {b["date"]: b for b in tqqq_bars}
    common_dates = sorted(set(qqq_by_date.keys()) & set(tqqq_by_date.keys()))

    if len(common_dates) < 260:
        logger.error(f"Only {len(common_dates)} common dates, need 260+")
        return

    # Simulation
    initial_capital = 100000.0
    cash = initial_capital
    shares = 0
    prev_regime = None
    regime_change_day = 0
    peak_value = initial_capital
    max_drawdown = 0.0
    num_trades = 0
    days_in_market = 0

    results = []
    qqq_start_price = qqq_by_date[common_dates[250]]["close"]  # Start after warmup
    tqqq_start_price = tqqq_by_date[common_dates[250]]["close"]

    for i, dt in enumerate(common_dates):
        if i < 250:  # Warmup period for SMA-250
            continue

        qqq_bar = qqq_by_date[dt]
        tqqq_bar = tqqq_by_date[dt]

        # Build closes array
        warmup_dates = common_dates[max(0, i - 260):i + 1]
        closes = [qqq_by_date[d]["close"] for d in warmup_dates if d in qqq_by_date]

        if len(closes) < 250:
            continue

        qqq_close = closes[-1]
        sma_50 = float(np.mean(closes[-50:]))
        sma_250 = float(np.mean(closes[-250:]))

        # Regime
        regime = detect_regime(qqq_close, sma_50, sma_250)

        # Oscillation protection
        if prev_regime and regime != prev_regime:
            if regime not in ("RISK_OFF", "BREAKDOWN"):
                if (i - regime_change_day) < LEVERAGE_CONFIG["min_regime_hold_days"]:
                    regime = prev_regime

        if regime != prev_regime:
            regime_change_day = i
            prev_regime = regime

        # Signals
        mom = calculate_momentum(closes)
        vol = calculate_realized_vol(closes)
        vol_regime = classify_vol_regime(vol)
        vol_adj = get_vol_adjustment(vol_regime)

        # Target allocation
        regime_pct = get_regime_target_pct(regime)
        target_pct = regime_pct

        # Apply momentum scaling
        if mom["score"] < LEVERAGE_CONFIG["min_momentum_score"]:
            target_pct = LEVERAGE_CONFIG["min_position_pct"]
        elif mom["score"] < 0.8:
            min_pct = LEVERAGE_CONFIG["min_position_pct"]
            scale = (mom["score"] - LEVERAGE_CONFIG["min_momentum_score"]) / (0.8 - LEVERAGE_CONFIG["min_momentum_score"])
            target_pct = min_pct + (regime_pct - min_pct) * scale

        # Vol adjustment
        target_pct *= vol_adj

        # Overextension check
        if sma_50 > 0 and (qqq_close - sma_50) / sma_50 > LEVERAGE_CONFIG["mean_reversion_threshold"]:
            target_pct *= 0.5

        # Capital and shares
        portfolio_value = cash + shares * tqqq_bar["close"]
        allocated = portfolio_value * LEVERAGE_CONFIG["max_portfolio_pct"]
        target_value = allocated * target_pct
        target_shares = max(0, int(target_value / tqqq_bar["close"])) if tqqq_bar["close"] > 0 else 0

        # Execute (at close price)
        delta = target_shares - shares
        if abs(delta * tqqq_bar["close"]) >= LEVERAGE_CONFIG["min_trade_value"] or (regime in ("RISK_OFF", "BREAKDOWN") and shares > 0):
            if delta > 0:
                cost = delta * tqqq_bar["close"]
                if cost <= cash:
                    shares += delta
                    cash -= cost
                    num_trades += 1
            elif delta < 0:
                proceeds = abs(delta) * tqqq_bar["close"]
                shares += delta  # delta is negative
                cash += proceeds
                num_trades += 1

                # Ensure no negative shares
                if shares < 0:
                    cash += abs(shares) * tqqq_bar["close"]
                    shares = 0

        portfolio_value = cash + shares * tqqq_bar["close"]
        peak_value = max(peak_value, portfolio_value)
        drawdown = (peak_value - portfolio_value) / peak_value * 100
        max_drawdown = max(max_drawdown, drawdown)

        if shares > 0:
            days_in_market += 1

        # Benchmarks
        qqq_return = (qqq_bar["close"] / qqq_start_price - 1) * 100
        tqqq_return = (tqqq_bar["close"] / tqqq_start_price - 1) * 100
        strategy_return = (portfolio_value / initial_capital - 1) * 100

        results.append({
            "date": dt,
            "qqq_close": qqq_close,
            "tqqq_close": tqqq_bar["close"],
            "regime": regime,
            "target_shares": target_shares,
            "held_shares": shares,
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "pnl_day": 0,  # Simplified
            "pnl_total_pct": round(strategy_return, 2),
            "drawdown_pct": round(drawdown, 2),
            "qqq_buy_hold_pct": round(qqq_return, 2),
            "tqqq_buy_hold_pct": round(tqqq_return, 2),
        })

    # Write results to DB
    conn.executemany(
        "INSERT INTO backtest_results (date, qqq_close, tqqq_close, regime, target_shares, "
        "held_shares, portfolio_value, cash, pnl_day, pnl_total_pct, drawdown_pct, "
        "qqq_buy_hold_pct, tqqq_buy_hold_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r["date"], r["qqq_close"], r["tqqq_close"], r["regime"], r["target_shares"],
          r["held_shares"], r["portfolio_value"], r["cash"], r["pnl_day"],
          r["pnl_total_pct"], r["drawdown_pct"], r["qqq_buy_hold_pct"],
          r["tqqq_buy_hold_pct"]) for r in results],
    )
    conn.commit()

    # Export CSV
    csv_path = DATA_DIR / "backtest_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Backtest results saved to {csv_path}")

    # Summary stats
    final = results[-1] if results else {}
    stats = {
        "start_date": results[0]["date"] if results else "N/A",
        "end_date": final.get("date", "N/A"),
        "total_return_pct": final.get("pnl_total_pct", 0),
        "max_drawdown_pct": max_drawdown,
        "qqq_buy_hold_pct": final.get("qqq_buy_hold_pct", 0),
        "tqqq_buy_hold_pct": final.get("tqqq_buy_hold_pct", 0),
        "num_trades": num_trades,
        "days_in_market": days_in_market,
        "total_days": len(results),
    }

    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"  Period: {stats['start_date']} to {stats['end_date']}")
    print(f"  Strategy Return: {stats['total_return_pct']:+.1f}%")
    print(f"  Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
    print(f"  Trades: {stats['num_trades']}")
    print(f"  Days in Market: {stats['days_in_market']}/{stats['total_days']}")
    print(f"\n  Benchmarks:")
    print(f"    QQQ Buy & Hold:  {stats['qqq_buy_hold_pct']:+.1f}%")
    print(f"    TQQQ Buy & Hold: {stats['tqqq_buy_hold_pct']:+.1f}%")
    print(f"{'='*50}\n")

    # Send to Telegram
    notifications.send_backtest_summary(stats, str(csv_path))

    conn.close()
    return stats


def cmd_force_exit():
    """Emergency sell all TQQQ."""
    from strategy.executor import force_exit
    import alpaca_client
    import notifications

    logger.info("FORCE EXIT initiated")
    result = force_exit(alpaca_client)

    if result["executed"]:
        notifications.send_error(
            "FORCE EXIT EXECUTED",
            f"Sold {result['shares_sold']} shares TQQQ\nOrder: {result['order']}"
        )
    else:
        print(f"Force exit: {result['reason']}")

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python job.py {run|pregame|status|backtest|force_exit}")
        print("  run [--halfday-check]  Execute daily strategy")
        print("  pregame                Pre-execution intel gathering (3:30 PM)")
        print("  status                 Show current state (no trading)")
        print("  backtest               Run historical backtest")
        print("  force_exit             Emergency sell all TQQQ")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        halfday = "--halfday-check" in sys.argv
        cmd_run(halfday_check=halfday)
    elif command == "pregame":
        cmd_pregame()
    elif command == "status":
        cmd_status()
    elif command == "backtest":
        cmd_backtest()
    elif command == "force_exit":
        cmd_force_exit()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
