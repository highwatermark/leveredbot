#!/usr/bin/env python3
"""
CLI entry point for the leveraged ETF strategy.

Usage:
    python job.py run               # Execute daily strategy
    python job.py run --halfday-check  # Only run if today is a half day
    python job.py morning           # Morning position check (9:35 AM ET)
    python job.py midday            # Midday position check (12:30 PM ET)
    python job.py pregame           # Pre-execution intel gathering (3:30 PM)
    python job.py status            # Show current state without trading
    python job.py backtest          # Run historical backtest
    python job.py force_exit        # Emergency exit all TQQQ
"""

import sys
import logging
import json
import importlib.util
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


def _calc_day_trades_remaining(account: dict) -> int:
    """Convert Alpaca's daytrade_count (trades USED) to trades REMAINING.

    PDT accounts with equity >= $25K have unlimited day trades.
    Non-PDT accounts get 3 day trades per rolling 5 business days.
    """
    if account.get("pattern_day_trader") and account.get("equity", 0) >= 25_000:
        return 999  # Unlimited for PDT accounts above $25K
    used = account.get("daytrade_count", 0)
    return max(0, 3 - used)


def _resolve_model_signal(
    knn_direction: str,
    knn_confidence: float,
    knn_adjustment: float,
    xgb_direction: str,
    xgb_confidence: float,
    xgb_adjustment: float,
) -> dict:
    """Arbitrate raw model outputs into a single effective model signal."""
    primary = LEVERAGE_CONFIG.get("model_primary", "knn")
    disagreement_action = LEVERAGE_CONFIG.get("model_disagreement_action", "reduce")
    disagreement_adj = LEVERAGE_CONFIG.get("model_disagreement_adjustment", 0.75)

    knn_active = knn_direction != "FLAT"
    xgb_active = xgb_direction != "FLAT"

    if not knn_active and not xgb_active:
        return {
            "direction": "FLAT",
            "confidence": max(knn_confidence, xgb_confidence),
            "adjustment": 1.0,
            "source": "neutral",
            "disagreement": False,
        }

    if knn_active and xgb_active and knn_direction == xgb_direction:
        return {
            "direction": knn_direction,
            "confidence": round(max(knn_confidence, xgb_confidence), 4),
            "adjustment": round(min(knn_adjustment, xgb_adjustment), 4),
            "source": "agreement",
            "disagreement": False,
        }

    if knn_active and not xgb_active:
        return {
            "direction": knn_direction,
            "confidence": round(knn_confidence, 4),
            "adjustment": round(min(1.0, knn_adjustment * disagreement_adj), 4),
            "source": "knn_only",
            "disagreement": True,
        }

    if xgb_active and not knn_active:
        return {
            "direction": xgb_direction,
            "confidence": round(xgb_confidence, 4),
            "adjustment": round(min(1.0, xgb_adjustment * disagreement_adj), 4),
            "source": "xgb_only",
            "disagreement": True,
        }

    if disagreement_action == "flat":
        return {
            "direction": "FLAT",
            "confidence": round(max(knn_confidence, xgb_confidence), 4),
            "adjustment": 1.0,
            "source": "conflict_flat",
            "disagreement": True,
        }

    # Opposite directional calls default to neutralized risk under "reduce" as well.
    primary_direction = knn_direction if primary == "knn" else xgb_direction
    primary_confidence = knn_confidence if primary == "knn" else xgb_confidence
    return {
        "direction": "FLAT",
        "confidence": round(primary_confidence, 4),
        "adjustment": 1.0,
        "source": f"conflict_{primary_direction.lower()}",
        "disagreement": True,
    }


def _backtest_portfolio_value(
    cash: float,
    tqqq_shares: int,
    sqqq_shares: int,
    tqqq_price: float,
    sqqq_price: float,
) -> float:
    """Mark the sleeve at a specific price snapshot."""
    return cash + (tqqq_shares * tqqq_price) + (sqqq_shares * sqqq_price)


def _backtest_holding_days(entry_date: str | None, current_date: str) -> int:
    """Return holding days for backtest position state."""
    if not entry_date:
        return 0
    try:
        return max(0, (date.fromisoformat(current_date) - date.fromisoformat(entry_date)).days)
    except ValueError:
        return 0


def _build_backtest_position_state(
    symbol: str,
    shares: int,
    meta: dict,
    bar: dict,
    prev_bar: dict | None,
    current_price: float,
    current_date: str,
):
    """Construct a PositionState from historical bars for pure PM checks."""
    from strategy.position_manager import PositionState

    avg_entry_price = meta.get("avg_entry_price", 0.0)
    unrealized_pnl_pct = ((current_price - avg_entry_price) / avg_entry_price) if avg_entry_price > 0 else 0.0
    prev_close = prev_bar["close"] if prev_bar else bar["open"]

    return PositionState(
        symbol=symbol,
        shares=shares,
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        market_value=shares * current_price,
        unrealized_pnl_pct=unrealized_pnl_pct,
        entry_date=meta.get("entry_date"),
        holding_days=_backtest_holding_days(meta.get("entry_date"), current_date),
        intraday_high=bar["high"],
        intraday_low=bar["low"],
        intraday_open=bar["open"],
        prev_close=prev_close,
        overnight_gap_pct=((bar["open"] - prev_close) / prev_close) if prev_close > 0 else 0.0,
        intraday_change_pct=((current_price - bar["open"]) / bar["open"]) if bar["open"] > 0 else 0.0,
        intraday_drawdown_pct=((bar["high"] - current_price) / bar["high"]) if bar["high"] > 0 else 0.0,
    )


def _reset_backtest_position(meta: dict) -> None:
    """Reset per-position bookkeeping after a full exit."""
    meta["avg_entry_price"] = 0.0
    meta["entry_date"] = None
    meta["high_watermark"] = 0.0
    meta["tiers_taken"] = []
    meta["gate_fail_streak"] = 0


def _apply_backtest_trade(
    symbol: str,
    delta_shares: int,
    fill_price: float,
    current_date: str,
    cash: float,
    share_map: dict,
    meta_map: dict,
) -> tuple[float, bool]:
    """Apply a simulated fill and update per-position state."""
    if delta_shares == 0:
        return cash, False

    shares_before = share_map[symbol]
    meta = meta_map[symbol]

    if delta_shares > 0:
        cost = delta_shares * fill_price
        if cost > cash:
            affordable = int(cash / fill_price) if fill_price > 0 else 0
            if affordable <= 0:
                return cash, False
            delta_shares = affordable
            cost = delta_shares * fill_price

        total_cost_before = shares_before * meta.get("avg_entry_price", 0.0)
        new_shares = shares_before + delta_shares
        meta["avg_entry_price"] = ((total_cost_before + cost) / new_shares) if new_shares > 0 else 0.0
        if shares_before == 0:
            meta["entry_date"] = current_date
            meta["tiers_taken"] = []
            meta["gate_fail_streak"] = 0
            meta["high_watermark"] = fill_price
        else:
            meta["high_watermark"] = max(meta.get("high_watermark", 0.0), fill_price)

        share_map[symbol] = new_shares
        cash -= cost
        return cash, True

    shares_to_sell = min(shares_before, abs(delta_shares))
    if shares_to_sell <= 0:
        return cash, False

    cash += shares_to_sell * fill_price
    remaining = shares_before - shares_to_sell
    share_map[symbol] = remaining
    if remaining <= 0:
        _reset_backtest_position(meta)
    return cash, True


def _run_backtest_morning_window(
    pm,
    symbol: str,
    shares: int,
    meta: dict,
    bar: dict,
    prev_bar: dict | None,
    qqq_bar: dict,
    prev_equity: float,
    sma_250: float,
    current_date: str,
) -> tuple[object | None, float]:
    """Mirror morning PM checks using daily open as the 9:35 snapshot."""
    if shares <= 0:
        return None, prev_equity

    state = _build_backtest_position_state(symbol, shares, meta, bar, prev_bar, bar["open"], current_date)
    account_equity = prev_equity + ((bar["open"] - (prev_bar["close"] if prev_bar else bar["open"])) * shares)
    daily_loss = pm.check_daily_loss_limit(account_equity, prev_equity)
    if daily_loss.should_exit:
        daily_loss.shares_to_sell = shares
        return daily_loss, bar["open"]

    candidates = [
        pm.check_gap_down(state),
        pm.check_stop_loss(state),
        pm.check_regime_emergency(state, qqq_bar["open"], sma_250),
    ]
    decision = pm._select_decision([c for c in candidates if c.should_exit])
    if decision:
        decision.shares_to_sell = decision.shares_to_sell or shares
    return decision, bar["open"]


def _run_backtest_midday_window(
    pm,
    symbol: str,
    shares: int,
    meta: dict,
    bar: dict,
    prev_bar: dict | None,
    prev_volume: float,
    prev_equity: float,
    current_date: str,
) -> tuple[object | None, float, float | None]:
    """Mirror midday PM checks using low for defensive exits and high for profit tiers."""
    if shares <= 0:
        return None, prev_equity, None

    adverse_price = bar["low"]
    account_equity = prev_equity + ((adverse_price - (prev_bar["close"] if prev_bar else bar["open"])) * shares)
    daily_loss = pm.check_daily_loss_limit(account_equity, prev_equity)
    if daily_loss.should_exit:
        daily_loss.shares_to_sell = shares
        return daily_loss, adverse_price, None

    meta["high_watermark"] = max(meta.get("high_watermark", 0.0), bar["high"])
    adverse_state = _build_backtest_position_state(symbol, shares, meta, bar, prev_bar, adverse_price, current_date)
    full_exits = [
        pm.check_stop_loss(adverse_state),
        pm.check_trailing_stop(adverse_state, meta.get("high_watermark", adverse_state.avg_entry_price)),
        pm.check_vol_spike(adverse_state, bar["volume"], prev_volume),
        pm.check_max_hold_period(adverse_state),
    ]
    decision = pm._select_decision([c for c in full_exits if c.should_exit])
    if decision:
        decision.shares_to_sell = decision.shares_to_sell or shares
        return decision, adverse_price, None

    if pm.cfg.get("pm_profit_taking_enabled", True):
        favorable_state = _build_backtest_position_state(symbol, shares, meta, bar, prev_bar, bar["high"], current_date)
        profit = pm.check_partial_profit(favorable_state, meta.get("tiers_taken", []))
        if profit.should_exit:
            return profit, bar["high"], profit.shares_to_sell

    return None, adverse_price, None


def _fetch_all_data(use_cache: bool = True, conn=None) -> "MarketData":
    """Fetch all market data needed for the strategy."""
    from concurrent.futures import ThreadPoolExecutor
    import alpaca_client
    from db.cache import get_bars_with_cache
    from pipeline_types import MarketData

    logger.info("Fetching market data...")

    today_str = datetime.now(ET).date().isoformat()

    # Parallel API calls for independent data
    with ThreadPoolExecutor(max_workers=3) as pool:
        account_future = pool.submit(alpaca_client.get_account)
        positions_future = pool.submit(alpaca_client.get_positions)
        calendar_future = pool.submit(alpaca_client.get_calendar, today_str)

        account = account_future.result()
        positions = positions_future.result()
        calendar = calendar_future.result()

    tqqq_pos = None
    sqqq_pos = None
    for p in positions:
        if p["symbol"] == LEVERAGE_CONFIG["bull_etf"]:
            tqqq_pos = p
        elif p["symbol"] == LEVERAGE_CONFIG["bear_etf"]:
            sqqq_pos = p

    is_half_day = calendar.get("is_half_day", False) if calendar else False

    # Snapshots
    try:
        snap_symbols = [LEVERAGE_CONFIG["underlying"], LEVERAGE_CONFIG["bull_etf"]]
        if LEVERAGE_CONFIG.get("use_sqqq_trading", False):
            snap_symbols.append(LEVERAGE_CONFIG["bear_etf"])
        snapshots = alpaca_client.get_snapshot(snap_symbols)
    except Exception as e:
        logger.warning(f"Snapshot fetch failed: {e}")
        snapshots = {}

    # Historical bars
    cal_days = LEVERAGE_CONFIG["history_calendar_days"]
    if use_cache:
        qqq_bars = get_bars_with_cache(
            LEVERAGE_CONFIG["underlying"], cal_days,
            alpaca_client.fetch_bars_for_cache, conn
        )
    else:
        start = (datetime.now(ET).date() - timedelta(days=cal_days)).isoformat()
        end = datetime.now(ET).date().isoformat()
        qqq_bars = alpaca_client.get_bars(LEVERAGE_CONFIG["underlying"], start, end)

    # Cross-asset bars for k-NN/XGBoost features (TLT, GLD, IWM)
    cross_asset_bars = {}
    if LEVERAGE_CONFIG.get("use_knn_signal", False) and use_cache:
        for sym in ("TLT", "GLD", "IWM"):
            try:
                cross_asset_bars[sym] = get_bars_with_cache(
                    sym, cal_days, alpaca_client.fetch_bars_for_cache, conn
                )
            except Exception as e:
                logger.warning(f"Cross-asset bar fetch failed for {sym}: {e}")

    # Microstructure features (intraday-derived, cached)
    microstructure_by_date = {}
    if LEVERAGE_CONFIG.get("use_microstructure", False) and LEVERAGE_CONFIG.get("use_knn_signal", False) and use_cache:
        try:
            from db.cache import get_microstructure_with_cache
            microstructure_by_date = get_microstructure_with_cache(
                LEVERAGE_CONFIG["underlying"], cal_days,
                alpaca_client.get_intraday_bars, conn
            )
        except Exception as e:
            logger.warning(f"Microstructure fetch failed: {e}")

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

    # SQQQ price from snapshot or position
    sqqq_price = None
    if LEVERAGE_CONFIG["bear_etf"] in snapshots:
        sqqq_price = snapshots[LEVERAGE_CONFIG["bear_etf"]].get("latest_trade_price")
    if sqqq_price is None and sqqq_pos:
        sqqq_price = sqqq_pos["current_price"]

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

    return MarketData(
        account=account,
        positions=positions,
        tqqq_position=tqqq_pos,
        calendar=calendar,
        is_half_day=is_half_day,
        snapshots=snapshots,
        qqq_bars=qqq_bars,
        qqq_closes=qqq_closes,
        qqq_current=qqq_current,
        tqqq_price=tqqq_price,
        daily_loss_pct=daily_loss_pct,
        trading_days_fetched=len(qqq_bars),
        sqqq_position=sqqq_pos,
        sqqq_price=sqqq_price,
        cross_asset_bars=cross_asset_bars,
        microstructure_by_date=microstructure_by_date,
    )


def _compute_signals(data: "MarketData", conn=None) -> "StrategySignals":
    """Compute all strategy signals from market data."""
    from strategy.signals import (
        calculate_momentum, calculate_realized_vol,
        classify_vol_regime, get_vol_adjustment, check_options_flow,
    )
    from strategy.regime import detect_regime, get_effective_regime
    from strategy.sizing import get_allocated_capital
    from db.models import get_strategy_state
    from pipeline_types import StrategySignals
    import uw_client

    closes = data.qqq_closes
    qqq_close = data.qqq_current or (closes[-1] if closes else 0)

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
    if LEVERAGE_CONFIG.get("use_combined_flow", False):
        flow = uw_client.get_combined_flow()
    else:
        flow = uw_client.get_tqqq_flow()
    is_bearish, flow_adj = check_options_flow(flow)

    # Strategy state (batched: regime + duration + losing days)
    state = get_strategy_state(conn)
    prev_regime = state["last_regime"]
    hold_days = state["regime_duration_days"]

    # Regime
    raw_regime = detect_regime(qqq_close, sma_50, sma_250)
    effective_regime = get_effective_regime(raw_regime, prev_regime, hold_days)

    # Capital
    account = data.account
    positions = data.positions
    capital = get_allocated_capital(account["equity"], positions)

    # Percentages above SMAs
    pct_above_sma50 = ((qqq_close / sma_50) - 1) * 100 if sma_50 > 0 else 0
    pct_above_sma250 = ((qqq_close / sma_250) - 1) * 100 if sma_250 > 0 else 0

    current_shares = data.tqqq_position["qty"] if data.tqqq_position else 0
    sqqq_current_shares = data.sqqq_position["qty"] if data.sqqq_position else 0
    has_tqqq = data.tqqq_position is not None and current_shares > 0

    # k-NN signal overlay
    knn_direction = "FLAT"
    knn_confidence = 0.5
    knn_adjustment = 1.0
    knn_probabilities = [0.5, 0.5]
    # XGBoost signal (always tracked separately)
    xgb_direction = "FLAT"
    xgb_confidence = 0.5
    xgb_adjustment = 1.0
    xgb_probabilities = [0.5, 0.5]

    if LEVERAGE_CONFIG.get("use_knn_signal", False):
        try:
            from strategy.vix_data import get_vix_data
            from pathlib import Path

            prediction_model = LEVERAGE_CONFIG.get("prediction_model", "knn")
            cross_bars = data.cross_asset_bars
            micro_data = data.microstructure_by_date

            # Fetch VIX data (cached, append-only)
            vix_by_date = {}
            try:
                vix_by_date = get_vix_data()
            except Exception as e:
                logger.warning(f"VIX data fetch failed, using defaults: {e}")

            # Run k-NN if selected
            if prediction_model in ("knn", "both"):
                from strategy.knn_signal import KNNSignal
                knn_model_path = Path(__file__).parent / LEVERAGE_CONFIG.get("knn_model_path", "data/knn_model.pkl")
                knn = KNNSignal()
                loaded = False
                if knn_model_path.exists():
                    loaded = knn.load(knn_model_path)
                if not loaded and len(data.qqq_bars) >= 400:
                    knn.fit_from_bars(data.qqq_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_bars, microstructure_by_date=micro_data)
                    knn.save(knn_model_path)
                if knn.is_fitted:
                    knn_result = knn.predict(data.qqq_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_bars, microstructure_by_date=micro_data)
                    if prediction_model == "knn" or prediction_model == "both":
                        knn_direction = knn_result["direction"]
                        knn_confidence = knn_result["confidence"]
                        knn_adjustment = knn_result["adjustment"]
                        knn_probabilities = knn_result["probabilities"]
                    logger.info(f"k-NN: {knn_result['direction']} (conf={knn_result['confidence']:.2f}, adj={knn_result['adjustment']}, P(down)={knn_result['probabilities'][0]:.2f}, P(up)={knn_result['probabilities'][1]:.2f})")

            # Run XGBoost if selected and available in the environment
            should_run_xgb = (
                LEVERAGE_CONFIG.get("use_xgb_signal", True)
                and prediction_model in ("xgb", "both")
            )
            if should_run_xgb:
                if importlib.util.find_spec("xgboost") is None:
                    logger.warning("xgboost not installed; skipping XGBoost signal")
                else:
                    from strategy.xgb_signal import XGBSignal
                    xgb_model_path = Path(__file__).parent / LEVERAGE_CONFIG.get("xgb_model_path", "data/xgb_model.pkl")
                    xgb = XGBSignal()
                    loaded = False
                    if xgb_model_path.exists():
                        loaded = xgb.load(xgb_model_path)
                    if not loaded and len(data.qqq_bars) >= 400:
                        xgb.fit_from_bars(data.qqq_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_bars, microstructure_by_date=micro_data)
                        xgb.save(xgb_model_path)
                    if xgb.is_fitted:
                        xgb_result = xgb.predict(data.qqq_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_bars, microstructure_by_date=micro_data)
                        xgb_direction = xgb_result["direction"]
                        xgb_confidence = xgb_result["confidence"]
                        xgb_adjustment = xgb_result["adjustment"]
                        xgb_probabilities = xgb_result["probabilities"]
                        logger.info(f"XGBoost: {xgb_direction} (conf={xgb_confidence:.2f}, adj={xgb_adjustment}, P(down)={xgb_probabilities[0]:.2f}, P(up)={xgb_probabilities[1]:.2f})")

        except Exception as e:
            logger.warning(f"Prediction model signal failed, using neutral: {e}")

    effective_model = _resolve_model_signal(
        knn_direction,
        knn_confidence,
        knn_adjustment,
        xgb_direction,
        xgb_confidence,
        xgb_adjustment,
    )

    return StrategySignals(
        qqq_close=qqq_close,
        sma_50=round(sma_50, 2),
        sma_250=round(sma_250, 2),
        pct_above_sma50=round(pct_above_sma50, 2),
        pct_above_sma250=round(pct_above_sma250, 2),
        momentum=mom,
        momentum_score=mom["score"],
        realized_vol=vol,
        vol_regime=vol_regime,
        vol_adjustment=vol_adj,
        flow=flow,
        options_flow_bearish=is_bearish,
        options_flow_adjustment=flow_adj,
        options_flow_ratio=flow.get("ratio", 1.0),
        raw_regime=raw_regime,
        effective_regime=effective_regime,
        previous_regime=prev_regime,
        regime_hold_days=hold_days,
        regime_changed=effective_regime != prev_regime if prev_regime else False,
        capital=capital,
        allocated_capital=capital["allocated_capital"],
        current_shares=current_shares,
        tqqq_price=data.tqqq_price or 0,
        daily_loss_pct=data.daily_loss_pct,
        qqq_closes=closes,
        trading_days_fetched=data.trading_days_fetched,
        day_trades_remaining=_calc_day_trades_remaining(account),
        account_equity=account["equity"],
        cash_balance=account["cash"],
        consecutive_losing_days=state["consecutive_losing_days"],
        sqqq_price=data.sqqq_price or 0,
        sqqq_current_shares=sqqq_current_shares,
        has_tqqq_position=has_tqqq,
        knn_direction=knn_direction,
        knn_confidence=knn_confidence,
        knn_adjustment=knn_adjustment,
        knn_probabilities=knn_probabilities,
        xgb_direction=xgb_direction,
        xgb_confidence=xgb_confidence,
        xgb_adjustment=xgb_adjustment,
        xgb_probabilities=xgb_probabilities,
        model_direction=effective_model["direction"],
        model_confidence=effective_model["confidence"],
        model_adjustment=effective_model["adjustment"],
        model_source=effective_model["source"],
        model_disagreement=effective_model["disagreement"],
    )


def _is_execution_window(is_half_day: bool = False) -> bool:
    """Check if current time is within the execution window (±10 min of target)."""
    now = datetime.now(ET)
    if is_half_day:
        target = LEVERAGE_CONFIG["execution_time_halfday"]
    else:
        target = LEVERAGE_CONFIG["execution_time_normal"]
    hour, minute = map(int, target.split(":"))
    target_minutes = hour * 60 + minute
    now_minutes = now.hour * 60 + now.minute
    return abs(now_minutes - target_minutes) <= 10


def _build_gate_data(signals: "StrategySignals", is_half_day: bool = False) -> dict:
    """Build the gate checklist input dict from computed signals."""
    return {
        "regime": signals.effective_regime,
        "qqq_close": signals.qqq_close,
        "sma_50": signals.sma_50,
        "sma_250": signals.sma_250,
        "momentum_score": signals.momentum_score,
        "realized_vol": signals.realized_vol,
        "vol_regime": signals.vol_regime,
        "daily_loss_pct": signals.daily_loss_pct,
        "qqq_closes": signals.qqq_closes,
        "holding_days_losing": signals.consecutive_losing_days,
        "is_execution_window": _is_execution_window(is_half_day),
        "allocated_capital": signals.allocated_capital,
        "day_trades_remaining": signals.day_trades_remaining,
        "options_flow_bearish": signals.options_flow_bearish,
        "options_flow_ratio": signals.options_flow_ratio,
        "trading_days_fetched": signals.trading_days_fetched,
        "current_shares": signals.current_shares,
        "knn_direction": signals.model_direction,
        "knn_confidence": signals.model_confidence,
    }


def _build_sizing_data(signals: "StrategySignals") -> dict:
    """Build the sizing calculation input dict from computed signals."""
    return {
        "regime": signals.effective_regime,
        "allocated_capital": signals.allocated_capital,
        "momentum_score": signals.momentum_score,
        "vol_regime": signals.vol_regime,
        "options_flow_adjustment": signals.options_flow_adjustment,
        "qqq_close": signals.qqq_close,
        "sma_50": signals.sma_50,
        "sma_250": signals.sma_250,
        "qqq_closes": signals.qqq_closes,
        "tqqq_price": signals.tqqq_price,
        "current_shares": signals.current_shares,
        "model_direction": signals.model_direction,
        "model_confidence": signals.model_confidence,
        "model_disagreement": signals.model_disagreement,
        "knn_adjustment": signals.model_adjustment,
        "daily_loss_pct": signals.daily_loss_pct,
    }


def _build_sqqq_gate_data(signals: "StrategySignals", is_half_day: bool = False, tqqq_just_exited: bool = False) -> dict:
    """Build the SQQQ gate checklist input dict from computed signals."""
    return {
        "knn_direction": signals.model_direction,
        "knn_confidence": signals.model_confidence,
        "vol_regime": signals.vol_regime,
        "allocated_capital": signals.allocated_capital,
        "is_execution_window": _is_execution_window(is_half_day),
        "day_trades_remaining": signals.day_trades_remaining,
        "trading_days_fetched": signals.trading_days_fetched,
        "has_tqqq_position": signals.has_tqqq_position,
        "regime": signals.effective_regime,
        "tqqq_just_exited": tqqq_just_exited,
        "qqq_close": signals.qqq_close,
        "sma_50": signals.sma_50,
        "sma_250": signals.sma_250,
        "qqq_closes": signals.qqq_closes,
        "daily_loss_pct": signals.daily_loss_pct,
        "model_direction": signals.model_direction,
        "model_confidence": signals.model_confidence,
        "model_disagreement": signals.model_disagreement,
        # Trend override fields
        "pct_above_sma50": signals.pct_above_sma50,
        "roc_slow": signals.momentum.get("roc_slow", 0),
        "raw_knn_direction": signals.knn_direction,
        "raw_knn_confidence": signals.knn_confidence,
        "raw_xgb_direction": signals.xgb_direction,
        "raw_xgb_confidence": signals.xgb_confidence,
    }


def _build_sqqq_sizing_data(signals: "StrategySignals") -> dict:
    """Build the SQQQ sizing calculation input dict from computed signals."""
    return {
        "knn_direction": signals.model_direction,
        "knn_confidence": signals.model_confidence,
        "vol_regime": signals.vol_regime,
        "allocated_capital": signals.allocated_capital,
        "sqqq_price": signals.sqqq_price,
        "current_shares": signals.sqqq_current_shares,
        "regime": signals.effective_regime,
        "qqq_close": signals.qqq_close,
        "sma_50": signals.sma_50,
        "sma_250": signals.sma_250,
        "qqq_closes": signals.qqq_closes,
        "daily_loss_pct": signals.daily_loss_pct,
        "model_direction": signals.model_direction,
        "model_confidence": signals.model_confidence,
        "model_disagreement": signals.model_disagreement,
        "pct_above_sma50": signals.pct_above_sma50,
        "roc_slow": signals.momentum.get("roc_slow", 0),
        "raw_knn_direction": signals.knn_direction,
        "raw_knn_confidence": signals.knn_confidence,
        "raw_xgb_direction": signals.xgb_direction,
        "raw_xgb_confidence": signals.xgb_confidence,
    }


def cmd_morning():
    """Morning position check (9:35 AM ET). Exits only — no entries."""
    from strategy.position_manager import PositionManager
    from db.models import get_connection, init_tables
    import alpaca_client
    import notifications

    logger.info(f"{'='*50}")
    logger.info(f"Leveraged ETF Strategy - MORNING CHECK - {_timestamp()}")
    logger.info(f"{'='*50}")

    if not LEVERAGE_CONFIG.get("pm_enabled", True):
        logger.info("Position manager disabled, skipping morning check")
        return

    conn = get_connection()
    try:
        init_tables(conn)

        # Check market is open
        today_str = datetime.now(ET).date().isoformat()
        calendar = alpaca_client.get_calendar(today_str)
        if calendar is None:
            logger.info("Market closed today, skipping morning check")
            return

        pm = PositionManager(alpaca_client=alpaca_client, config=LEVERAGE_CONFIG)
        decisions = pm.run_morning_check(conn)

        if decisions:
            actions = [f"{d.exit_type}: {d.reason}" for d in decisions if d.should_exit]
            logger.info(f"Morning check: {len(actions)} actions taken")
            for a in actions:
                logger.info(f"  {a}")
        else:
            logger.info("Morning check: no actions needed")

        # Print status dashboard
        try:
            positions = alpaca_client.get_positions()
            managed = {LEVERAGE_CONFIG["bull_etf"], LEVERAGE_CONFIG["bear_etf"]}
            pos_data = []
            for p in positions:
                if p["symbol"] in managed:
                    snapshots = alpaca_client.get_snapshot([p["symbol"]])
                    snap = snapshots.get(p["symbol"], {})
                    prev_close = snap.get("prev_daily_bar_close", 0) or 0
                    open_price = snap.get("daily_bar_open", 0) or 0
                    gap_pct = ((open_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    stop_level = p["avg_entry_price"] * (1 - LEVERAGE_CONFIG.get("pm_stop_loss_pct", 0.08))
                    pos_data.append({
                        "symbol": p["symbol"],
                        "shares": p["qty"],
                        "pnl_pct": p["unrealized_plpc"] * 100,
                        "gap_pct": gap_pct,
                        "stop_level": stop_level,
                    })
            if pos_data:
                notifications.send_morning_summary({
                    "positions": pos_data,
                    "actions": [f"{d.exit_type}: {d.reason}" for d in decisions if d.should_exit],
                })
        except Exception as e:
            logger.warning(f"Failed to send morning summary: {e}")

    finally:
        conn.close()


def cmd_midday():
    """Midday position check (12:30 PM ET). Exits only — no entries."""
    from strategy.position_manager import PositionManager
    from db.models import get_connection, init_tables
    import alpaca_client

    logger.info(f"{'='*50}")
    logger.info(f"Leveraged ETF Strategy - MIDDAY CHECK - {_timestamp()}")
    logger.info(f"{'='*50}")

    if not LEVERAGE_CONFIG.get("pm_enabled", True):
        logger.info("Position manager disabled, skipping midday check")
        return

    conn = get_connection()
    try:
        init_tables(conn)

        # Check market is open
        today_str = datetime.now(ET).date().isoformat()
        calendar = alpaca_client.get_calendar(today_str)
        if calendar is None:
            logger.info("Market closed today, skipping midday check")
            return

        pm = PositionManager(alpaca_client=alpaca_client, config=LEVERAGE_CONFIG)
        decisions = pm.run_midday_check(conn)

        if decisions:
            actions = [f"{d.exit_type}: {d.reason}" for d in decisions if d.should_exit]
            logger.info(f"Midday check: {len(actions)} actions taken")
            for a in actions:
                logger.info(f"  {a}")
        else:
            logger.info("Midday check: no actions needed")
    finally:
        conn.close()


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
            # Interruptible sleep: check every 10s instead of blocking for 5 min
            for _ in range(poll_interval // 10):
                time.sleep(10)

        try:
            flow = uw_client.get_tqqq_flow()
            flow_samples.append(flow)
            logger.info(
                f"  Poll {i+1}/{poll_count}: P/C ratio={flow['ratio']:.2f} "
                f"puts=${flow['put_premium']:,.0f} calls=${flow['call_premium']:,.0f} "
                f"alerts={flow['alert_count']}"
            )
        except Exception as e:
            logger.warning(f"  Poll {i+1}/{poll_count} failed: {e}")
            continue

    # Aggregate flow data
    if not flow_samples:
        logger.warning("All flow polls failed, using neutral defaults")
        total_put, total_call, avg_ratio, bearish_count = 0, 0, 1.0, 0
    else:
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


def _already_traded_today(conn=None, symbol: str | None = None) -> bool:
    """Check if we already executed a trade today (prevents double-buy from halfday + main run)."""
    from db.models import get_db
    today_str = datetime.now(ET).date().isoformat()
    with get_db(conn) as c:
        if symbol:
            row = c.execute(
                "SELECT COUNT(*) as cnt FROM decisions WHERE date = ? AND status = 'EXECUTED' AND symbol = ?",
                [today_str, symbol],
            ).fetchone()
        else:
            row = c.execute(
                "SELECT COUNT(*) as cnt FROM decisions WHERE date = ? AND status = 'EXECUTED'",
                [today_str],
            ).fetchone()
        return row["cnt"] > 0 if row else False


def cmd_run(halfday_check: bool = False):
    """Execute the daily strategy pipeline."""
    from strategy.sizing import (
        run_gate_checklist,
        calculate_target_shares,
        run_sqqq_gate_checklist,
        calculate_sqqq_target_shares,
        is_sqqq_regime_allowed,
        is_sqqq_signal_active,
    )
    from strategy.executor import execute_rebalance
    from db.models import (
        get_connection,
        init_tables,
        log_daily_decision,
        update_decision,
        log_regime_change,
        log_daily_performance,
        get_today_pregame,
        get_latest_performance,
        get_first_performance,
    )
    import alpaca_client
    import notifications

    logger.info(f"{'='*50}")
    logger.info(f"Leveraged ETF Strategy - RUN - {_timestamp()}")
    logger.info(f"{'='*50}")

    conn = get_connection()
    try:
        init_tables(conn)

        # Fetch data
        data = _fetch_all_data(conn=conn)

        # Half-day check: if --halfday-check and not a half day, exit
        if halfday_check and not data.is_half_day:
            logger.info("Not a half day, exiting (--halfday-check mode)")
            return
        if data.is_half_day:
            logger.info("Half day detected!")
            notifications.send_halfday_alert()

        # Check market is open
        if data.calendar is None:
            logger.info("Market closed today, exiting")
            return

        # Verify data quality
        if data.trading_days_fetched < 250:
            msg = f"Only {data.trading_days_fetched} trading days, need 250 for SMA"
            logger.error(msg)
            notifications.send_error("Insufficient Data", msg)
            return

        # Data staleness check: last bar should be from today or previous trading day
        if data.qqq_bars:
            last_bar_date = date.fromisoformat(data.qqq_bars[-1]["date"])
            today = datetime.now(ET).date()
            days_stale = (today - last_bar_date).days
            if days_stale > 3:  # Allow weekends (2 days) + 1 buffer
                msg = f"Data is stale: last bar is {data.qqq_bars[-1]['date']} ({days_stale} days old)"
                logger.error(msg)
                notifications.send_error("Stale Data", msg)
                return

        # Update high watermark if holding managed positions
        if LEVERAGE_CONFIG.get("pm_enabled", True):
            try:
                from db.models import get_position_watermark, update_position_watermark
                for sym in [LEVERAGE_CONFIG["bull_etf"], LEVERAGE_CONFIG["bear_etf"]]:
                    pos = data.tqqq_position if sym == LEVERAGE_CONFIG["bull_etf"] else data.sqqq_position
                    if pos and pos["qty"] > 0:
                        price = pos["current_price"]
                        wm = get_position_watermark(sym, conn)
                        if wm is None or price > wm["high_price"]:
                            update_position_watermark(sym, price, datetime.now(ET).date().isoformat(), conn)
            except Exception as e:
                logger.warning(f"Watermark update failed: {e}")

        # Compute signals
        signals = _compute_signals(data, conn=conn)

        # Check for pregame intel
        pregame = get_today_pregame(conn)
        if pregame:
            logger.info(f"Pregame intel: sentiment={pregame['pregame_sentiment']}, "
                         f"flow_ratio={pregame['flow_avg_ratio']:.2f}, "
                         f"notes={pregame['pregame_notes']}")
            # Override flow data with pregame aggregated data if available
            if pregame["flow_samples"] and pregame["flow_samples"] > 0:
                signals.options_flow_ratio = pregame["flow_avg_ratio"]
                signals.options_flow_bearish = pregame["flow_bearish_samples"] >= 3
                if signals.options_flow_bearish:
                    signals.options_flow_adjustment = 1.0 - LEVERAGE_CONFIG["options_flow_reduction_pct"]
        else:
            logger.info("No pregame intel available for today")

        # Run gate checklist
        gates_passed, gates_failed = run_gate_checklist(_build_gate_data(signals, data.is_half_day))

        # Calculate target position
        target = calculate_target_shares(_build_sizing_data(signals))

        logger.info(f"Regime: {signals.effective_regime} | Momentum: {signals.momentum_score:.2f}")
        logger.info(f"Vol: {signals.realized_vol:.1f}% ({signals.vol_regime})")
        logger.info(f"Gates: {'PASS' if gates_passed else 'FAIL'} ({gates_failed})")
        logger.info(f"Target: {target['target_shares']} shares ({target['action']})")

        # Execute
        is_emergency = signals.effective_regime in ("RISK_OFF", "BREAKDOWN")
        is_exit_or_reduction = target["action"] in ("EXIT", "SELL")
        should_trade = gates_passed or is_emergency or is_exit_or_reduction

        # Stale position force-exit: if gates have failed for N consecutive days while holding
        stale_max = LEVERAGE_CONFIG.get("stale_position_max_days", 5)
        if (not should_trade
                and signals.current_shares > 0
                and not is_emergency):
            from db.models import get_consecutive_gate_failures
            consecutive_failures = get_consecutive_gate_failures(conn)
            if consecutive_failures >= stale_max:
                logger.warning(
                    f"STALE POSITION: gates failed {consecutive_failures} consecutive days "
                    f"while holding {signals.current_shares} shares. Forcing exit."
                )
                is_emergency = True
                should_trade = True
                target["action"] = "EXIT"
                target["target_shares"] = 0
                target["delta_shares"] = -signals.current_shares
                target["delta_value"] = -signals.current_shares * signals.tqqq_price

        # Direct bull->inverse rotation is disabled by default under a controlling regime.
        wants_rotation_to_sqqq = (
            LEVERAGE_CONFIG.get("use_sqqq_trading", False)
            and LEVERAGE_CONFIG.get("allow_tqqq_to_sqqq_rotation", False)
            and signals.has_tqqq_position
            and signals.knn_direction == "SHORT"
            and signals.knn_confidence >= LEVERAGE_CONFIG.get("sqqq_min_knn_confidence", 0.60)
            and is_sqqq_regime_allowed(signals.effective_regime)
        )
        if wants_rotation_to_sqqq:
            logger.info(f"ROTATION: k-NN SHORT (conf={signals.knn_confidence:.2f}) — "
                         f"will exit TQQQ to rotate to SQQQ")
            should_trade = True
            target["action"] = "EXIT"
            target["target_shares"] = 0
            target["delta_shares"] = -signals.current_shares
            target["delta_value"] = -signals.current_shares * signals.tqqq_price

        # Dedup: skip if we already executed a buy today (prevents halfday + main run double-buy)
        # Only dedup buys — sells/exits should always go through
        if (should_trade and not is_emergency and not is_exit_or_reduction
                and not wants_rotation_to_sqqq
                and target["action"] == "BUY" and _already_traded_today(conn)):
            logger.info("Already traded today, skipping execution (dedup)")
            should_trade = False

        will_trade = should_trade and target["action"] != "HOLD"

        # Log decision BEFORE execution (intent-first for crash safety)
        today_et = datetime.now(ET).date().isoformat()
        decision = {
            "date": today_et,
            "timestamp": _timestamp(),
            "qqq_close": signals.qqq_close,
            "qqq_sma_50": signals.sma_50,
            "qqq_sma_250": signals.sma_250,
            "qqq_pct_above_sma50": signals.pct_above_sma50,
            "qqq_pct_above_sma250": signals.pct_above_sma250,
            "qqq_roc_5": signals.momentum["roc_fast"],
            "qqq_roc_20": signals.momentum["roc_slow"],
            "realized_vol_20d": signals.realized_vol,
            "vol_regime": signals.vol_regime,
            "options_flow_put_premium": signals.flow.get("put_premium", 0),
            "options_flow_call_premium": signals.flow.get("call_premium", 0),
            "options_flow_ratio": signals.options_flow_ratio,
            "options_flow_bearish": 1 if signals.options_flow_bearish else 0,
            "options_flow_adjustment": signals.options_flow_adjustment,
            "regime": signals.effective_regime,
            "regime_changed": 1 if signals.regime_changed else 0,
            "previous_regime": signals.previous_regime,
            "momentum_score": signals.momentum_score,
            "momentum_factor": signals.momentum["raw_score"],
            "gates_passed": 1 if gates_passed else 0,
            "gates_failed": gates_failed,
            "target_allocation_pct": target["target_allocation_pct"],
            "target_dollar_value": target["target_value"],
            "target_shares": target["target_shares"],
            "current_shares": signals.current_shares,
            "order_action": target["action"],
            "order_shares": 0,
            "order_value": 0,
            "order_id": None,
            "fill_price": None,
            "fill_time": None,
            "execution_window": "HALFDAY" if data.is_half_day else "NORMAL",
            "account_equity": signals.account_equity,
            "allocated_capital": signals.allocated_capital,
            "tqqq_position_value": data.tqqq_position["market_value"] if data.tqqq_position else 0,
            "tqqq_pnl_pct": data.tqqq_position["unrealized_plpc"] * 100 if data.tqqq_position else 0,
            "other_positions_value": signals.capital["other_positions_value"],
            "cash_balance": signals.cash_balance,
            "day_trades_remaining": signals.day_trades_remaining,
            "trading_days_fetched": signals.trading_days_fetched,
            "is_half_day": 1 if data.is_half_day else 0,
            "knn_direction": signals.knn_direction,
            "knn_confidence": round(signals.knn_confidence, 4),
            "knn_adjustment": signals.knn_adjustment,
            "xgb_direction": signals.xgb_direction,
            "xgb_confidence": round(signals.xgb_confidence, 4),
            "xgb_adjustment": signals.xgb_adjustment,
            "symbol": "TQQQ",
            "sqqq_position_value": data.sqqq_position["market_value"] if data.sqqq_position else 0,
            "sqqq_pnl_pct": data.sqqq_position["unrealized_plpc"] * 100 if data.sqqq_position else 0,
            "status": "PENDING" if will_trade else "COMPLETE",
        }
        decision_id = log_daily_decision(decision, conn)

        # Execute trade and update decision with results
        execution_result = {"executed": False, "action": "HOLD", "order": None, "reason": "Gates failed"}
        sweep_presold = False

        if will_trade:
            try:
                # Sells/exits bypass PDT — selling shares bought on a prior day isn't a day trade
                is_selling = target["action"] in ("EXIT", "SELL")
                # Free cash from the T-bill sweep before a buy so the order can't bounce
                if not is_selling and target.get("delta_value", 0) > 0:
                    from strategy.cash_sweep import free_cash_for_buy
                    presell = free_cash_for_buy(
                        alpaca_client,
                        target["delta_value"],
                        signals.capital.get("cash_available", 0),
                    )
                    sweep_presold = presell["sold"]
                    if sweep_presold:
                        logger.info(f"Cash sweep: {presell['reason']}")
                execution_result = execute_rebalance(
                    target["target_shares"],
                    signals.current_shares,
                    signals.tqqq_price,
                    alpaca_client,
                    is_emergency=is_emergency or is_selling,
                    day_trades_remaining=signals.day_trades_remaining,
                )
                logger.info(f"Execution: {execution_result['reason']}")
                order = execution_result.get("order") or {}
                update_decision(decision_id, {
                    "status": "EXECUTED" if execution_result.get("executed") else "FAILED",
                    "order_action": execution_result.get("action", target["action"]),
                    "order_shares": execution_result.get("shares", 0),
                    "order_value": execution_result.get("value", 0),
                    "order_id": order.get("order_id"),
                    "fill_price": order.get("filled_avg_price"),
                }, conn)
            except Exception as e:
                logger.error(f"Trade execution failed: {e}")
                update_decision(decision_id, {"status": "FAILED"}, conn)
                raise
        elif target["action"] == "HOLD":
            execution_result = {"executed": False, "action": "HOLD", "order": None, "reason": "At target"}

        # ── SQQQ trading logic (when enabled) ──
        sqqq_action = "HOLD"
        sqqq_order_shares = 0
        sqqq_execution_result = {"executed": False, "action": "HOLD"}

        # Rotation remains an explicit opt-in policy; default path never bypasses mutual exclusivity.
        tqqq_just_exited = wants_rotation_to_sqqq and execution_result.get("executed", False)

        if LEVERAGE_CONFIG.get("use_sqqq_trading", False):
            sqqq_gate_data = _build_sqqq_gate_data(signals, data.is_half_day, tqqq_just_exited=tqqq_just_exited)
            sqqq_gates_passed, sqqq_gates_failed = run_sqqq_gate_checklist(sqqq_gate_data)
            sqqq_target = calculate_sqqq_target_shares(_build_sqqq_sizing_data(signals))
            logger.info(f"SQQQ gates: {'PASS' if sqqq_gates_passed else 'FAIL'} ({sqqq_gates_failed})")
            logger.info(f"SQQQ target: {sqqq_target['target_shares']} shares ({sqqq_target['action']})")

            regime_allows_sqqq = is_sqqq_regime_allowed(signals.effective_regime)
            signal_supports_sqqq = is_sqqq_signal_active(sqqq_gate_data)

            # Phase 1: SQQQ EXIT (runs first, before any TQQQ entry)
            # Exit if the bearish thesis no longer holds, inverse regime is disallowed, or the long sleeve is valid.
            if signals.sqqq_current_shares > 0:
                should_exit_sqqq = (
                    not regime_allows_sqqq
                    or not signal_supports_sqqq
                    or gates_passed  # TQQQ gates pass → rotate to TQQQ
                )
                if should_exit_sqqq:
                    logger.info("Exiting SQQQ position")
                    sqqq_execution_result = execute_rebalance(
                        0, signals.sqqq_current_shares, signals.sqqq_price,
                        alpaca_client, is_emergency=True,
                        day_trades_remaining=signals.day_trades_remaining,
                        symbol="SQQQ",
                    )
                    sqqq_action = "EXIT"
                    sqqq_order_shares = signals.sqqq_current_shares
                    logger.info(f"SQQQ exit: {sqqq_execution_result['reason']}")

            # Phase 2: SQQQ ENTRY (runs last, after TQQQ execution)
            # Enter only when inverse regime is valid and no long sleeve is being opened this cycle.
            elif (sqqq_gates_passed
                  and (not signals.has_tqqq_position or tqqq_just_exited)
                  and not (will_trade and target["action"] == "BUY")
                  and sqqq_target["action"] == "BUY"
                  and not _already_traded_today(conn, symbol="SQQQ")):
                # Decrement day trades remaining after TQQQ exit used one
                sqqq_day_trades = signals.day_trades_remaining
                if tqqq_just_exited:
                    sqqq_day_trades = max(0, sqqq_day_trades - 1)
                logger.info("Entering SQQQ position")
                sqqq_execution_result = execute_rebalance(
                    sqqq_target["target_shares"], 0, signals.sqqq_price,
                    alpaca_client, day_trades_remaining=sqqq_day_trades,
                    symbol="SQQQ",
                )
                sqqq_action = sqqq_target["action"]
                sqqq_order_shares = sqqq_target["target_shares"]
                logger.info(f"SQQQ entry: {sqqq_execution_result['reason']}")

            # Log SQQQ decision if action taken
            if sqqq_action != "HOLD":
                sqqq_decision = {
                    "date": today_et,
                    "timestamp": _timestamp(),
                    "qqq_close": signals.qqq_close,
                    "qqq_sma_50": signals.sma_50,
                    "qqq_sma_250": signals.sma_250,
                    "regime": signals.effective_regime,
                    "knn_direction": signals.knn_direction,
                    "knn_confidence": round(signals.knn_confidence, 4),
                    "symbol": "SQQQ",
                    "order_action": sqqq_action,
                    "order_shares": sqqq_execution_result.get("shares", sqqq_order_shares),
                    "order_value": sqqq_execution_result.get("value", 0),
                    "account_equity": signals.account_equity,
                    "allocated_capital": signals.allocated_capital,
                    "sqqq_position_value": data.sqqq_position["market_value"] if data.sqqq_position else 0,
                    "sqqq_pnl_pct": data.sqqq_position["unrealized_plpc"] * 100 if data.sqqq_position else 0,
                    "status": "EXECUTED" if sqqq_execution_result.get("executed") else "FAILED",
                }
                sqqq_order = sqqq_execution_result.get("order") or {}
                sqqq_decision["order_id"] = sqqq_order.get("order_id")
                sqqq_decision["fill_price"] = sqqq_order.get("filled_avg_price")
                log_daily_decision(sqqq_decision, conn)

        # ── Cash-yield sweep: park idle allocated capital in T-bills (non-fatal) ──
        if LEVERAGE_CONFIG.get("use_cash_sweep", False):
            try:
                from strategy.cash_sweep import execute_sweep
                fresh_positions = alpaca_client.get_positions()
                strat_value = sum(
                    abs(p.get("market_value", 0)) for p in fresh_positions
                    if p["symbol"] in (LEVERAGE_CONFIG["bull_etf"], LEVERAGE_CONFIG["bear_etf"])
                )
                # If a TQQQ buy just executed, floor at target value in case the
                # fill hasn't reflected in the positions API yet
                if execution_result.get("executed") and target["action"] == "BUY":
                    strat_value = max(strat_value, target["target_shares"] * signals.tqqq_price)
                sweep_result = execute_sweep(
                    alpaca_client,
                    signals.allocated_capital,
                    strat_value,
                    skip_buy=sweep_presold,
                )
                if sweep_result.get("executed"):
                    logger.info(f"Cash sweep: {sweep_result['reason']}")
            except Exception as e:
                logger.warning(f"Cash sweep failed (non-fatal): {e}")

        # Log regime change
        if signals.regime_changed and signals.previous_regime:
            regime_data = {
                "date": datetime.now(ET).date().isoformat(),
                "qqq_close": signals.qqq_close,
                "qqq_sma_50": signals.sma_50,
                "qqq_sma_250": signals.sma_250,
                "trigger_reason": f"Detected: {signals.raw_regime}",
            }
            log_regime_change(signals.previous_regime, signals.effective_regime, regime_data, conn)

            # Send regime alert
            action_desc = execution_result.get("reason", target["action"])
            if is_emergency and signals.current_shares > 0:
                action_desc = f"Selling all {signals.current_shares} shares TQQQ"
            alert_data = {
                **regime_data,
                "realized_vol_20d": signals.realized_vol,
                "vol_regime": signals.vol_regime,
                "options_flow_ratio": signals.options_flow_ratio,
                "options_flow_bearish": signals.options_flow_bearish,
                "action_description": action_desc,
            }
            notifications.send_regime_alert(signals.previous_regime, signals.effective_regime, alert_data)

        # Log performance
        tqqq_val = data.tqqq_position["market_value"] if data.tqqq_position else 0
        tqqq_pnl = data.tqqq_position["unrealized_pl"] if data.tqqq_position else 0
        sqqq_val = data.sqqq_position["market_value"] if data.sqqq_position else 0
        strategy_equity = signals.account_equity - signals.capital["other_positions_value"]
        prev_perf = get_latest_performance(conn)
        first_perf = get_first_performance(conn)
        strategy_pnl_day = strategy_equity - prev_perf["strategy_equity"] if prev_perf else 0.0
        strategy_anchor = first_perf["strategy_equity"] if first_perf and first_perf.get("strategy_equity") else strategy_equity
        strategy_total_return_pct = (
            ((strategy_equity / strategy_anchor) - 1) * 100
            if strategy_anchor > 0 else 0.0
        )

        benchmark_qqq_pct = 0.0
        try:
            benchmark_anchor_row = conn.execute(
                "SELECT qqq_close FROM decisions WHERE symbol='TQQQ' ORDER BY id ASC LIMIT 1"
            ).fetchone()
            benchmark_anchor = benchmark_anchor_row["qqq_close"] if benchmark_anchor_row else None
            if benchmark_anchor and benchmark_anchor > 0:
                benchmark_qqq_pct = ((signals.qqq_close / benchmark_anchor) - 1) * 100
        except Exception:
            benchmark_qqq_pct = 0.0

        perf = {
            "date": today_et,
            "account_equity": signals.account_equity,
            "other_positions_value": signals.capital["other_positions_value"],
            "strategy_equity": strategy_equity,
            "strategy_pnl_day": strategy_pnl_day,
            "tqqq_shares": signals.current_shares,
            "tqqq_avg_cost": data.tqqq_position["avg_entry_price"] if data.tqqq_position else 0,
            "tqqq_current_price": signals.tqqq_price,
            "tqqq_position_value": tqqq_val,
            "tqqq_pnl_day": tqqq_pnl,
            "tqqq_pnl_total": tqqq_pnl,
            "tqqq_pnl_pct": data.tqqq_position["unrealized_plpc"] * 100 if data.tqqq_position else 0,
            "regime": signals.effective_regime,
            "allocated_capital": signals.allocated_capital,
            "realized_vol": signals.realized_vol,
            "benchmark_qqq_pct": benchmark_qqq_pct,
            "strategy_total_return_pct": strategy_total_return_pct,
            "sqqq_shares": signals.sqqq_current_shares,
            "sqqq_position_value": sqqq_val,
            "sqqq_pnl_pct": data.sqqq_position["unrealized_plpc"] * 100 if data.sqqq_position else 0,
        }
        log_daily_performance(perf, conn)

        # Compute QQQ benchmark return (since position entry)
        qqq_benchmark_pct = 0.0
        if data.tqqq_position and data.tqqq_position["qty"] > 0:
            from db.models import get_position_entry_date
            entry_date = get_position_entry_date(conn)
            if entry_date and data.qqq_bars:
                entry_bars = [b for b in data.qqq_bars if b["date"] >= entry_date]
                if len(entry_bars) >= 2:
                    qqq_entry_price = entry_bars[0]["close"]
                    if qqq_entry_price > 0:
                        qqq_benchmark_pct = ((signals.qqq_close - qqq_entry_price) / qqq_entry_price) * 100

        # Send daily report
        report_data = {
            "regime": signals.effective_regime,
            "regime_days": signals.regime_hold_days,
            "qqq_close": signals.qqq_close,
            "qqq_sma_50": signals.sma_50,
            "qqq_sma_250": signals.sma_250,
            "qqq_pct_above_sma50": signals.pct_above_sma50,
            "qqq_pct_above_sma250": signals.pct_above_sma250,
            "momentum_score": signals.momentum_score,
            "realized_vol_20d": signals.realized_vol,
            "vol_regime": signals.vol_regime,
            "options_flow_ratio": signals.options_flow_ratio,
            "options_flow_bearish": signals.options_flow_bearish,
            "trading_days_fetched": signals.trading_days_fetched,
            "gates_passed": 16 - len(gates_failed),
            "qqq_benchmark_pct": round(qqq_benchmark_pct, 2),
            "gates_failed_list": gates_failed,
            "day_trades_remaining": signals.day_trades_remaining,
            "order_action": execution_result.get("action", target["action"]),
            "order_shares": execution_result.get("shares", abs(target["delta_shares"])),
            "target_dollar_value": target["target_value"],
            "allocated_capital": signals.allocated_capital,
            "current_shares": signals.current_shares,
            "tqqq_position_value": tqqq_val,
            "tqqq_pnl_pct": perf["tqqq_pnl_pct"],
            "strategy_equity": round(strategy_equity, 2),
            "strategy_pnl_day": round(strategy_pnl_day, 2),
            "strategy_total_return_pct": round(strategy_total_return_pct, 2),
            "knn_direction": signals.knn_direction,
            "knn_confidence": signals.knn_confidence,
            "knn_adjustment": signals.knn_adjustment,
            "knn_probabilities": signals.knn_probabilities,
            "xgb_direction": signals.xgb_direction,
            "xgb_confidence": signals.xgb_confidence,
            "xgb_adjustment": signals.xgb_adjustment,
            "xgb_probabilities": signals.xgb_probabilities,
            "model_direction": signals.model_direction,
            "model_confidence": signals.model_confidence,
            "model_adjustment": signals.model_adjustment,
            "model_source": signals.model_source,
            "model_disagreement": signals.model_disagreement,
            "sqqq_current_shares": signals.sqqq_current_shares,
            "sqqq_position_value": sqqq_val,
            "sqqq_pnl_pct": perf["sqqq_pnl_pct"],
            "sqqq_action": sqqq_action,
            "sqqq_order_shares": sqqq_order_shares,
            "pct_above_sma50": signals.pct_above_sma50,
            "pct_above_sma250": signals.pct_above_sma250,
            "effective_regime": signals.effective_regime,
        }
        if pregame:
            report_data["pregame_sentiment"] = pregame["pregame_sentiment"]
            report_data["pregame_notes"] = pregame["pregame_notes"]
        notifications.send_daily_report(report_data)

        logger.info("Daily run complete")
    finally:
        conn.close()


def cmd_status():
    """Show current state without trading."""
    from strategy.signals import (
        calculate_momentum, calculate_realized_vol,
        classify_vol_regime,
    )
    from strategy.regime import detect_regime, get_effective_regime
    from strategy.sizing import get_allocated_capital, run_gate_checklist, calculate_target_shares, run_sqqq_gate_checklist, calculate_sqqq_target_shares
    from db.models import init_tables

    init_tables()
    data = _fetch_all_data()
    signals = _compute_signals(data)

    print(f"\n{'='*50}")
    print(f"  Leveraged ETF Strategy - STATUS")
    print(f"  {_timestamp()}")
    print(f"{'='*50}\n")

    # Regime
    print(f"Regime: {signals.effective_regime} (raw: {signals.raw_regime})")
    print(f"  Previous: {signals.previous_regime or 'N/A'}")
    print(f"  Hold days: {signals.regime_hold_days}")

    # QQQ
    print(f"\nQQQ: ${signals.qqq_close:.2f}")
    print(f"  SMA-50:  ${signals.sma_50:.2f} ({signals.pct_above_sma50:+.1f}%)")
    print(f"  SMA-250: ${signals.sma_250:.2f} ({signals.pct_above_sma250:+.1f}%)")
    print(f"  Trading days loaded: {signals.trading_days_fetched}")

    # Momentum
    mom = signals.momentum
    print(f"\nMomentum: {signals.momentum_score:.2f}")
    print(f"  ROC-5: {mom['roc_fast']:+.4f}")
    print(f"  ROC-20: {mom['roc_slow']:+.4f}")

    # Vol
    print(f"\nRealized Vol: {signals.realized_vol:.1f}% ({signals.vol_regime})")

    # Flow
    flow = signals.flow
    print(f"\nOptions Flow: P/C ratio = {flow['ratio']:.2f} ({'Bearish' if flow['is_bearish'] else 'Neutral'})")
    print(f"  Put premium: ${flow['put_premium']:,.0f}")
    print(f"  Call premium: ${flow['call_premium']:,.0f}")
    print(f"  Alerts: {flow['alert_count']}")
    if flow.get("error"):
        print(f"  Warning: {flow['error']}")

    # k-NN Signal
    print(f"\nk-NN Signal: {signals.knn_direction} (conf={signals.knn_confidence:.2f}, adj={signals.knn_adjustment})")
    if signals.knn_direction != "FLAT":
        probs = signals.knn_probabilities
        print(f"  P(down)={probs[0]:.2f}  P(up)={probs[1]:.2f}")
    report_only = LEVERAGE_CONFIG.get("knn_report_only", True)
    print(f"  Mode: {'REPORT-ONLY' if report_only else 'ACTIVE'}")

    # XGBoost Signal
    print(f"\nXGBoost Signal: {signals.xgb_direction} (conf={signals.xgb_confidence:.2f}, adj={signals.xgb_adjustment})")
    if signals.xgb_direction != "FLAT":
        xgb_probs = signals.xgb_probabilities
        print(f"  P(down)={xgb_probs[0]:.2f}  P(up)={xgb_probs[1]:.2f}")

    # Effective model
    print(f"\nEffective Model: {signals.model_direction} (conf={signals.model_confidence:.2f}, adj={signals.model_adjustment})")
    print(f"  Source: {signals.model_source}")
    print(f"  Disagreement: {'Yes' if signals.model_disagreement else 'No'}")

    # Position
    tqqq = data.tqqq_position
    if tqqq:
        print(f"\nTQQQ Position: {tqqq['qty']} shares @ ${tqqq['avg_entry_price']:.2f}")
        print(f"  Market value: ${tqqq['market_value']:,.2f}")
        print(f"  P/L: ${tqqq['unrealized_pl']:,.2f} ({tqqq['unrealized_plpc']*100:+.1f}%)")
    else:
        print(f"\nTQQQ Position: None")

    # SQQQ position
    sqqq = data.sqqq_position
    if sqqq:
        print(f"SQQQ Position: {sqqq['qty']} shares @ ${sqqq['avg_entry_price']:.2f}")
        print(f"  Market value: ${sqqq['market_value']:,.2f}")
        print(f"  P/L: ${sqqq['unrealized_pl']:,.2f} ({sqqq['unrealized_plpc']*100:+.1f}%)")
    elif LEVERAGE_CONFIG.get("use_sqqq_trading", False):
        print(f"SQQQ Position: None")

    # Capital
    cap = signals.capital
    print(f"\nCapital Allocation:")
    print(f"  Total equity: ${cap['total_equity']:,.2f}")
    print(f"  Other positions: ${cap['other_positions_value']:,.2f}")
    print(f"  Allocated (30% cap): ${cap['allocated_capital']:,.2f}")
    print(f"  Cash available: ${cap['cash_available']:,.2f}")

    # PDT
    print(f"\nPDT: {signals.day_trades_remaining} day trades used")

    # Gate checklist
    gates_passed, gates_failed = run_gate_checklist(_build_gate_data(signals))

    print(f"\nTQQQ Gate Checklist: {'PASS' if gates_passed else 'FAIL'}")
    if gates_failed:
        for g in gates_failed:
            print(f"  FAIL: {g}")
    else:
        print(f"  All 16 gates passed")

    # SQQQ gate checklist (informational)
    sqqq_gates_passed, sqqq_gates_failed = run_sqqq_gate_checklist(_build_sqqq_gate_data(signals))
    sqqq_enabled = LEVERAGE_CONFIG.get("use_sqqq_trading", False)
    sqqq_label = "" if sqqq_enabled else " (DISABLED)"
    print(f"\nSQQQ Gate Checklist{sqqq_label}: {'PASS' if sqqq_gates_passed else 'FAIL'}")
    if sqqq_gates_failed:
        for g in sqqq_gates_failed:
            print(f"  FAIL: {g}")
    else:
        print(f"  All 9 gates passed")

    # What-if TQQQ
    target = calculate_target_shares(_build_sizing_data(signals))

    print(f"\nTQQQ What-if (no execution):")
    print(f"  Target: {target['target_shares']} shares (${target['target_value']:,.0f})")
    print(f"  Current: {target['current_shares']} shares")
    print(f"  Delta: {target['delta_shares']:+d} shares (${target['delta_value']:+,.0f})")
    print(f"  Action: {target['action']}")
    if target["limiting_factors"]:
        print(f"  Limiting: {', '.join(target['limiting_factors'])}")

    # What-if SQQQ
    sqqq_target = calculate_sqqq_target_shares(_build_sqqq_sizing_data(signals))
    print(f"\nSQQQ What-if (no execution):")
    print(f"  Target: {sqqq_target['target_shares']} shares (${sqqq_target['target_value']:,.0f})")
    print(f"  Current: {sqqq_target['current_shares']} shares")
    print(f"  Delta: {sqqq_target['delta_shares']:+d} shares (${sqqq_target['delta_value']:+,.0f})")
    print(f"  Action: {sqqq_target['action']}")
    if sqqq_target["limiting_factors"]:
        print(f"  Limiting: {', '.join(sqqq_target['limiting_factors'])}")

    print(f"\nHalf day: {'Yes' if data.is_half_day else 'No'}")
    print(f"{'='*50}\n")


def cmd_backtest():
    """Run historical backtest simulation."""
    import alpaca_client
    from db.models import get_connection, init_tables
    from db.cache import get_bars_with_cache
    from strategy.regime import detect_regime
    from strategy.signals import calculate_momentum, calculate_realized_vol, classify_vol_regime
    from strategy.sizing import (
        run_gate_checklist,
        calculate_target_shares,
        run_sqqq_gate_checklist,
        calculate_sqqq_target_shares,
    )
    import notifications
    from sklearn.metrics import roc_auc_score

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
    sqqq_bars = get_bars_with_cache(
        "SQQQ", 900, alpaca_client.fetch_bars_for_cache, conn
    )

    logger.info(f"QQQ bars: {len(qqq_bars)}, TQQQ bars: {len(tqqq_bars)}, SQQQ bars: {len(sqqq_bars)}")

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
    sqqq_by_date = {b["date"]: b for b in sqqq_bars}
    common_dates = sorted(set(qqq_by_date.keys()) & set(tqqq_by_date.keys()) & set(sqqq_by_date.keys()))

    if len(common_dates) < 260:
        logger.error(f"Only {len(common_dates)} common dates, need 260+")
        return

    # Model validation setup (walk-forward: train on first 300, predict on rest)
    knn_model = None
    xgb_model = None
    knn_bars = [qqq_by_date[dt] for dt in common_dates]  # chronological bars
    knn_train_cutoff = 300
    knn_stats = {
        "correct": 0, "total": 0,
        "long_correct": 0, "long_total": 0,
        "short_correct": 0, "short_total": 0,
        "flat_count": 0,
        "regime_agree": 0, "regime_disagree": 0,
        "disagree_knn_right": 0, "disagree_knn_wrong": 0,
        "scores": [], "labels": [],
    }
    xgb_stats = {
        "correct": 0, "total": 0,
        "long_correct": 0, "long_total": 0,
        "short_correct": 0, "short_total": 0,
        "flat_count": 0,
        "scores": [], "labels": [],
    }

    # Fetch VIX data, cross-asset bars, and microstructure for k-NN/XGBoost features
    vix_by_date = {}
    cross_asset_bars = {}
    microstructure_by_date = {}
    if LEVERAGE_CONFIG.get("use_knn_signal", False):
        try:
            from strategy.vix_data import get_vix_data
            vix_by_date = get_vix_data()
        except Exception as e:
            logger.warning(f"VIX data fetch failed for backtest, using defaults: {e}")
        for sym in ("TLT", "GLD", "IWM"):
            try:
                cross_asset_bars[sym] = get_bars_with_cache(
                    sym, 900, alpaca_client.fetch_bars_for_cache, conn
                )
            except Exception as e:
                logger.warning(f"Cross-asset bar fetch failed for {sym}: {e}")
        if LEVERAGE_CONFIG.get("use_microstructure", False):
            try:
                from db.cache import get_microstructure_with_cache
                microstructure_by_date = get_microstructure_with_cache(
                    "QQQ", 900, alpaca_client.get_intraday_bars, conn
                )
            except Exception as e:
                logger.warning(f"Microstructure fetch failed for backtest: {e}")

    if LEVERAGE_CONFIG.get("use_knn_signal", False) and len(knn_bars) > knn_train_cutoff + 50:
        from strategy.knn_signal import KNNSignal, FeatureCalculator
        from strategy.xgb_signal import XGBSignal
        knn_model = KNNSignal(n_neighbors=LEVERAGE_CONFIG.get("knn_neighbors", 7))
        xgb_model = XGBSignal(
            n_estimators=LEVERAGE_CONFIG.get("xgb_n_estimators", 200),
            max_depth=LEVERAGE_CONFIG.get("xgb_max_depth", 4),
            learning_rate=LEVERAGE_CONFIG.get("xgb_learning_rate", 0.05),
        )
        if not knn_model.fit_from_bars(knn_bars[:knn_train_cutoff + 1], vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date):
            logger.warning("k-NN training failed for backtest validation")
            knn_model = None
        else:
            logger.info(f"k-NN trained on {knn_model.training_samples} samples for validation")
        if not xgb_model.fit_from_bars(knn_bars[:knn_train_cutoff + 1], vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date):
            logger.warning("XGBoost training failed for backtest validation")
            xgb_model = None

    # Simulation
    from strategy.position_manager import PositionManager

    initial_capital = 100000.0
    cash = initial_capital
    tqqq_shares = 0
    sqqq_shares = 0
    prev_regime = None
    regime_change_day = 0
    peak_value = initial_capital
    max_drawdown = 0.0
    num_trades = 0
    days_in_market = 0
    prev_close_equity = initial_capital
    pm = PositionManager(config=LEVERAGE_CONFIG)
    position_meta = {
        "TQQQ": {
            "avg_entry_price": 0.0,
            "entry_date": None,
            "high_watermark": 0.0,
            "tiers_taken": [],
            "gate_fail_streak": 0,
        },
        "SQQQ": {
            "avg_entry_price": 0.0,
            "entry_date": None,
            "high_watermark": 0.0,
            "tiers_taken": [],
            "gate_fail_streak": 0,
        },
    }

    results = []
    qqq_start_price = qqq_by_date[common_dates[250]]["close"]  # Start after warmup
    tqqq_start_price = tqqq_by_date[common_dates[250]]["close"]

    for i, dt in enumerate(common_dates):
        if i < 250:  # Warmup period for SMA-250
            continue

        qqq_bar = qqq_by_date[dt]
        tqqq_bar = tqqq_by_date[dt]
        sqqq_bar = sqqq_by_date[dt]
        prev_dt = common_dates[i - 1] if i > 0 else None
        prev_tqqq_bar = tqqq_by_date[prev_dt] if prev_dt else None
        prev_sqqq_bar = sqqq_by_date[prev_dt] if prev_dt else None
        had_exposure = tqqq_shares > 0 or sqqq_shares > 0
        tqqq_just_exited = False

        # Build closes array
        warmup_dates = common_dates[max(0, i - 260):i + 1]
        closes = [qqq_by_date[d]["close"] for d in warmup_dates if d in qqq_by_date]
        prior_dates = common_dates[max(0, i - 260):i]
        prior_closes = [qqq_by_date[d]["close"] for d in prior_dates if d in qqq_by_date]

        if len(closes) < 250:
            continue
        if len(prior_closes) < 250:
            continue

        qqq_close = closes[-1]
        sma_50 = float(np.mean(closes[-50:]))
        sma_250 = float(np.mean(closes[-250:]))
        sma_250_prev = float(np.mean(prior_closes[-250:]))

        # Morning risk checks at the open.
        morning_tqqq, morning_tqqq_price = _run_backtest_morning_window(
            pm, "TQQQ", tqqq_shares, position_meta["TQQQ"], tqqq_bar, prev_tqqq_bar,
            qqq_bar, prev_close_equity, sma_250_prev, dt,
        )
        if morning_tqqq and tqqq_shares > 0:
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "TQQQ", -morning_tqqq.shares_to_sell, morning_tqqq_price, dt, cash,
                share_map, position_meta,
            )
            if traded:
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
                had_exposure = True
                tqqq_just_exited = True

        morning_sqqq, morning_sqqq_price = _run_backtest_morning_window(
            pm, "SQQQ", sqqq_shares, position_meta["SQQQ"], sqqq_bar, prev_sqqq_bar,
            qqq_bar, prev_close_equity, sma_250_prev, dt,
        )
        if morning_sqqq and sqqq_shares > 0:
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "SQQQ", -morning_sqqq.shares_to_sell, morning_sqqq_price, dt, cash,
                share_map, position_meta,
            )
            if traded:
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
                had_exposure = True

        # Midday risk and profit-taking checks.
        midday_tqqq, midday_tqqq_price, _ = _run_backtest_midday_window(
            pm, "TQQQ", tqqq_shares, position_meta["TQQQ"], tqqq_bar, prev_tqqq_bar,
            prev_tqqq_bar["volume"] if prev_tqqq_bar else 0, prev_close_equity, dt,
        )
        if midday_tqqq and tqqq_shares > 0:
            sell_shares = midday_tqqq.shares_to_sell or tqqq_shares
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "TQQQ", -sell_shares, midday_tqqq_price, dt, cash, share_map, position_meta,
            )
            if traded:
                if midday_tqqq.exit_type == "PARTIAL_PROFIT":
                    favorable_state = _build_backtest_position_state(
                        "TQQQ", tqqq_shares, position_meta["TQQQ"], tqqq_bar,
                        prev_tqqq_bar, tqqq_bar["high"], dt,
                    )
                    gain_pct = ((favorable_state.current_price - favorable_state.avg_entry_price) / favorable_state.avg_entry_price) * 100 if favorable_state.avg_entry_price > 0 else 0
                    for tier in LEVERAGE_CONFIG.get("pm_profit_tiers", []):
                        if gain_pct >= tier["threshold_pct"] and tier["threshold_pct"] not in position_meta["TQQQ"]["tiers_taken"]:
                            position_meta["TQQQ"]["tiers_taken"].append(tier["threshold_pct"])
                            break
                elif share_map["TQQQ"] == 0:
                    tqqq_just_exited = True
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
                had_exposure = True

        midday_sqqq, midday_sqqq_price, _ = _run_backtest_midday_window(
            pm, "SQQQ", sqqq_shares, position_meta["SQQQ"], sqqq_bar, prev_sqqq_bar,
            prev_sqqq_bar["volume"] if prev_sqqq_bar else 0, prev_close_equity, dt,
        )
        if midday_sqqq and sqqq_shares > 0:
            sell_shares = midday_sqqq.shares_to_sell or sqqq_shares
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "SQQQ", -sell_shares, midday_sqqq_price, dt, cash, share_map, position_meta,
            )
            if traded:
                if midday_sqqq.exit_type == "PARTIAL_PROFIT":
                    favorable_state = _build_backtest_position_state(
                        "SQQQ", sqqq_shares, position_meta["SQQQ"], sqqq_bar,
                        prev_sqqq_bar, sqqq_bar["high"], dt,
                    )
                    gain_pct = ((favorable_state.current_price - favorable_state.avg_entry_price) / favorable_state.avg_entry_price) * 100 if favorable_state.avg_entry_price > 0 else 0
                    for tier in LEVERAGE_CONFIG.get("pm_profit_tiers", []):
                        if gain_pct >= tier["threshold_pct"] and tier["threshold_pct"] not in position_meta["SQQQ"]["tiers_taken"]:
                            position_meta["SQQQ"]["tiers_taken"].append(tier["threshold_pct"])
                            break
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
                had_exposure = True

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

        knn_dir = "FLAT"
        knn_conf = 0.5
        knn_adj = 1.0
        xgb_dir = "FLAT"
        xgb_conf = 0.5
        xgb_adj = 1.0

        # Walk-forward model predictions and validation
        if knn_model is not None and i >= knn_train_cutoff and i + 1 < len(common_dates):
            knn_result = knn_model.predict(knn_bars[:i + 1], vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
            knn_dir = knn_result["direction"]
            knn_conf = knn_result["confidence"]
            knn_adj = knn_result["adjustment"]
            next_dt = common_dates[i + 1]
            actual_up = qqq_by_date[next_dt]["close"] > qqq_close
            knn_stats["scores"].append(knn_result["probabilities"][1])
            knn_stats["labels"].append(1 if actual_up else 0)
            if knn_dir == "FLAT":
                knn_stats["flat_count"] += 1
            else:
                predicted_up = knn_dir == "LONG"
                knn_stats["total"] += 1
                if predicted_up == actual_up:
                    knn_stats["correct"] += 1
                if knn_dir == "LONG":
                    knn_stats["long_total"] += 1
                    if actual_up:
                        knn_stats["long_correct"] += 1
                else:
                    knn_stats["short_total"] += 1
                    if not actual_up:
                        knn_stats["short_correct"] += 1
                regime_bullish = regime in ("STRONG_BULL", "BULL")
                knn_bullish = knn_dir == "LONG"
                if regime_bullish == knn_bullish:
                    knn_stats["regime_agree"] += 1
                else:
                    knn_stats["regime_disagree"] += 1
                    if predicted_up == actual_up:
                        knn_stats["disagree_knn_right"] += 1
                    else:
                        knn_stats["disagree_knn_wrong"] += 1

        if xgb_model is not None and i >= knn_train_cutoff and i + 1 < len(common_dates):
            xgb_result = xgb_model.predict(knn_bars[:i + 1], vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
            xgb_dir = xgb_result["direction"]
            xgb_conf = xgb_result["confidence"]
            xgb_adj = xgb_result["adjustment"]
            next_dt = common_dates[i + 1]
            actual_up = qqq_by_date[next_dt]["close"] > qqq_close
            xgb_stats["scores"].append(xgb_result["probabilities"][1])
            xgb_stats["labels"].append(1 if actual_up else 0)
            if xgb_dir == "FLAT":
                xgb_stats["flat_count"] += 1
            else:
                predicted_up = xgb_dir == "LONG"
                xgb_stats["total"] += 1
                if predicted_up == actual_up:
                    xgb_stats["correct"] += 1
                if xgb_dir == "LONG":
                    xgb_stats["long_total"] += 1
                    if actual_up:
                        xgb_stats["long_correct"] += 1
                else:
                    xgb_stats["short_total"] += 1
                    if not actual_up:
                        xgb_stats["short_correct"] += 1

        effective_model = _resolve_model_signal(knn_dir, knn_conf, knn_adj, xgb_dir, xgb_conf, xgb_adj)

        # Signals
        mom = calculate_momentum(closes)
        vol = calculate_realized_vol(closes)
        vol_regime = classify_vol_regime(vol)
        portfolio_value = _backtest_portfolio_value(cash, tqqq_shares, sqqq_shares, tqqq_bar["close"], sqqq_bar["close"])
        allocated = portfolio_value * LEVERAGE_CONFIG["max_portfolio_pct"]

        gate_data = {
            "regime": regime,
            "qqq_close": qqq_close,
            "sma_50": sma_50,
            "sma_250": sma_250,
            "momentum_score": mom["score"],
            "realized_vol": vol,
            "vol_regime": vol_regime,
            "daily_loss_pct": 0.0,
            "qqq_closes": closes,
            "holding_days_losing": 0,
            "is_execution_window": True,
            "allocated_capital": allocated,
            "day_trades_remaining": 999,
            "options_flow_bearish": False,
            "options_flow_ratio": 1.0,
            "trading_days_fetched": len(closes),
            "current_shares": tqqq_shares,
            "knn_direction": effective_model["direction"],
            "knn_confidence": effective_model["confidence"],
        }
        tqqq_gates_passed, _ = run_gate_checklist(gate_data)
        tqqq_target = calculate_target_shares({
            "regime": regime,
            "allocated_capital": allocated,
            "momentum_score": mom["score"],
            "vol_regime": vol_regime,
            "options_flow_adjustment": 1.0,
            "qqq_close": qqq_close,
            "sma_50": sma_50,
            "qqq_closes": closes,
            "tqqq_price": tqqq_bar["close"],
            "current_shares": tqqq_shares,
            "model_direction": effective_model["direction"],
            "model_confidence": effective_model["confidence"],
            "model_disagreement": effective_model["disagreement"],
            "knn_adjustment": effective_model["adjustment"],
        })
        is_emergency = regime in ("RISK_OFF", "BREAKDOWN")
        is_exit_or_reduction = tqqq_target["action"] in ("EXIT", "SELL")
        should_trade_tqqq = tqqq_gates_passed or is_emergency or is_exit_or_reduction

        stale_max = LEVERAGE_CONFIG.get("stale_position_max_days", 5)
        if tqqq_shares > 0 and not should_trade_tqqq and not is_emergency:
            position_meta["TQQQ"]["gate_fail_streak"] += 1
            if position_meta["TQQQ"]["gate_fail_streak"] >= stale_max:
                tqqq_target["action"] = "EXIT"
                tqqq_target["target_shares"] = 0
                tqqq_target["delta_shares"] = -tqqq_shares
                tqqq_target["delta_value"] = -tqqq_shares * tqqq_bar["close"]
                should_trade_tqqq = True
        else:
            position_meta["TQQQ"]["gate_fail_streak"] = 0

        sqqq_gate_data = {
            "knn_direction": effective_model["direction"],
            "knn_confidence": effective_model["confidence"],
            "vol_regime": vol_regime,
            "allocated_capital": allocated,
            "is_execution_window": True,
            "day_trades_remaining": 999,
            "trading_days_fetched": len(closes),
            "has_tqqq_position": tqqq_shares > 0,
            "regime": regime,
            "tqqq_just_exited": tqqq_just_exited,
            "pct_above_sma50": ((qqq_close / sma_50) - 1) if sma_50 > 0 else 0.0,
            "roc_slow": mom["roc_slow"],
            "raw_knn_direction": knn_dir,
            "raw_knn_confidence": knn_conf,
            "raw_xgb_direction": xgb_dir,
            "raw_xgb_confidence": xgb_conf,
        }
        sqqq_gates_passed, _ = run_sqqq_gate_checklist(sqqq_gate_data)
        sqqq_target = calculate_sqqq_target_shares({
            "knn_direction": effective_model["direction"],
            "knn_confidence": effective_model["confidence"],
            "vol_regime": vol_regime,
            "allocated_capital": allocated,
            "sqqq_price": sqqq_bar["close"],
            "current_shares": sqqq_shares,
            "regime": regime,
            "pct_above_sma50": ((qqq_close / sma_50) - 1) if sma_50 > 0 else 0.0,
            "roc_slow": mom["roc_slow"],
            "raw_knn_direction": knn_dir,
            "raw_knn_confidence": knn_conf,
            "raw_xgb_direction": xgb_dir,
            "raw_xgb_confidence": xgb_conf,
        })

        # Mutual exclusivity: if long sleeve is valid, flatten inverse first.
        if sqqq_shares > 0 and (tqqq_gates_passed or sqqq_target["target_shares"] == 0):
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "SQQQ", -sqqq_shares, sqqq_bar["close"], dt, cash, share_map, position_meta,
            )
            if traded:
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
                had_exposure = True

        # Execute TQQQ rebalance at close.
        tqqq_delta = tqqq_target["target_shares"] - tqqq_shares
        if tqqq_delta > 0 and tqqq_target["action"] == "BUY":
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "TQQQ", tqqq_delta, tqqq_bar["close"], dt, cash, share_map, position_meta,
            )
            if traded:
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
        elif tqqq_delta < 0 and tqqq_target["action"] in ("SELL", "EXIT"):
            share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
            cash, traded = _apply_backtest_trade(
                "TQQQ", tqqq_delta, tqqq_bar["close"], dt, cash, share_map, position_meta,
            )
            if traded:
                tqqq_shares = share_map["TQQQ"]
                sqqq_shares = share_map["SQQQ"]
                num_trades += 1
                if tqqq_shares == 0:
                    tqqq_just_exited = True

        # Enter/resize SQQQ only when long sleeve is not active.
        if tqqq_shares == 0:
            sqqq_delta = sqqq_target["target_shares"] - sqqq_shares
            if sqqq_delta > 0 and sqqq_gates_passed and sqqq_target["action"] == "BUY":
                share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
                cash, traded = _apply_backtest_trade(
                    "SQQQ", sqqq_delta, sqqq_bar["close"], dt, cash, share_map, position_meta,
                )
                if traded:
                    tqqq_shares = share_map["TQQQ"]
                    sqqq_shares = share_map["SQQQ"]
                    num_trades += 1
            elif sqqq_delta < 0 and sqqq_target["action"] in ("SELL", "EXIT"):
                share_map = {"TQQQ": tqqq_shares, "SQQQ": sqqq_shares}
                cash, traded = _apply_backtest_trade(
                    "SQQQ", sqqq_delta, sqqq_bar["close"], dt, cash, share_map, position_meta,
                )
                if traded:
                    tqqq_shares = share_map["TQQQ"]
                    sqqq_shares = share_map["SQQQ"]
                    num_trades += 1

        portfolio_value = _backtest_portfolio_value(cash, tqqq_shares, sqqq_shares, tqqq_bar["close"], sqqq_bar["close"])
        peak_value = max(peak_value, portfolio_value)
        drawdown = (peak_value - portfolio_value) / peak_value * 100
        max_drawdown = max(max_drawdown, drawdown)

        if had_exposure or tqqq_shares > 0 or sqqq_shares > 0:
            days_in_market += 1

        # Benchmarks
        qqq_return = (qqq_bar["close"] / qqq_start_price - 1) * 100
        tqqq_return = (tqqq_bar["close"] / tqqq_start_price - 1) * 100
        strategy_return = (portfolio_value / initial_capital - 1) * 100
        pnl_day = portfolio_value - prev_close_equity

        results.append({
            "date": dt,
            "qqq_close": qqq_close,
            "tqqq_close": tqqq_bar["close"],
            "sqqq_close": sqqq_bar["close"],
            "regime": regime,
            "target_shares": tqqq_target["target_shares"],
            "held_shares": tqqq_shares,
            "sqqq_shares": sqqq_shares,
            "effective_model_direction": effective_model["direction"],
            "model_source": effective_model["source"],
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "pnl_day": round(pnl_day, 2),
            "pnl_total_pct": round(strategy_return, 2),
            "drawdown_pct": round(drawdown, 2),
            "qqq_buy_hold_pct": round(qqq_return, 2),
            "tqqq_buy_hold_pct": round(tqqq_return, 2),
        })
        prev_close_equity = portfolio_value

    # Write results to DB
    conn.executemany(
        "INSERT INTO backtest_results (date, qqq_close, tqqq_close, sqqq_close, regime, target_shares, "
        "held_shares, sqqq_shares, effective_model_direction, model_source, portfolio_value, cash, pnl_day, pnl_total_pct, drawdown_pct, "
        "qqq_buy_hold_pct, tqqq_buy_hold_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r["date"], r["qqq_close"], r["tqqq_close"], r["sqqq_close"], r["regime"], r["target_shares"],
          r["held_shares"], r["sqqq_shares"], r["effective_model_direction"], r["model_source"], r["portfolio_value"], r["cash"], r["pnl_day"],
          r["pnl_total_pct"], r["drawdown_pct"], r["qqq_buy_hold_pct"],
          r["tqqq_buy_hold_pct"]) for r in results],
    )
    conn.commit()

    # Export CSV
    from config import DATA_DIR as runtime_data_dir
    csv_path = runtime_data_dir / "backtest_results.csv"
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

    # k-NN validation summary
    if knn_model is not None and knn_stats["total"] > 0:
        ks = knn_stats
        accuracy = ks["correct"] / ks["total"] * 100
        long_acc = (ks["long_correct"] / ks["long_total"] * 100) if ks["long_total"] > 0 else 0
        short_acc = (ks["short_correct"] / ks["short_total"] * 100) if ks["short_total"] > 0 else 0
        agree_rate = ks["regime_agree"] / (ks["regime_agree"] + ks["regime_disagree"]) * 100 if (ks["regime_agree"] + ks["regime_disagree"]) > 0 else 0
        disagree_total = ks["disagree_knn_right"] + ks["disagree_knn_wrong"]
        disagree_acc = (ks["disagree_knn_right"] / disagree_total * 100) if disagree_total > 0 else 0

        print(f"{'='*50}")
        print(f"  k-NN SIGNAL VALIDATION")
        print(f"{'='*50}")
        print(f"  Training samples: {knn_model.training_samples}")
        print(f"  Test predictions: {ks['total']} (+ {ks['flat_count']} FLAT skipped)")
        print(f"\n  Overall accuracy: {accuracy:.1f}% ({ks['correct']}/{ks['total']})")
        print(f"    LONG:  {long_acc:.1f}% ({ks['long_correct']}/{ks['long_total']})")
        print(f"    SHORT: {short_acc:.1f}% ({ks['short_correct']}/{ks['short_total']})")
        print(f"\n  Regime agreement: {agree_rate:.1f}%")
        print(f"  Disagreements: {disagree_total}")
        if disagree_total > 0:
            print(f"    k-NN correct when disagreeing: {disagree_acc:.1f}% ({ks['disagree_knn_right']}/{disagree_total})")
        print(f"{'='*50}\n")

        stats["knn_accuracy"] = round(accuracy, 1)
        stats["knn_long_accuracy"] = round(long_acc, 1)
        stats["knn_short_accuracy"] = round(short_acc, 1)
        stats["knn_regime_agreement"] = round(agree_rate, 1)
        stats["knn_disagreement_accuracy"] = round(disagree_acc, 1)
        stats["knn_test_predictions"] = ks["total"]

    if xgb_model is not None and xgb_stats["total"] > 0:
        xs = xgb_stats
        xgb_accuracy = xs["correct"] / xs["total"] * 100
        xgb_long_acc = (xs["long_correct"] / xs["long_total"] * 100) if xs["long_total"] > 0 else 0
        xgb_short_acc = (xs["short_correct"] / xs["short_total"] * 100) if xs["short_total"] > 0 else 0
        xgb_auc = roc_auc_score(xs["labels"], xs["scores"]) if len(set(xs["labels"])) > 1 else 0.5
        print(f"{'='*50}")
        print(f"  XGBOOST SIGNAL VALIDATION")
        print(f"{'='*50}")
        print(f"  Test predictions: {xs['total']} (+ {xs['flat_count']} FLAT skipped)")
        print(f"  Overall accuracy: {xgb_accuracy:.1f}% ({xs['correct']}/{xs['total']})")
        print(f"    LONG:  {xgb_long_acc:.1f}% ({xs['long_correct']}/{xs['long_total']})")
        print(f"    SHORT: {xgb_short_acc:.1f}% ({xs['short_correct']}/{xs['short_total']})")
        print(f"  ROC AUC: {xgb_auc:.3f}")
        print(f"{'='*50}\n")
        stats["xgb_accuracy"] = round(xgb_accuracy, 1)
        stats["xgb_long_accuracy"] = round(xgb_long_acc, 1)
        stats["xgb_short_accuracy"] = round(xgb_short_acc, 1)
        stats["xgb_auc"] = round(xgb_auc, 3)
        stats["xgb_test_predictions"] = xs["total"]

    # Send to Telegram
    notifications.send_backtest_summary(stats, str(csv_path))

    conn.close()
    return stats


def cmd_force_exit():
    """Emergency sell all TQQQ and SQQQ."""
    from strategy.executor import force_exit
    import alpaca_client
    import notifications

    logger.info("FORCE EXIT initiated")

    # Exit TQQQ
    tqqq_result = force_exit(alpaca_client)
    if tqqq_result["executed"]:
        notifications.send_error(
            "FORCE EXIT EXECUTED",
            f"Sold {tqqq_result['shares_sold']} shares TQQQ\nOrder: {tqqq_result['order']}"
        )
    else:
        print(f"TQQQ force exit: {tqqq_result['reason']}")

    # Exit SQQQ
    sqqq_result = force_exit(alpaca_client, symbol="SQQQ")
    if sqqq_result["executed"]:
        notifications.send_error(
            "FORCE EXIT EXECUTED",
            f"Sold {sqqq_result['shares_sold']} shares SQQQ\nOrder: {sqqq_result['order']}"
        )
    else:
        print(f"SQQQ force exit: {sqqq_result['reason']}")

    return {"tqqq": tqqq_result, "sqqq": sqqq_result}


def main():
    if len(sys.argv) < 2:
        print("Usage: python job.py {run|morning|midday|pregame|status|backtest|force_exit}")
        print("  run [--halfday-check]  Execute daily strategy")
        print("  morning                Morning position check (9:35 AM ET)")
        print("  midday                 Midday position check (12:30 PM ET)")
        print("  pregame                Pre-execution intel gathering (3:30 PM)")
        print("  status                 Show current state (no trading)")
        print("  backtest               Run historical backtest")
        print("  force_exit             Emergency sell all TQQQ")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        halfday = "--halfday-check" in sys.argv
        cmd_run(halfday_check=halfday)
    elif command == "morning":
        cmd_morning()
    elif command == "midday":
        cmd_midday()
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
