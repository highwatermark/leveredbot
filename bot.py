#!/usr/bin/env python3
"""
Telegram bot for leveraged ETF strategy commands.

Uses raw httpx polling against the Bot API (no python-telegram-bot dependency).
Commands:
    /leverage         - Show current strategy status
    /leverageperf     - Show performance history
    /leverageregime   - Show regime history and transitions
    /leverageexit     - Force exit all positions (emergency)
    /leveragebacktest - Run backtest and send results
    /leveragevol      - Show current realized volatility breakdown
    /leverageflow     - Show TQQQ options flow sentiment
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import httpx
import pytz

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID

ET = pytz.timezone("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TIMEOUT = 15


def _send(chat_id: int | str, text: str) -> bool:
    """Send a message to a Telegram chat."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": ""},
            )
            resp.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return False


def _is_admin(chat_id: int | str) -> bool:
    return str(chat_id) == str(TELEGRAM_ADMIN_ID)


# ── Command handlers ──


def cmd_leverage(chat_id: int | str) -> None:
    """Show current strategy status."""
    from db.models import init_tables, get_strategy_state
    from strategy.signals import (
        calculate_momentum, calculate_realized_vol,
        classify_vol_regime,
    )
    from strategy.regime import detect_regime, get_effective_regime
    from strategy.sizing import get_allocated_capital

    init_tables()

    try:
        from job import _fetch_all_data, _compute_signals
        data = _fetch_all_data()
        signals = _compute_signals(data)
    except Exception as e:
        _send(chat_id, f"Error fetching data: {e}")
        return

    tqqq = data.tqqq_position
    pos_text = "None"
    if tqqq and tqqq["qty"] > 0:
        pos_text = (
            f"{tqqq['qty']} shares @ ${tqqq['avg_entry_price']:.2f}\n"
            f"  Value: ${tqqq['market_value']:,.2f}\n"
            f"  P/L: {tqqq['unrealized_plpc']*100:+.1f}%"
        )

    sqqq = data.sqqq_position
    sqqq_text = ""
    if sqqq and sqqq["qty"] > 0:
        sqqq_text = (
            f"\nSQQQ: {sqqq['qty']} shares @ ${sqqq['avg_entry_price']:.2f}\n"
            f"  Value: ${sqqq['market_value']:,.2f}\n"
            f"  P/L: {sqqq['unrealized_plpc']*100:+.1f}%"
        )

    text = (
        f"Leveraged ETF Status\n"
        f"{'=' * 25}\n\n"
        f"Regime: {signals.effective_regime} (Day {signals.regime_hold_days})\n"
        f"QQQ: ${signals.qqq_close:.2f}\n"
        f"  SMA50: ${signals.sma_50:.2f} ({signals.pct_above_sma50:+.1f}%)\n"
        f"  SMA250: ${signals.sma_250:.2f} ({signals.pct_above_sma250:+.1f}%)\n\n"
        f"Momentum: {signals.momentum_score:.2f}\n"
        f"Vol: {signals.realized_vol:.1f}% ({signals.vol_regime})\n"
        f"k-NN: {signals.knn_direction} ({signals.knn_confidence:.0%} conf)\n\n"
        f"TQQQ: {pos_text}\n"
        f"{sqqq_text}\n"
        f"Allocated: ${signals.allocated_capital:,.0f}\n"
        f"PDT: {signals.day_trades_remaining} day trades used"
    )

    _send(chat_id, text)


def cmd_leverageperf(chat_id: int | str) -> None:
    """Show performance history."""
    from db.models import init_tables, get_performance_summary

    init_tables()
    perf = get_performance_summary(days=30)

    text = (
        f"Performance (Last {perf['days']} days)\n"
        f"{'=' * 25}\n\n"
        f"Total P/L: ${perf['total_pnl']:,.2f}\n"
        f"Avg Daily: ${perf['avg_daily_pnl']:,.2f}\n"
        f"Best Day: ${perf['best_day']:,.2f}\n"
        f"Worst Day: ${perf['worst_day']:,.2f}\n"
        f"Total Return: {perf.get('latest_total_return_pct', 0):+.1f}%"
    )

    _send(chat_id, text)


def cmd_leverageregime(chat_id: int | str) -> None:
    """Show regime history and transitions."""
    from db.models import init_tables, get_regime_history

    init_tables()
    history = get_regime_history(days=60)

    if not history:
        _send(chat_id, "No regime transitions recorded yet.")
        return

    text = f"Regime History (Last 60 days)\n{'=' * 25}\n\n"
    for r in history[:10]:  # Last 10 transitions
        text += f"{r['date']}: {r['old_regime']} -> {r['new_regime']}\n"
        text += f"  QQQ: ${r['qqq_close']:.2f}\n"

    _send(chat_id, text)


def cmd_leverageexit(chat_id: int | str) -> None:
    """Force exit all positions."""
    from job import cmd_force_exit

    _send(chat_id, "Initiating force exit...")
    try:
        result = cmd_force_exit()
        tqqq = result.get("tqqq", {})
        sqqq = result.get("sqqq", {})
        text = f"Force Exit Results\n{'=' * 25}\n\n"
        text += f"TQQQ: {'Sold ' + str(tqqq.get('shares_sold', 0)) + ' shares' if tqqq.get('executed') else tqqq.get('reason', 'No position')}\n"
        text += f"SQQQ: {'Sold ' + str(sqqq.get('shares_sold', 0)) + ' shares' if sqqq.get('executed') else sqqq.get('reason', 'No position')}"
        _send(chat_id, text)
    except Exception as e:
        _send(chat_id, f"Force exit failed: {e}")


def cmd_leveragebacktest(chat_id: int | str) -> None:
    """Run backtest and send results."""
    _send(chat_id, "Starting backtest (this may take a minute)...")
    try:
        from job import cmd_backtest
        stats = cmd_backtest()
        if stats:
            text = (
                f"Backtest Complete\n{'=' * 25}\n\n"
                f"Period: {stats.get('start_date')} to {stats.get('end_date')}\n"
                f"Strategy: {stats.get('total_return_pct', 0):+.1f}%\n"
                f"Max DD: {stats.get('max_drawdown_pct', 0):.1f}%\n"
                f"Trades: {stats.get('num_trades', 0)}\n\n"
                f"QQQ B&H: {stats.get('qqq_buy_hold_pct', 0):+.1f}%\n"
                f"TQQQ B&H: {stats.get('tqqq_buy_hold_pct', 0):+.1f}%"
            )
            _send(chat_id, text)
        else:
            _send(chat_id, "Backtest returned no results.")
    except Exception as e:
        _send(chat_id, f"Backtest failed: {e}")


def cmd_leveragevol(chat_id: int | str) -> None:
    """Show current realized volatility breakdown."""
    from db.models import init_tables
    from strategy.signals import calculate_realized_vol, classify_vol_regime, get_vol_adjustment

    init_tables()

    try:
        from job import _fetch_all_data
        data = _fetch_all_data()
    except Exception as e:
        _send(chat_id, f"Error fetching data: {e}")
        return

    closes = data.qqq_closes
    vol_20 = calculate_realized_vol(closes, window=20)
    vol_10 = calculate_realized_vol(closes, window=10) if len(closes) >= 11 else 0
    vol_5 = calculate_realized_vol(closes, window=5) if len(closes) >= 6 else 0
    regime = classify_vol_regime(vol_20)
    adj = get_vol_adjustment(regime)

    text = (
        f"Volatility Breakdown\n{'=' * 25}\n\n"
        f"20-day RV: {vol_20:.1f}% ({regime})\n"
        f"10-day RV: {vol_10:.1f}%\n"
        f" 5-day RV: {vol_5:.1f}%\n\n"
        f"Allocation multiplier: {adj:.0%}\n"
        f"Thresholds: <15 LOW | 15-25 NORMAL | 25-35 HIGH | >35 EXTREME"
    )

    _send(chat_id, text)


def cmd_leverageflow(chat_id: int | str) -> None:
    """Show TQQQ options flow sentiment."""
    from db.models import init_tables
    import uw_client
    from strategy.signals import check_options_flow

    init_tables()

    try:
        flow = uw_client.get_tqqq_flow()
    except Exception as e:
        _send(chat_id, f"Error fetching flow: {e}")
        return

    is_bearish, adj = check_options_flow(flow)
    label = "BEARISH" if is_bearish else "NEUTRAL"

    text = (
        f"TQQQ Options Flow\n{'=' * 25}\n\n"
        f"Sentiment: {label}\n"
        f"P/C Ratio: {flow['ratio']:.2f}\n"
        f"Put Premium: ${flow['put_premium']:,.0f}\n"
        f"Call Premium: ${flow['call_premium']:,.0f}\n"
        f"Alerts: {flow['alert_count']}\n"
        f"Adjustment: {adj:.0%}"
    )

    if flow.get("error"):
        text += f"\nWarning: {flow['error']}"

    _send(chat_id, text)


# ── Command dispatch ──

COMMANDS = {
    "/leverage": cmd_leverage,
    "/leverageperf": cmd_leverageperf,
    "/leverageregime": cmd_leverageregime,
    "/leverageexit": cmd_leverageexit,
    "/leveragebacktest": cmd_leveragebacktest,
    "/leveragevol": cmd_leveragevol,
    "/leverageflow": cmd_leverageflow,
}


def poll_loop():
    """Long-poll for Telegram updates and dispatch commands."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    logger.info("Bot started, polling for updates...")
    offset = 0

    while True:
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.get(
                    f"{TELEGRAM_API}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                resp.raise_for_status()
                data = resp.json()

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = message.get("chat", {}).get("id")

                if not chat_id:
                    continue

                if not _is_admin(chat_id):
                    _send(chat_id, "Unauthorized.")
                    continue

                # Extract command (strip @botname if present)
                cmd = text.split()[0].split("@")[0].lower() if text else ""

                if cmd in COMMANDS:
                    logger.info(f"Handling command: {cmd}")
                    try:
                        COMMANDS[cmd](chat_id)
                    except Exception as e:
                        logger.error(f"Command {cmd} failed: {e}")
                        _send(chat_id, f"Command failed: {e}")
                elif cmd.startswith("/leverage"):
                    _send(chat_id, f"Unknown command: {cmd}\nAvailable: {', '.join(COMMANDS.keys())}")

        except KeyboardInterrupt:
            logger.info("Bot stopped")
            break
        except Exception as e:
            logger.error(f"Poll error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    poll_loop()
