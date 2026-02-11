"""
Telegram notification formatting and sending.

Sends daily reports, regime alerts, half-day alerts, errors,
and backtest summaries. Uses httpx for direct Bot API calls.
"""

import logging
from pathlib import Path

import httpx

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID

logger = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TIMEOUT = 15


def _send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_ID:
        logger.warning("Telegram not configured, skipping notification")
        print(text)  # Print to stdout as fallback
        return False

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": TELEGRAM_ADMIN_ID,
                    "text": text,
                    "parse_mode": parse_mode,
                },
            )
            resp.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        print(f"[Telegram failed] {text}")
        return False


def _send_document(file_path: str, caption: str = "") -> bool:
    """Send a file via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_ID:
        logger.warning("Telegram not configured, skipping file send")
        return False

    try:
        with httpx.Client(timeout=30) as client:
            with open(file_path, "rb") as f:
                resp = client.post(
                    f"{TELEGRAM_API}/sendDocument",
                    data={"chat_id": TELEGRAM_ADMIN_ID, "caption": caption},
                    files={"document": (Path(file_path).name, f)},
                )
                resp.raise_for_status()
                return True
    except Exception as e:
        logger.error(f"Telegram file send failed: {e}")
        return False


def send_daily_report(data: dict) -> bool:
    """Send formatted daily strategy report."""
    regime = data.get("regime", "UNKNOWN")
    regime_emoji = {
        "STRONG_BULL": "\U0001f7e2", "BULL": "\U0001f7e2",
        "CAUTIOUS": "\U0001f7e1", "RISK_OFF": "\U0001f534",
        "BREAKDOWN": "\U0001f534",
    }.get(regime, "\u26aa")

    regime_days = data.get("regime_days", "?")
    qqq_close = data.get("qqq_close", 0)
    sma_50 = data.get("qqq_sma_50", 0)
    sma_250 = data.get("qqq_sma_250", 0)
    pct_sma50 = data.get("qqq_pct_above_sma50", 0)
    pct_sma250 = data.get("qqq_pct_above_sma250", 0)

    momentum = data.get("momentum_score", 0)
    mom_label = "Strong" if momentum > 0.6 else "Moderate" if momentum > 0.3 else "Weak"

    vol = data.get("realized_vol_20d", 0)
    vol_regime = data.get("vol_regime", "UNKNOWN")

    flow_ratio = data.get("options_flow_ratio", 0)
    flow_label = "Bearish" if data.get("options_flow_bearish") else "Neutral"

    trading_days = data.get("trading_days_fetched", 0)
    gates = data.get("gates_passed", 0)
    gates_total = gates + len(data.get("gates_failed_list", []))
    gate_str = f"\u2705 All {gates_total} gates passed" if not data.get("gates_failed_list") else f"\u274c {len(data.get('gates_failed_list', []))} gates failed"

    action = data.get("order_action", "HOLD")
    order_shares = data.get("order_shares", 0)
    target_val = data.get("target_dollar_value", 0)
    allocated = data.get("allocated_capital", 0)

    current_shares = data.get("current_shares", 0)
    tqqq_value = data.get("tqqq_position_value", 0)
    tqqq_pnl = data.get("tqqq_pnl_pct", 0)
    day_trades = data.get("day_trades_remaining", "?")

    text = (
        f"\U0001f4ca Leveraged ETF Daily Report\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
        f"Regime: {regime_emoji} {regime} (Day {regime_days})\n"
        f"QQQ: ${qqq_close:.2f}\n"
        f"  \u2514 SMA50: ${sma_50:.2f} ({pct_sma50:+.1f}% above)\n"
        f"  \u2514 SMA250: ${sma_250:.2f} ({pct_sma250:+.1f}% above)\n\n"
        f"Momentum: {momentum:.2f} ({mom_label})\n"
        f"Realized Vol: {vol:.1f}% ({vol_regime})\n"
        f"Options Flow: {flow_label} (P/C ratio: {flow_ratio:.1f})\n\n"
        f"Data Quality: \u2705 {trading_days} trading days loaded\n"
        f"Gate Check: {gate_str}\n"
        f"PDT Status: {day_trades} day trades remaining\n\n"
        f"Action: {action}"
    )

    if action in ("BUY", "SELL", "REBALANCE") and order_shares:
        side = "Buy" if action == "BUY" else "Sell"
        text += f" \u2192 {side} {abs(order_shares)} shares TQQQ\n"
    else:
        text += "\n"

    text += (
        f"Target: ${target_val:,.0f} of ${allocated:,.0f} allocated\n\n"
        f"Position: {current_shares} shares TQQQ (${tqqq_value:,.0f})\n"
        f"P/L: {tqqq_pnl:+.1f}%\n"
    )

    # QQQ benchmark comparison (when holding a position)
    qqq_bench = data.get("qqq_benchmark_pct")
    if qqq_bench is not None and current_shares > 0:
        vs_label = "outperforming" if tqqq_pnl > qqq_bench else "underperforming"
        text += f"Benchmark: QQQ {qqq_bench:+.1f}% ({vs_label})\n"

    # SQQQ position (if held)
    sqqq_shares = data.get("sqqq_current_shares", 0)
    if sqqq_shares > 0:
        sqqq_value = data.get("sqqq_position_value", 0)
        sqqq_pnl = data.get("sqqq_pnl_pct", 0)
        text += f"\nSQQQ Position: {sqqq_shares} shares (${sqqq_value:,.0f})\n"
        text += f"SQQQ P/L: {sqqq_pnl:+.1f}%\n"

    # SQQQ action (if taken)
    sqqq_action = data.get("sqqq_action")
    if sqqq_action and sqqq_action != "HOLD":
        sqqq_order = data.get("sqqq_order_shares", 0)
        side = "Buy" if sqqq_action == "BUY" else "Sell"
        text += f"SQQQ Action: {side} {abs(sqqq_order)} shares\n"

    # k-NN signal overlay (report-only)
    knn_dir = data.get("knn_direction", "FLAT")
    if knn_dir != "FLAT":
        knn_conf = data.get("knn_confidence", 0.5)
        knn_emoji = "\U0001f7e2" if knn_dir == "LONG" else "\U0001f534"
        text += f"\nk-NN Signal: {knn_emoji} {knn_dir} ({knn_conf:.0%} conf)\n"
    else:
        text += f"\nk-NN Signal: \u26aa FLAT (low conviction)\n"

    # Pregame intel if available
    if data.get("pregame_sentiment"):
        text += (
            f"\nPre-Game: {data['pregame_sentiment']}\n"
            f"  {data.get('pregame_notes', '')}\n"
        )

    text += f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"

    return _send_message(text, parse_mode="")


def send_regime_alert(old: str | None, new: str, data: dict) -> bool:
    """Send regime change alert."""
    qqq_close = data.get("qqq_close", 0)
    sma_250 = data.get("qqq_sma_250", 0)
    vol = data.get("realized_vol_20d", 0)
    vol_regime = data.get("vol_regime", "")
    flow_ratio = data.get("options_flow_ratio", 0)
    flow_label = "Bearish" if data.get("options_flow_bearish") else "Neutral"

    pct_sma250 = ((qqq_close / sma_250) - 1) * 100 if sma_250 else 0
    above_below = "above" if pct_sma250 >= 0 else "below"

    action = data.get("action_description", "No action")

    text = (
        f"\U0001f6a8 REGIME CHANGE ALERT\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"{old or 'NONE'} \u2192 {new}\n\n"
        f"QQQ: ${qqq_close:.2f} ({abs(pct_sma250):.1f}% {above_below} SMA250)\n"
        f"Realized Vol: {vol:.1f}% ({vol_regime})\n"
        f"Options Flow: {flow_label} (P/C ratio: {flow_ratio:.1f})\n\n"
        f"ACTION: {action}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
    )

    return _send_message(text, parse_mode="")


def send_halfday_alert() -> bool:
    """Send half-day detection alert."""
    text = (
        "\u23f0 HALF DAY DETECTED\n"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        "Market closes at 1:00 PM EST today\n"
        "Execution moved to 12:45 PM EST\n"
        "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
    )
    return _send_message(text, parse_mode="")


def send_error(title: str, detail: str) -> bool:
    """Send error notification."""
    text = f"\u26a0\ufe0f {title}\n\n{detail}"
    return _send_message(text, parse_mode="")


def send_position_exit_alert(event_data: dict) -> bool:
    """Send Telegram alert for a position exit event."""
    exit_type = event_data.get("exit_type", "UNKNOWN")
    emoji = {
        "STOP_LOSS": "\U0001f6d1",       # stop sign
        "TRAILING_STOP": "\U0001f4c9",    # chart decreasing
        "GAP_DOWN": "\u26a1",             # lightning
        "VOL_SPIKE": "\U0001f4a5",        # collision
        "REGIME_EMERGENCY": "\U0001f6a8", # rotating light
        "MAX_HOLD": "\u23f0",             # alarm clock
        "DAILY_LOSS_LIMIT": "\U0001f198", # SOS
        "PARTIAL_PROFIT": "\U0001f4b0",   # money bag
    }.get(exit_type, "\u26a0\ufe0f")

    urgency_label = " [URGENT]" if event_data.get("urgency") == "URGENT" else ""
    symbol = event_data.get("symbol", "???")
    shares = event_data.get("shares_sold", 0)
    price = event_data.get("price", 0)
    pnl = event_data.get("pnl_pct", 0)
    reason = event_data.get("reason", "")
    window = event_data.get("window", "")

    text = (
        f"{emoji} POSITION EXIT{urgency_label}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"Type: {exit_type}\n"
        f"Window: {window}\n"
        f"Symbol: {symbol}\n"
        f"Shares sold: {shares}\n"
        f"Price: ${price:.2f}\n"
        f"P/L: {pnl:+.1f}%\n\n"
        f"Reason: {reason}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
    )
    return _send_message(text, parse_mode="")


def send_morning_summary(data: dict) -> bool:
    """Send morning check summary: gap %, position state, stop levels, actions taken."""
    actions = data.get("actions", [])
    positions = data.get("positions", [])
    action_count = len(actions)

    text = (
        f"\U0001f305 Morning Check Summary\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
    )

    for pos in positions:
        symbol = pos.get("symbol", "???")
        shares = pos.get("shares", 0)
        pnl = pos.get("pnl_pct", 0)
        gap = pos.get("gap_pct", 0)
        stop_level = pos.get("stop_level", 0)
        text += (
            f"{symbol}: {shares} shares (P/L: {pnl:+.1f}%)\n"
            f"  Gap: {gap:+.1f}% | Stop: ${stop_level:.2f}\n"
        )

    if actions:
        text += f"\nActions taken: {action_count}\n"
        for a in actions:
            text += f"  {a}\n"
    else:
        text += "\nNo actions taken\n"

    text += f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"

    return _send_message(text, parse_mode="")


def send_backtest_summary(stats: dict, csv_path: str | None = None) -> bool:
    """Send backtest results summary and optionally the CSV file."""
    total_return = stats.get("total_return_pct", 0)
    max_dd = stats.get("max_drawdown_pct", 0)
    qqq_return = stats.get("qqq_buy_hold_pct", 0)
    tqqq_return = stats.get("tqqq_buy_hold_pct", 0)
    num_trades = stats.get("num_trades", 0)
    days_in_market = stats.get("days_in_market", 0)
    total_days = stats.get("total_days", 0)
    start_date = stats.get("start_date", "?")
    end_date = stats.get("end_date", "?")
    market_pct = f" ({days_in_market/total_days*100:.0f}%)" if total_days > 0 else ""

    text = (
        f"\U0001f4c8 Backtest Results ({start_date} to {end_date})\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
        f"Strategy Return: {total_return:+.1f}%\n"
        f"Max Drawdown: {max_dd:.1f}%\n"
        f"Trades: {num_trades}\n"
        f"Days in Market: {days_in_market}/{total_days}{market_pct}\n\n"
        f"Benchmarks:\n"
        f"  QQQ Buy & Hold: {qqq_return:+.1f}%\n"
        f"  TQQQ Buy & Hold: {tqqq_return:+.1f}%\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
    )

    result = _send_message(text, parse_mode="")

    if csv_path:
        _send_document(csv_path, caption="Backtest full results")

    return result
