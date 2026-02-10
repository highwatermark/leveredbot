"""
Windowed position manager for intraday defensive exits.

Runs morning (9:35 AM ET) and midday (12:30 PM ET) checks on held positions.
All entries happen at EOD (3:50 PM) — this module only does exits and partial profit-taking.

Exit types:
- STOP_LOSS: Position down >8% from entry
- TRAILING_STOP: Price dropped >6% from high watermark
- GAP_DOWN: Overnight gap >4%
- VOL_SPIKE: Volatility jumped >50% from prior day
- REGIME_EMERGENCY: QQQ fell >3% below SMA-250 intraday
- MAX_HOLD: Losing position held >15 days
- DAILY_LOSS_LIMIT: Account down >5% today (exit all)
- PARTIAL_PROFIT: Sell 25% at +8%, +15%, +25% tiers
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pytz

from config import LEVERAGE_CONFIG

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


@dataclass
class PositionState:
    """Snapshot of a held position with enriched context."""
    symbol: str
    shares: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl_pct: float

    # From DB
    entry_date: str | None = None
    holding_days: int = 0

    # From snapshot
    intraday_high: float = 0.0
    intraday_low: float = 0.0
    intraday_open: float = 0.0
    prev_close: float = 0.0

    # Computed
    overnight_gap_pct: float = 0.0
    intraday_change_pct: float = 0.0
    intraday_drawdown_pct: float = 0.0


@dataclass
class ExitDecision:
    """Result of an exit check."""
    should_exit: bool
    exit_type: str  # STOP_LOSS, TRAILING_STOP, GAP_DOWN, VOL_SPIKE, REGIME_EMERGENCY, MAX_HOLD, DAILY_LOSS_LIMIT, PARTIAL_PROFIT
    shares_to_sell: int = 0
    reason: str = ""
    urgency: str = "NORMAL"  # NORMAL or URGENT


class PositionManager:
    """Manages intraday position checks and defensive exits."""

    def __init__(self, alpaca_client=None, config: dict | None = None):
        self.alpaca_client = alpaca_client
        self.cfg = config or LEVERAGE_CONFIG

    # ── Core entry points ──

    def run_morning_check(self, conn) -> list[ExitDecision]:
        """Morning check (9:35 AM ET): gap-down, stop-loss, regime emergency, daily loss limit."""
        logger.info("Running morning position check")

        states = self._get_position_states(conn)
        if not states:
            logger.info("No positions held, morning check complete")
            return []

        decisions = []
        account = self.alpaca_client.get_account()
        account_equity = account["equity"]
        day_trades_remaining = account["daytrade_count"]

        # Daily loss limit check (applies to all positions)
        prev_equity = self._get_prev_equity(conn)
        daily_loss = self.check_daily_loss_limit(account_equity, prev_equity)
        if daily_loss.should_exit:
            for state in states:
                d = ExitDecision(
                    should_exit=True,
                    exit_type="DAILY_LOSS_LIMIT",
                    shares_to_sell=state.shares,
                    reason=daily_loss.reason,
                    urgency="URGENT",
                )
                decisions.append(d)
                self._execute_exit(d, state, conn, day_trades_remaining, "MORNING")
            return decisions

        # Per-position checks
        for state in states:
            candidates = []

            # Gap-down check
            candidates.append(self.check_gap_down(state))

            # Stop-loss check
            candidates.append(self.check_stop_loss(state))

            # Regime emergency check
            qqq_snap = self._get_qqq_snapshot()
            qqq_current = qqq_snap.get("latest_trade_price", 0) if qqq_snap else 0
            sma_250 = self._get_sma_250(conn)
            candidates.append(self.check_regime_emergency(state, qqq_current, sma_250))

            # Pick best decision
            best = self._select_decision([c for c in candidates if c.should_exit])
            if best:
                best.shares_to_sell = best.shares_to_sell or state.shares
                self._execute_exit(best, state, conn, day_trades_remaining, "MORNING")
                decisions.append(best)

        return decisions

    def run_midday_check(self, conn) -> list[ExitDecision]:
        """Midday check (12:30 PM ET): trailing stop, stop-loss, vol spike, partial profit, daily loss."""
        logger.info("Running midday position check")

        states = self._get_position_states(conn)
        if not states:
            logger.info("No positions held, midday check complete")
            return []

        decisions = []
        account = self.alpaca_client.get_account()
        account_equity = account["equity"]
        day_trades_remaining = account["daytrade_count"]

        # Daily loss limit check
        prev_equity = self._get_prev_equity(conn)
        daily_loss = self.check_daily_loss_limit(account_equity, prev_equity)
        if daily_loss.should_exit:
            for state in states:
                d = ExitDecision(
                    should_exit=True,
                    exit_type="DAILY_LOSS_LIMIT",
                    shares_to_sell=state.shares,
                    reason=daily_loss.reason,
                    urgency="URGENT",
                )
                decisions.append(d)
                self._execute_exit(d, state, conn, day_trades_remaining, "MIDDAY")
            return decisions

        # Per-position checks
        for state in states:
            candidates = []

            # Stop-loss
            candidates.append(self.check_stop_loss(state))

            # Trailing stop
            wm = self._get_high_watermark(state.symbol, conn)
            high_watermark = wm["high_price"] if wm else state.avg_entry_price
            # Update watermark if current price exceeds it
            if state.current_price > high_watermark:
                high_watermark = state.current_price
                self._update_high_watermark(state.symbol, state.current_price, conn)
            candidates.append(self.check_trailing_stop(state, high_watermark))

            # Vol spike
            snapshots = self.alpaca_client.get_snapshot([state.symbol])
            snap = snapshots.get(state.symbol, {})
            current_vol = snap.get("daily_bar_volume", 0)
            prev_vol = self._get_prev_volume(state.symbol, conn)
            candidates.append(self.check_vol_spike(state, current_vol, prev_vol))

            # Max hold period
            candidates.append(self.check_max_hold_period(state))

            # Full exit candidates
            full_exits = [c for c in candidates if c.should_exit]

            # Partial profit (only if no full exit is triggering)
            if not full_exits and self.cfg.get("pm_profit_taking_enabled", True):
                tiers_taken = self._get_tiers_taken(state.symbol, conn)
                profit = self.check_partial_profit(state, tiers_taken)
                if profit.should_exit:
                    self._execute_exit(profit, state, conn, day_trades_remaining, "MIDDAY")
                    decisions.append(profit)
                    continue

            # Pick best full exit
            best = self._select_decision(full_exits)
            if best:
                best.shares_to_sell = best.shares_to_sell or state.shares
                self._execute_exit(best, state, conn, day_trades_remaining, "MIDDAY")
                decisions.append(best)

        return decisions

    # ── Pure exit checks (independently testable) ──

    def check_stop_loss(self, state: PositionState) -> ExitDecision:
        """Exit if position down >8% from entry price."""
        threshold = self.cfg.get("pm_stop_loss_pct", 0.08)
        if state.avg_entry_price <= 0:
            return ExitDecision(should_exit=False, exit_type="STOP_LOSS")

        loss_pct = (state.avg_entry_price - state.current_price) / state.avg_entry_price
        if loss_pct >= threshold:
            return ExitDecision(
                should_exit=True,
                exit_type="STOP_LOSS",
                shares_to_sell=state.shares,
                reason=f"{state.symbol} down {loss_pct:.1%} from entry ${state.avg_entry_price:.2f} (threshold: {threshold:.0%})",
                urgency="URGENT",
            )
        return ExitDecision(should_exit=False, exit_type="STOP_LOSS")

    def check_trailing_stop(self, state: PositionState, high_watermark: float) -> ExitDecision:
        """Exit if price dropped >6% from recorded high watermark."""
        threshold = self.cfg.get("pm_trailing_stop_pct", 0.06)
        if high_watermark <= 0:
            return ExitDecision(should_exit=False, exit_type="TRAILING_STOP")

        drop_pct = (high_watermark - state.current_price) / high_watermark
        if drop_pct >= threshold:
            return ExitDecision(
                should_exit=True,
                exit_type="TRAILING_STOP",
                shares_to_sell=state.shares,
                reason=f"{state.symbol} dropped {drop_pct:.1%} from high ${high_watermark:.2f} (threshold: {threshold:.0%})",
                urgency="NORMAL",
            )
        return ExitDecision(should_exit=False, exit_type="TRAILING_STOP")

    def check_gap_down(self, state: PositionState) -> ExitDecision:
        """Exit if overnight gap >4%. For SQQQ, checks gap UP (inverse)."""
        threshold = self.cfg.get("pm_gap_down_exit_pct", 0.04)
        if state.prev_close <= 0:
            return ExitDecision(should_exit=False, exit_type="GAP_DOWN")

        gap_pct = (state.intraday_open - state.prev_close) / state.prev_close

        # For SQQQ, a gap UP in the underlying is bad (inverse ETF)
        if state.symbol == "SQQQ":
            # SQQQ gaps down when QQQ gaps up — check for negative gap
            if gap_pct <= -threshold:
                return ExitDecision(
                    should_exit=True,
                    exit_type="GAP_DOWN",
                    shares_to_sell=state.shares,
                    reason=f"{state.symbol} gapped {gap_pct:+.1%} (QQQ likely gapped up, threshold: {threshold:.0%})",
                    urgency="URGENT",
                )
        else:
            # For TQQQ and others, a gap down is bad
            if gap_pct <= -threshold:
                return ExitDecision(
                    should_exit=True,
                    exit_type="GAP_DOWN",
                    shares_to_sell=state.shares,
                    reason=f"{state.symbol} gapped {gap_pct:+.1%} (threshold: {threshold:.0%})",
                    urgency="URGENT",
                )

        return ExitDecision(should_exit=False, exit_type="GAP_DOWN")

    def check_regime_emergency(self, state: PositionState, qqq_current: float, sma_250: float) -> ExitDecision:
        """Exit if QQQ fell >3% below SMA-250 intraday. For SQQQ, checks QQQ rising above."""
        threshold = self.cfg.get("pm_regime_emergency_pct", 0.03)
        if sma_250 <= 0:
            return ExitDecision(should_exit=False, exit_type="REGIME_EMERGENCY")

        pct_vs_sma = (qqq_current - sma_250) / sma_250

        if state.symbol == "SQQQ":
            # SQQQ benefits from QQQ falling — emergency is QQQ rising above SMA-250
            if pct_vs_sma >= threshold:
                return ExitDecision(
                    should_exit=True,
                    exit_type="REGIME_EMERGENCY",
                    shares_to_sell=state.shares,
                    reason=f"QQQ {pct_vs_sma:+.1%} above SMA-250 — bearish thesis invalidated (threshold: {threshold:.0%})",
                    urgency="URGENT",
                )
        else:
            # For TQQQ, QQQ falling below SMA-250 is emergency
            if pct_vs_sma <= -threshold:
                return ExitDecision(
                    should_exit=True,
                    exit_type="REGIME_EMERGENCY",
                    shares_to_sell=state.shares,
                    reason=f"QQQ {pct_vs_sma:+.1%} below SMA-250 — regime breakdown (threshold: {threshold:.0%})",
                    urgency="URGENT",
                )

        return ExitDecision(should_exit=False, exit_type="REGIME_EMERGENCY")

    def check_vol_spike(self, state: PositionState, current_vol: float, prev_vol: float) -> ExitDecision:
        """Exit if volume jumped >50% from prior day."""
        threshold = self.cfg.get("pm_vol_spike_exit_pct", 0.50)
        if prev_vol <= 0:
            return ExitDecision(should_exit=False, exit_type="VOL_SPIKE")

        vol_change = (current_vol - prev_vol) / prev_vol
        if vol_change >= threshold:
            return ExitDecision(
                should_exit=True,
                exit_type="VOL_SPIKE",
                shares_to_sell=state.shares,
                reason=f"{state.symbol} volume spiked {vol_change:.0%} ({current_vol:,.0f} vs prev {prev_vol:,.0f}, threshold: {threshold:.0%})",
                urgency="NORMAL",
            )
        return ExitDecision(should_exit=False, exit_type="VOL_SPIKE")

    def check_max_hold_period(self, state: PositionState) -> ExitDecision:
        """Exit if losing position held >15 days."""
        max_days = self.cfg.get("pm_max_hold_days_losing", 15)
        if state.unrealized_pnl_pct >= 0:
            return ExitDecision(should_exit=False, exit_type="MAX_HOLD")
        if state.holding_days > max_days:
            return ExitDecision(
                should_exit=True,
                exit_type="MAX_HOLD",
                shares_to_sell=state.shares,
                reason=f"{state.symbol} losing ({state.unrealized_pnl_pct:+.1%}) and held {state.holding_days} days (max: {max_days})",
                urgency="NORMAL",
            )
        return ExitDecision(should_exit=False, exit_type="MAX_HOLD")

    def check_daily_loss_limit(self, account_equity: float, prev_equity: float) -> ExitDecision:
        """Exit all if account down >5% today."""
        threshold = self.cfg.get("pm_daily_loss_limit_pct", 0.05)
        if prev_equity <= 0:
            return ExitDecision(should_exit=False, exit_type="DAILY_LOSS_LIMIT")

        loss_pct = (prev_equity - account_equity) / prev_equity
        if loss_pct >= threshold:
            return ExitDecision(
                should_exit=True,
                exit_type="DAILY_LOSS_LIMIT",
                shares_to_sell=0,  # Will be filled per-position
                reason=f"Account down {loss_pct:.1%} today (${prev_equity:,.0f} -> ${account_equity:,.0f}, threshold: {threshold:.0%})",
                urgency="URGENT",
            )
        return ExitDecision(should_exit=False, exit_type="DAILY_LOSS_LIMIT")

    def check_partial_profit(self, state: PositionState, tiers_taken: list[float]) -> ExitDecision:
        """Sell 25% at each untaken tier (+8/+15/+25%)."""
        if not self.cfg.get("pm_profit_taking_enabled", True):
            return ExitDecision(should_exit=False, exit_type="PARTIAL_PROFIT")

        tiers = self.cfg.get("pm_profit_tiers", [])
        if state.avg_entry_price <= 0 or state.shares <= 0:
            return ExitDecision(should_exit=False, exit_type="PARTIAL_PROFIT")

        gain_pct = ((state.current_price - state.avg_entry_price) / state.avg_entry_price) * 100

        for tier in tiers:
            threshold = tier["threshold_pct"]
            fraction = tier["sell_fraction"]
            if threshold in tiers_taken:
                continue
            if gain_pct >= threshold:
                shares_to_sell = max(1, int(state.shares * fraction))
                return ExitDecision(
                    should_exit=True,
                    exit_type="PARTIAL_PROFIT",
                    shares_to_sell=shares_to_sell,
                    reason=f"{state.symbol} up {gain_pct:.1f}% — taking {fraction:.0%} profit at +{threshold}% tier",
                    urgency="NORMAL",
                )

        return ExitDecision(should_exit=False, exit_type="PARTIAL_PROFIT")

    # ── Helpers ──

    def _get_position_states(self, conn) -> list[PositionState]:
        """Fetch live positions from Alpaca, enrich with DB context."""
        from db.models import get_position_entry_date

        positions = self.alpaca_client.get_positions()
        managed_symbols = {self.cfg["bull_etf"], self.cfg["bear_etf"]}
        snapshots = {}

        symbols_held = [p["symbol"] for p in positions if p["symbol"] in managed_symbols]
        if symbols_held:
            snapshots = self.alpaca_client.get_snapshot(symbols_held)

        states = []
        for p in positions:
            if p["symbol"] not in managed_symbols:
                continue

            snap = snapshots.get(p["symbol"], {})
            prev_close = snap.get("prev_daily_bar_close", 0) or 0
            intraday_open = snap.get("daily_bar_open", 0) or 0
            intraday_high = snap.get("daily_bar_high", 0) or 0
            intraday_low = snap.get("daily_bar_low", 0) or 0
            current_price = snap.get("latest_trade_price") or p["current_price"]

            # Computed fields
            overnight_gap_pct = ((intraday_open - prev_close) / prev_close) if prev_close > 0 else 0
            intraday_change_pct = ((current_price - intraday_open) / intraday_open) if intraday_open > 0 else 0
            intraday_drawdown_pct = ((intraday_high - current_price) / intraday_high) if intraday_high > 0 else 0

            # Entry date and holding days
            entry_date = get_position_entry_date(conn)
            holding_days = 0
            if entry_date:
                from datetime import date as _date
                try:
                    ed = _date.fromisoformat(entry_date)
                    holding_days = (datetime.now(ET).date() - ed).days
                except (ValueError, TypeError):
                    pass

            states.append(PositionState(
                symbol=p["symbol"],
                shares=p["qty"],
                avg_entry_price=p["avg_entry_price"],
                current_price=current_price,
                market_value=p["market_value"],
                unrealized_pnl_pct=p["unrealized_plpc"],
                entry_date=entry_date,
                holding_days=holding_days,
                intraday_high=intraday_high,
                intraday_low=intraday_low,
                intraday_open=intraday_open,
                prev_close=prev_close,
                overnight_gap_pct=overnight_gap_pct,
                intraday_change_pct=intraday_change_pct,
                intraday_drawdown_pct=intraday_drawdown_pct,
            ))

        return states

    def _get_high_watermark(self, symbol: str, conn) -> dict | None:
        """Read watermark from DB."""
        from db.models import get_position_watermark
        return get_position_watermark(symbol, conn)

    def _update_high_watermark(self, symbol: str, price: float, conn) -> None:
        """Write watermark to DB."""
        from db.models import update_position_watermark
        today_str = datetime.now(ET).date().isoformat()
        update_position_watermark(symbol, price, today_str, conn)

    def _get_tiers_taken(self, symbol: str, conn) -> list[float]:
        """Query position_events for profit tiers taken since entry."""
        from db.models import get_profit_tiers_taken, get_position_entry_date
        entry_date = get_position_entry_date(conn)
        if not entry_date:
            return []
        return get_profit_tiers_taken(symbol, entry_date, conn)

    def _get_prev_equity(self, conn) -> float:
        """Get previous day's account equity from performance table."""
        from db.models import get_db
        with get_db(conn) as c:
            row = c.execute(
                "SELECT account_equity FROM decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row["account_equity"] if row and row["account_equity"] else 0

    def _get_qqq_snapshot(self) -> dict:
        """Get QQQ snapshot data."""
        try:
            snapshots = self.alpaca_client.get_snapshot([self.cfg["underlying"]])
            return snapshots.get(self.cfg["underlying"], {})
        except Exception as e:
            logger.warning(f"Failed to get QQQ snapshot: {e}")
            return {}

    def _get_sma_250(self, conn) -> float:
        """Get SMA-250 from cached bars."""
        try:
            from db.cache import get_cached_bars
            import numpy as np
            bars = get_cached_bars(self.cfg["underlying"], 250, conn)
            if bars and len(bars) >= 250:
                closes = [b["close"] for b in bars]
                return float(np.mean(closes[-250:]))
        except Exception as e:
            logger.warning(f"Failed to compute SMA-250 from cache: {e}")
        return 0

    def _get_prev_volume(self, symbol: str, conn) -> float:
        """Get previous day's volume from cached bars."""
        try:
            from db.cache import get_cached_bars
            bars = get_cached_bars(symbol, 5, conn)
            if bars and len(bars) >= 2:
                return bars[-2]["volume"]
        except Exception:
            pass
        return 0

    def _execute_exit(self, decision: ExitDecision, state: PositionState, conn,
                      day_trades_remaining: int, window: str) -> None:
        """Submit exit order, log to position_events, send Telegram alert."""
        from strategy.executor import execute_rebalance
        from db.models import save_position_event, update_position_event
        import notifications

        # PDT check: URGENT bypasses, NORMAL checks reserve
        if decision.urgency != "URGENT":
            reserve = self.cfg.get("pm_min_day_trades_reserve", 2)
            if day_trades_remaining < reserve:
                logger.info(f"Skipping {decision.exit_type} for {state.symbol}: "
                            f"only {day_trades_remaining} day trades (reserve: {reserve})")
                # Log as skipped
                event_data = {
                    "date": datetime.now(ET).date().isoformat(),
                    "timestamp": datetime.now(ET).isoformat(),
                    "symbol": state.symbol,
                    "window": window,
                    "event_type": decision.exit_type,
                    "shares_before": state.shares,
                    "shares_sold": 0,
                    "shares_after": state.shares,
                    "price": state.current_price,
                    "pnl_pct": state.unrealized_pnl_pct * 100,
                    "order_status": "SKIPPED_PDT",
                    "trigger_detail": decision.reason,
                }
                save_position_event(event_data, conn)
                decision.should_exit = False
                return

        # Log intent before execution
        event_data = {
            "date": datetime.now(ET).date().isoformat(),
            "timestamp": datetime.now(ET).isoformat(),
            "symbol": state.symbol,
            "window": window,
            "event_type": decision.exit_type,
            "shares_before": state.shares,
            "shares_sold": decision.shares_to_sell,
            "shares_after": state.shares - decision.shares_to_sell,
            "price": state.current_price,
            "pnl_pct": state.unrealized_pnl_pct * 100,
            "order_status": "PENDING",
            "trigger_detail": decision.reason,
            "high_watermark": None,
        }
        if decision.exit_type == "PARTIAL_PROFIT":
            # Extract tier from reason
            for tier in self.cfg.get("pm_profit_tiers", []):
                gain_pct = ((state.current_price - state.avg_entry_price) / state.avg_entry_price) * 100
                if gain_pct >= tier["threshold_pct"]:
                    event_data["profit_tier_pct"] = tier["threshold_pct"]
        if decision.exit_type == "TRAILING_STOP":
            wm = self._get_high_watermark(state.symbol, conn)
            event_data["high_watermark"] = wm["high_price"] if wm else None

        event_id = save_position_event(event_data, conn)

        # Execute
        target_shares = state.shares - decision.shares_to_sell
        is_urgent = decision.urgency == "URGENT"
        try:
            result = execute_rebalance(
                target_shares, state.shares, state.current_price,
                self.alpaca_client, is_emergency=is_urgent,
                day_trades_remaining=day_trades_remaining,
                symbol=state.symbol,
            )
            order = result.get("order") or {}
            update_position_event(event_id, {
                "order_id": order.get("order_id"),
                "order_status": "EXECUTED" if result.get("executed") else "FAILED",
            }, conn)
            logger.info(f"Exit {decision.exit_type} for {state.symbol}: {result['reason']}")
        except Exception as e:
            update_position_event(event_id, {"order_status": "FAILED"}, conn)
            logger.error(f"Exit execution failed for {state.symbol}: {e}")

        # Send Telegram alert
        try:
            notifications.send_position_exit_alert({
                "symbol": state.symbol,
                "exit_type": decision.exit_type,
                "shares_sold": decision.shares_to_sell,
                "price": state.current_price,
                "pnl_pct": state.unrealized_pnl_pct * 100,
                "reason": decision.reason,
                "urgency": decision.urgency,
                "window": window,
            })
        except Exception as e:
            logger.warning(f"Failed to send exit alert: {e}")

    @staticmethod
    def _select_decision(decisions: list[ExitDecision]) -> ExitDecision | None:
        """Pick highest-urgency decision. URGENT > NORMAL; same urgency picks largest shares_to_sell."""
        if not decisions:
            return None

        urgency_order = {"URGENT": 0, "NORMAL": 1}
        decisions.sort(key=lambda d: (urgency_order.get(d.urgency, 2), -d.shares_to_sell))
        return decisions[0]
