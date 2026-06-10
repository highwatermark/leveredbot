"""
Expectancy-based action ranker for TQQQ / CASH / SQQQ.

Uses the same feature pipeline as the current k-NN/XGBoost stack, but trains
on a trade-decision target:
    given today's close, which action has the best realized expectancy under
    a mirrored live-exit path over the next few sessions?

This v1 uses daily OHLCV + cached VIX/cross-asset/microstructure inputs and an
approximate multi-day exit simulator built from the same pure PositionManager
checks used by the live/backtest system.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from config import LEVERAGE_CONFIG
from strategy.knn_signal import FeatureCalculator, FEATURE_VERSION, MIN_TRAINING_SAMPLES
from strategy.position_manager import PositionManager, PositionState
from strategy.regime import detect_regime
from strategy.sizing import is_sqqq_regime_allowed

logger = logging.getLogger(__name__)

ACTION_LABELS = ("CASH", "TQQQ", "SQQQ")
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_LABELS)}


@dataclass
class SimulatedTrade:
    action: str
    expectancy: float
    exit_reason: str
    holding_days: int


def _build_position_state(
    symbol: str,
    entry_price: float,
    current_price: float,
    bar: dict,
    prev_close: float,
    entry_date: str,
    current_date: str,
) -> PositionState:
    holding_days = 0
    try:
        from datetime import date as _date
        holding_days = (_date.fromisoformat(current_date) - _date.fromisoformat(entry_date)).days
    except ValueError:
        pass

    unrealized = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0.0
    return PositionState(
        symbol=symbol,
        shares=100,
        avg_entry_price=entry_price,
        current_price=current_price,
        market_value=current_price * 100,
        unrealized_pnl_pct=unrealized,
        entry_date=entry_date,
        holding_days=holding_days,
        intraday_high=bar["high"],
        intraday_low=bar["low"],
        intraday_open=bar["open"],
        prev_close=prev_close,
        overnight_gap_pct=((bar["open"] - prev_close) / prev_close) if prev_close > 0 else 0.0,
        intraday_change_pct=((current_price - bar["open"]) / bar["open"]) if bar["open"] > 0 else 0.0,
        intraday_drawdown_pct=((bar["high"] - current_price) / bar["high"]) if bar["high"] > 0 else 0.0,
    )


def _simulate_action_expectancy(
    action: str,
    entry_index: int,
    common_dates: list[str],
    qqq_by_date: dict[str, dict],
    tqqq_by_date: dict[str, dict],
    sqqq_by_date: dict[str, dict],
    closes_until_entry: list[float],
    max_hold_days: int,
) -> SimulatedTrade:
    """Simulate one trade from today's close using the mirrored daily exit path."""
    if action == "CASH":
        return SimulatedTrade(action="CASH", expectancy=0.0, exit_reason="CASH", holding_days=0)

    pm = PositionManager(config=LEVERAGE_CONFIG)
    symbol = "TQQQ" if action == "TQQQ" else "SQQQ"
    bar_map = tqqq_by_date if symbol == "TQQQ" else sqqq_by_date

    entry_date = common_dates[entry_index]
    entry_price = bar_map[entry_date]["close"]
    remaining_fraction = 1.0
    realized_value = 0.0
    high_watermark = entry_price
    tiers_taken: list[float] = []
    exit_reason = "HOLD_TO_HORIZON"
    last_trade_index = min(len(common_dates) - 1, entry_index + max_hold_days)

    for idx in range(entry_index + 1, last_trade_index + 1):
        dt = common_dates[idx]
        prev_dt = common_dates[idx - 1]
        bar = bar_map[dt]
        prev_bar = bar_map[prev_dt]
        qqq_bar = qqq_by_date[dt]

        prior_closes = [qqq_by_date[d]["close"] for d in common_dates[max(0, idx - 250):idx]]
        sma_250 = float(np.mean(prior_closes[-250:])) if len(prior_closes) >= 250 else 0.0

        # Morning open exits.
        state_open = _build_position_state(symbol, entry_price, bar["open"], bar, prev_bar["close"], entry_date, dt)
        morning_candidates = [
            pm.check_gap_down(state_open),
            pm.check_stop_loss(state_open),
            pm.check_regime_emergency(state_open, qqq_bar["open"], sma_250),
        ]
        morning = pm._select_decision([c for c in morning_candidates if c.should_exit])
        if morning:
            realized_value += remaining_fraction * bar["open"]
            exit_reason = morning.exit_type
            remaining_fraction = 0.0
            break

        # Midday full exits on adverse path.
        high_watermark = max(high_watermark, bar["high"])
        state_low = _build_position_state(symbol, entry_price, bar["low"], bar, prev_bar["close"], entry_date, dt)
        midday_candidates = [
            pm.check_stop_loss(state_low),
            pm.check_trailing_stop(state_low, high_watermark),
            pm.check_vol_spike(state_low, bar["volume"], prev_bar["volume"]),
            pm.check_max_hold_period(state_low),
        ]
        midday = pm._select_decision([c for c in midday_candidates if c.should_exit])
        if midday:
            realized_value += remaining_fraction * bar["low"]
            exit_reason = midday.exit_type
            remaining_fraction = 0.0
            break

        # Partial profit tiers on favorable path.
        state_high = _build_position_state(symbol, entry_price, bar["high"], bar, prev_bar["close"], entry_date, dt)
        profit = pm.check_partial_profit(state_high, tiers_taken)
        if profit.should_exit and remaining_fraction > 0:
            gain_pct = ((bar["high"] - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            for tier in LEVERAGE_CONFIG.get("pm_profit_tiers", []):
                threshold = tier["threshold_pct"]
                if gain_pct >= threshold and threshold not in tiers_taken:
                    sell_fraction = float(tier["sell_fraction"])
                    remaining_fraction = max(0.0, remaining_fraction - sell_fraction)
                    realized_value += sell_fraction * bar["high"]
                    tiers_taken.append(threshold)
                    exit_reason = "PARTIAL_PROFIT"
                    break

        if idx == last_trade_index and remaining_fraction > 0:
            realized_value += remaining_fraction * bar["close"]
            remaining_fraction = 0.0
            exit_reason = "CLOSE_HORIZON"

    expectancy = (realized_value / entry_price) - 1.0 if entry_price > 0 else 0.0
    return SimulatedTrade(
        action=action,
        expectancy=float(expectancy),
        exit_reason=exit_reason,
        holding_days=max(1, min(max_hold_days, last_trade_index - entry_index)),
    )


def determine_best_action(
    entry_index: int,
    common_dates: list[str],
    qqq_by_date: dict[str, dict],
    tqqq_by_date: dict[str, dict],
    sqqq_by_date: dict[str, dict],
    max_hold_days: int | None = None,
    cash_buffer_pct: float | None = None,
) -> tuple[str, dict[str, SimulatedTrade]]:
    """Return best action label and the simulated trade details for each sleeve."""
    max_hold_days = max_hold_days or LEVERAGE_CONFIG.get("expectancy_max_hold_days", 5)
    cash_buffer_pct = cash_buffer_pct if cash_buffer_pct is not None else LEVERAGE_CONFIG.get("expectancy_cash_buffer_pct", 0.0025)

    closes = [qqq_by_date[d]["close"] for d in common_dates[max(0, entry_index - 250):entry_index + 1]]
    qqq_close = closes[-1]
    sma_50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else 0.0
    sma_250 = float(np.mean(closes[-250:])) if len(closes) >= 250 else 0.0
    regime = detect_regime(qqq_close, sma_50, sma_250)

    candidates = {
        "CASH": SimulatedTrade("CASH", 0.0, "CASH", 0),
        "TQQQ": _simulate_action_expectancy("TQQQ", entry_index, common_dates, qqq_by_date, tqqq_by_date, sqqq_by_date, closes, max_hold_days),
        "SQQQ": _simulate_action_expectancy("SQQQ", entry_index, common_dates, qqq_by_date, tqqq_by_date, sqqq_by_date, closes, max_hold_days),
    }

    eligible = {"CASH", "TQQQ"}
    if is_sqqq_regime_allowed(regime):
        eligible.add("SQQQ")
    if regime in ("RISK_OFF", "BREAKDOWN"):
        eligible.discard("TQQQ")

    best = "CASH"
    best_value = cash_buffer_pct
    for action in eligible:
        expectancy = candidates[action].expectancy
        if action != "CASH" and expectancy > best_value:
            best = action
            best_value = expectancy

    return best, candidates


class ExpectancySignal:
    """Multiclass expectancy ranker using XGBoost."""

    def __init__(
        self,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
    ):
        self.n_estimators = n_estimators or LEVERAGE_CONFIG.get("xgb_n_estimators", 200)
        self.max_depth = max_depth or LEVERAGE_CONFIG.get("xgb_max_depth", 4)
        self.learning_rate = learning_rate or LEVERAGE_CONFIG.get("xgb_learning_rate", 0.05)
        self.model: XGBClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.is_fitted = False
        self.training_samples = 0
        self.feature_count = FeatureCalculator.FEATURE_COUNT
        self.feature_version = FEATURE_VERSION

    def fit_from_aligned_bars(
        self,
        qqq_bars: list[dict],
        tqqq_bars: list[dict],
        sqqq_bars: list[dict],
        vix_by_date: dict[str, float] | None = None,
        cross_asset_bars: dict[str, list[dict]] | None = None,
        microstructure_by_date: dict[str, dict[str, float]] | None = None,
    ) -> bool:
        qqq_by_date = {b["date"]: b for b in qqq_bars}
        tqqq_by_date = {b["date"]: b for b in tqqq_bars}
        sqqq_by_date = {b["date"]: b for b in sqqq_bars}
        common_dates = sorted(set(qqq_by_date) & set(tqqq_by_date) & set(sqqq_by_date))
        ordered_qqq = [qqq_by_date[d] for d in common_dates]

        X: list[np.ndarray] = []
        y: list[int] = []
        calc = FeatureCalculator()
        max_hold_days = LEVERAGE_CONFIG.get("expectancy_max_hold_days", 5)

        for i in range(200, len(common_dates) - max_hold_days - 1):
            feat = calc.compute_features(
                ordered_qqq, i,
                vix_by_date=vix_by_date,
                cross_asset_bars=cross_asset_bars,
                microstructure_by_date=microstructure_by_date,
            )
            if feat is None:
                continue
            label, _ = determine_best_action(
                i, common_dates, qqq_by_date, tqqq_by_date, sqqq_by_date,
                max_hold_days=max_hold_days,
            )
            X.append(feat)
            y.append(ACTION_TO_ID[label])

        if len(X) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient expectancy training data: {len(X)} samples (need {MIN_TRAINING_SAMPLES})")
            self.is_fitted = False
            return False

        X_arr = np.array(X)
        y_arr = np.array(y)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=len(ACTION_LABELS),
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X_scaled, y_arr)
        self.is_fitted = True
        self.training_samples = len(X)
        logger.info(f"Expectancy model trained: {len(X)} samples, features={self.feature_count}")
        return True

    def predict(
        self,
        qqq_bars: list[dict],
        vix_by_date: dict[str, float] | None = None,
        cross_asset_bars: dict[str, list[dict]] | None = None,
        microstructure_by_date: dict[str, dict[str, float]] | None = None,
    ) -> dict:
        if not self.is_fitted or self.model is None or self.scaler is None:
            return self._neutral_prediction("Model not fitted")

        feat = FeatureCalculator.compute_features(
            qqq_bars,
            len(qqq_bars) - 1,
            vix_by_date=vix_by_date,
            cross_asset_bars=cross_asset_bars,
            microstructure_by_date=microstructure_by_date,
        )
        if feat is None:
            return self._neutral_prediction("Insufficient feature data")

        probs = self.model.predict_proba(self.scaler.transform(feat.reshape(1, -1)))[0]
        best_idx = int(np.argmax(probs))
        return {
            "action": ACTION_LABELS[best_idx],
            "confidence": round(float(probs[best_idx]), 4),
            "probabilities": {
                ACTION_LABELS[i]: round(float(probs[i]), 4) for i in range(len(ACTION_LABELS))
            },
        }

    def _neutral_prediction(self, reason: str) -> dict:
        logger.warning(f"Expectancy model neutral prediction: {reason}")
        return {
            "action": "CASH",
            "confidence": 0.5,
            "probabilities": {"CASH": 1.0, "TQQQ": 0.0, "SQQQ": 0.0},
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "training_samples": self.training_samples,
                "feature_count": self.feature_count,
                "feature_version": FEATURE_VERSION,
            }, f)

    def load(self, path: Path | str) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("feature_count") != FeatureCalculator.FEATURE_COUNT:
                logger.warning("Expectancy model feature count mismatch; retrain required.")
                return False
            if data.get("feature_version") != FEATURE_VERSION:
                logger.warning("Expectancy model feature version mismatch; retrain required.")
                return False
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.n_estimators = data.get("n_estimators", self.n_estimators)
            self.max_depth = data.get("max_depth", self.max_depth)
            self.learning_rate = data.get("learning_rate", self.learning_rate)
            self.training_samples = data.get("training_samples", 0)
            self.feature_count = data.get("feature_count", FeatureCalculator.FEATURE_COUNT)
            self.feature_version = data.get("feature_version", FEATURE_VERSION)
            self.is_fitted = self.model is not None and self.scaler is not None
            return self.is_fitted
        except Exception as e:
            logger.warning(f"Expectancy model load failed: {e}")
            return False
