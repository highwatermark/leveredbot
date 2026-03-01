"""
k-NN direction prediction as a signal overlay.

Predicts next-day QQQ direction (LONG/SHORT/FLAT) using distance-weighted
k-NN classification on a 20-feature vector computed from daily bar data.

Features (20):
  # Price-based
  1. intraday_return      - open->close return
  2. five_day_return      - 5-day cumulative return
  # Volatility
  3. intraday_range       - (high-low)/open
  4. atr_ratio            - current ATR vs 20-day avg ATR
  # Trend
  5. distance_from_7ma    - (close-MA7)/MA7  (short-term mean reversion)
  6. distance_from_50ma   - (close-MA50)/MA50
  7. ma_7_20_cross        - (MA7-MA20)/close (fast cross signal)
  8. ma_20_50_cross       - (MA20-MA50)/close, normalized cross
  # Momentum
  9. rsi_14               - 14-day RSI (normalized 0-1)
  # Volume
  10. volume_ratio        - today's volume vs 20-day average
  # VIX
  11. vix_level           - current VIX / 100 (scaled)
  12. vix_change          - daily VIX change (pct)
  # Cross-asset
  13. tlt_return_5d       - 5-day TLT return (bond sentiment)
  14. gld_return_5d       - 5-day GLD return (gold sentiment)
  15. iwm_qqq_ratio_change_5d - 5-day change in IWM/QQQ ratio (breadth)
  # Seasonality
  16. day_of_week         - 0-4 scaled to 0.0-1.0
  # Microstructure (intraday)
  17. last_hour_volume_ratio  - volume(3-4pm) / volume(rest of day)
  18. vwap_deviation          - (close - VWAP) / VWAP
  19. closing_momentum        - last_30min_return - day_return
  20. volume_acceleration     - volume(last 2h) / volume(first 2h)

Usage:
    knn = KNNSignal()
    knn.fit_from_bars(qqq_bars, vix_by_date=vix_data, cross_asset_bars=cross,
                      microstructure_by_date=micro)
    result = knn.predict(qqq_bars, vix_by_date=vix_data, cross_asset_bars=cross,
                         microstructure_by_date=micro)
"""

import pickle
import logging
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import LEVERAGE_CONFIG

logger = logging.getLogger(__name__)

# Minimum training samples required
MIN_TRAINING_SAMPLES = 200

# Feature version — bump when feature set changes to auto-reject old models
FEATURE_VERSION = 3


class FeatureCalculator:
    """Compute the 20-feature vector from daily bar data + VIX + cross-asset + microstructure."""

    FEATURE_COUNT = 20

    # Default VIX values when data is unavailable (VIX ~20 is "normal")
    DEFAULT_VIX = 20.0

    @staticmethod
    def calculate_rsi(closes: list[float], period: int = 14) -> float:
        """RSI from closing prices. Returns 50.0 if insufficient data."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_true_range(bars: list[dict], index: int) -> float:
        """Compute True Range at a given bar index."""
        bar = bars[index]
        high, low = bar["high"], bar["low"]
        if index > 0:
            prev_close = bars[index - 1]["close"]
            return max(high - low, abs(high - prev_close), abs(low - prev_close))
        return high - low

    @staticmethod
    def _get_cross_asset_return(
        cross_asset_bars: dict[str, list[dict]],
        symbol: str,
        bar_date: str,
        period: int = 5,
    ) -> float:
        """Get N-day return for a cross-asset symbol aligned to bar_date.

        Returns 0.0 if data is unavailable or insufficient.
        """
        bars_list = cross_asset_bars.get(symbol, [])
        if not bars_list:
            return 0.0

        # Build date->index lookup on first call (cached via dict)
        date_idx = {b["date"]: i for i, b in enumerate(bars_list)}
        idx = date_idx.get(bar_date)
        if idx is None or idx < period:
            return 0.0

        current_close = bars_list[idx]["close"]
        past_close = bars_list[idx - period]["close"]
        if past_close <= 0:
            return 0.0
        return (current_close - past_close) / past_close

    @staticmethod
    def _get_iwm_qqq_ratio_change(
        cross_asset_bars: dict[str, list[dict]],
        qqq_close: float,
        qqq_close_5d_ago: float,
        bar_date: str,
        period: int = 5,
    ) -> float:
        """5-day change in IWM/QQQ ratio.

        Returns 0.0 if data is unavailable.
        """
        iwm_bars = cross_asset_bars.get("IWM", [])
        if not iwm_bars:
            return 0.0

        date_idx = {b["date"]: i for i, b in enumerate(iwm_bars)}
        idx = date_idx.get(bar_date)
        if idx is None or idx < period:
            return 0.0

        iwm_now = iwm_bars[idx]["close"]
        iwm_past = iwm_bars[idx - period]["close"]

        if qqq_close <= 0 or qqq_close_5d_ago <= 0 or iwm_past <= 0:
            return 0.0

        ratio_now = iwm_now / qqq_close
        ratio_past = iwm_past / qqq_close_5d_ago
        if ratio_past <= 0:
            return 0.0
        return (ratio_now - ratio_past) / ratio_past

    @staticmethod
    def compute_features(
        bars: list[dict],
        index: int,
        vix_by_date: dict[str, float] | None = None,
        cross_asset_bars: dict[str, list[dict]] | None = None,
        microstructure_by_date: dict[str, dict[str, float]] | None = None,
    ) -> np.ndarray | None:
        """
        Compute 20-feature vector at a given bar index.

        Args:
            bars: List of bar dicts with keys: date, open, high, low, close, volume
            index: Index of the bar to compute features for (needs 200+ bars before it)
            vix_by_date: Optional mapping of date->VIX close for VIX features.
            cross_asset_bars: Optional mapping of symbol->bars for TLT, GLD, IWM.
            microstructure_by_date: Optional mapping of date->microstructure features.

        Returns:
            numpy array of 20 features, or None if insufficient data
        """
        if index < 200:
            return None

        bar = bars[index]
        prev = bars[index - 1]
        closes = [b["close"] for b in bars[max(0, index - 200):index + 1]]

        open_price = bar["open"]
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        bar_date = bar.get("date", "")
        prev_date = prev.get("date", "")

        cross_asset_bars = cross_asset_bars or {}

        # ── Price-based ──

        # Feature 1: intraday return
        intraday_return = (close - open_price) / open_price if open_price > 0 else 0

        # Feature 2: 5-day return
        if index >= 5:
            five_day_ago = bars[index - 5]["close"]
            five_day_return = (close - five_day_ago) / five_day_ago if five_day_ago > 0 else 0
        else:
            five_day_return = 0

        # ── Volatility ──

        # Feature 3: intraday range
        intraday_range = (high - low) / open_price if open_price > 0 else 0

        # Feature 4: ATR ratio — today's True Range vs 20-day average TR
        if index >= 20:
            today_tr = FeatureCalculator._compute_true_range(bars, index)
            tr_values = [FeatureCalculator._compute_true_range(bars, i) for i in range(index - 19, index + 1)]
            avg_tr = float(np.mean(tr_values))
            atr_ratio = today_tr / avg_tr if avg_tr > 0 else 1.0
        else:
            atr_ratio = 1.0

        # ── Trend ──

        # Feature 5: distance from 7-MA (short-term mean reversion)
        if len(closes) >= 7:
            ma7 = float(np.mean(closes[-7:]))
            dist_7ma = (close - ma7) / ma7 if ma7 > 0 else 0
        else:
            ma7 = 0
            dist_7ma = 0

        # Feature 6: distance from 50-MA
        if len(closes) >= 50:
            ma50 = float(np.mean(closes[-50:]))
            dist_50ma = (close - ma50) / ma50 if ma50 > 0 else 0
        else:
            ma50 = 0
            dist_50ma = 0

        # Compute MA20 for cross signals (not a standalone feature)
        if len(closes) >= 20:
            ma20 = float(np.mean(closes[-20:]))
        else:
            ma20 = 0

        # Feature 7: MA 7/20 cross — (MA7 - MA20) normalized by price
        if ma7 > 0 and ma20 > 0 and close > 0:
            ma_7_20_cross = (ma7 - ma20) / close
        else:
            ma_7_20_cross = 0

        # Feature 8: MA 20/50 cross — (MA20 - MA50) normalized by price
        if ma20 > 0 and ma50 > 0 and close > 0:
            ma_20_50_cross = (ma20 - ma50) / close
        else:
            ma_20_50_cross = 0

        # ── Momentum ──

        # Feature 9: RSI-14 (normalized 0-1)
        rsi = FeatureCalculator.calculate_rsi(closes, period=14)
        rsi_normalized = rsi / 100.0

        # ── Volume ──

        # Feature 10: volume ratio — today vs 20-day average
        today_volume = bar.get("volume", 0) or 0
        if index >= 20 and today_volume > 0:
            volumes = [bars[i].get("volume", 0) or 0 for i in range(index - 19, index + 1)]
            avg_volume = float(np.mean(volumes))
            volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        # ── VIX ──

        vix_by_date = vix_by_date or {}

        vix_today = vix_by_date.get(bar_date, FeatureCalculator.DEFAULT_VIX)
        vix_prev = vix_by_date.get(prev_date, FeatureCalculator.DEFAULT_VIX)

        # Feature 11: VIX level (scaled to ~0-1 range; VIX 20 -> 0.20)
        vix_level = vix_today / 100.0

        # Feature 12: VIX daily change (percentage)
        vix_change = (vix_today - vix_prev) / vix_prev if vix_prev > 0 else 0

        # ── Cross-asset ──

        # Feature 13: TLT 5-day return
        tlt_return_5d = FeatureCalculator._get_cross_asset_return(
            cross_asset_bars, "TLT", bar_date, period=5
        )

        # Feature 14: GLD 5-day return
        gld_return_5d = FeatureCalculator._get_cross_asset_return(
            cross_asset_bars, "GLD", bar_date, period=5
        )

        # Feature 15: IWM/QQQ ratio change 5d
        qqq_close_5d_ago = bars[index - 5]["close"] if index >= 5 else close
        iwm_qqq_ratio_change_5d = FeatureCalculator._get_iwm_qqq_ratio_change(
            cross_asset_bars, close, qqq_close_5d_ago, bar_date, period=5
        )

        # ── Seasonality ──

        # Feature 16: day of week (0=Monday .. 4=Friday, scaled to 0-1)
        try:
            from datetime import datetime as _dt
            dt = _dt.strptime(bar_date, "%Y-%m-%d")
            day_of_week = dt.weekday() / 4.0  # 0.0 - 1.0
        except (ValueError, TypeError):
            day_of_week = 0.5  # Fallback to midweek

        # ── Microstructure ──
        microstructure_by_date = microstructure_by_date or {}
        micro = microstructure_by_date.get(bar_date, {})

        # Feature 17: last_hour_volume_ratio
        last_hour_volume_ratio = micro.get("last_hour_volume_ratio", 0.0)

        # Feature 18: vwap_deviation
        vwap_deviation_feat = micro.get("vwap_deviation", 0.0)

        # Feature 19: closing_momentum
        closing_momentum = micro.get("closing_momentum", 0.0)

        # Feature 20: volume_acceleration
        volume_acceleration = micro.get("volume_acceleration", 0.0)

        return np.array([
            intraday_return,          # 0 - price
            five_day_return,          # 1 - price
            intraday_range,           # 2 - volatility
            atr_ratio,                # 3 - volatility
            dist_7ma,                 # 4 - trend
            dist_50ma,                # 5 - trend
            ma_7_20_cross,            # 6 - trend
            ma_20_50_cross,           # 7 - trend
            rsi_normalized,           # 8 - momentum
            volume_ratio,             # 9 - volume
            vix_level,                # 10 - vix
            vix_change,               # 11 - vix
            tlt_return_5d,            # 12 - cross-asset
            gld_return_5d,            # 13 - cross-asset
            iwm_qqq_ratio_change_5d,  # 14 - cross-asset
            day_of_week,              # 15 - seasonality
            last_hour_volume_ratio,   # 16 - microstructure
            vwap_deviation_feat,      # 17 - microstructure
            closing_momentum,         # 18 - microstructure
            volume_acceleration,      # 19 - microstructure
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "intraday_return",
            "five_day_return",
            "intraday_range",
            "atr_ratio",
            "distance_from_7ma",
            "distance_from_50ma",
            "ma_7_20_cross",
            "ma_20_50_cross",
            "rsi_14",
            "volume_ratio",
            "vix_level",
            "vix_change",
            "tlt_return_5d",
            "gld_return_5d",
            "iwm_qqq_ratio_change_5d",
            "day_of_week",
            "last_hour_volume_ratio",
            "vwap_deviation",
            "closing_momentum",
            "volume_acceleration",
        ]


class KNNSignal:
    """
    k-NN direction prediction signal.

    Predicts whether QQQ will close up or down the next trading day,
    using distance-weighted k-NN on a 16-feature vector.
    """

    def __init__(
        self,
        n_neighbors: int | None = None,
        min_confidence: float | None = None,
    ):
        self.n_neighbors = n_neighbors or LEVERAGE_CONFIG.get("knn_neighbors", 7)
        self.min_confidence = min_confidence or LEVERAGE_CONFIG.get("knn_min_confidence", 0.55)
        self.model: KNeighborsClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.is_fitted = False
        self.training_samples = 0
        self.feature_count = FeatureCalculator.FEATURE_COUNT
        self.feature_version = FEATURE_VERSION

    def fit_from_bars(
        self,
        bars: list[dict],
        vix_by_date: dict[str, float] | None = None,
        cross_asset_bars: dict[str, list[dict]] | None = None,
        microstructure_by_date: dict[str, dict[str, float]] | None = None,
    ) -> bool:
        """
        Train the k-NN model from historical bar data.

        Each training sample: features at day i -> label is direction at day i+1.
        Label: 1 if next day's close > today's close, else 0.

        Returns:
            True if training succeeded, False if insufficient data
        """
        calc = FeatureCalculator()
        X = []
        y = []

        for i in range(200, len(bars) - 1):
            features = calc.compute_features(bars, i, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
            if features is None:
                continue

            # Label: 1 if next day close > today close
            next_close = bars[i + 1]["close"]
            today_close = bars[i]["close"]
            label = 1 if next_close > today_close else 0

            X.append(features)
            y.append(label)

        if len(X) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient training data: {len(X)} samples (need {MIN_TRAINING_SAMPLES})")
            self.is_fitted = False
            return False

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights="distance",
            metric="minkowski",
            p=2,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.training_samples = len(X)
        self.feature_count = FeatureCalculator.FEATURE_COUNT
        self.feature_version = FEATURE_VERSION

        logger.info(f"k-NN model trained: {len(X)} samples, k={self.n_neighbors}, features={self.feature_count}")
        return True

    def predict(
        self,
        bars: list[dict],
        vix_by_date: dict[str, float] | None = None,
        cross_asset_bars: dict[str, list[dict]] | None = None,
        microstructure_by_date: dict[str, dict[str, float]] | None = None,
    ) -> dict:
        """
        Predict next-day direction from the latest bar data.

        Returns:
            {
                "direction": "LONG" | "SHORT" | "FLAT",
                "confidence": float (0.0-1.0),
                "adjustment": float (sizing multiplier),
                "probabilities": [p_down, p_up],
            }
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            return self._neutral_prediction("Model not fitted")

        calc = FeatureCalculator()
        features = calc.compute_features(bars, len(bars) - 1, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
        if features is None:
            return self._neutral_prediction("Insufficient feature data")

        X = self.scaler.transform(features.reshape(1, -1))
        probabilities = self.model.predict_proba(X)[0]

        # probabilities[0] = P(down), probabilities[1] = P(up)
        p_up = float(probabilities[1]) if len(probabilities) > 1 else 0.5
        p_down = float(probabilities[0])

        confidence = max(p_up, p_down)

        if confidence < self.min_confidence:
            direction = "FLAT"
            adjustment = 0.75  # Low conviction -> reduce 25%
        elif p_up > p_down:
            direction = "LONG"
            adjustment = self._conviction_adjustment(confidence, bullish=True)
        else:
            direction = "SHORT"
            adjustment = self._conviction_adjustment(confidence, bullish=False)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "adjustment": round(adjustment, 4),
            "probabilities": [round(p_down, 4), round(p_up, 4)],
        }

    @staticmethod
    def _conviction_adjustment(confidence: float, bullish: bool) -> float:
        """
        Continuous conviction-based sizing adjustment.

        LONG predictions: high confidence -> no reduction.
          0.55-0.65 -> 1.0 (mild conviction, full size)
          0.65+     -> 1.0 (strong conviction, full size)

        SHORT predictions: higher confidence -> more reduction.
          0.55-0.65 -> 0.75 (mild disagreement, reduce 25%)
          0.65-0.80 -> linear from 0.75 down to 0.40
          0.80+     -> 0.40 (strong disagreement, reduce 60%)
        """
        if bullish:
            return 1.0

        # SHORT: scale reduction with confidence
        if confidence <= 0.65:
            return 0.75
        elif confidence >= 0.80:
            return 0.40
        else:
            # Linear interpolation: 0.65->0.75, 0.80->0.40
            t = (confidence - 0.65) / (0.80 - 0.65)
            return 0.75 - t * 0.35

    def _neutral_prediction(self, reason: str) -> dict:
        """Return a neutral prediction when model can't predict."""
        logger.warning(f"k-NN neutral prediction: {reason}")
        return {
            "direction": "FLAT",
            "confidence": 0.5,
            "adjustment": 1.0,  # Neutral = no impact on sizing
            "probabilities": [0.5, 0.5],
        }

    def save(self, path: Path | str) -> None:
        """Persist the fitted model and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_neighbors": self.n_neighbors,
                "min_confidence": self.min_confidence,
                "training_samples": self.training_samples,
                "feature_count": self.feature_count,
                "feature_version": FEATURE_VERSION,
            }, f)
        logger.info(f"k-NN model saved to {path} ({self.feature_count} features, v{FEATURE_VERSION})")

    def load(self, path: Path | str) -> bool:
        """Load a previously fitted model from disk.

        Returns False if the model was trained with a different feature count
        or feature version, forcing a retrain.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            saved_features = data.get("feature_count", 10)
            saved_version = data.get("feature_version", 1)

            if saved_features != FeatureCalculator.FEATURE_COUNT:
                logger.warning(
                    f"Model feature count mismatch: saved={saved_features}, "
                    f"expected={FeatureCalculator.FEATURE_COUNT}. Retrain required."
                )
                return False

            if saved_version != FEATURE_VERSION:
                logger.warning(
                    f"Model feature version mismatch: saved=v{saved_version}, "
                    f"expected=v{FEATURE_VERSION}. Retrain required."
                )
                return False

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.n_neighbors = data["n_neighbors"]
            self.min_confidence = data["min_confidence"]
            self.training_samples = data.get("training_samples", 0)
            self.feature_count = saved_features
            self.feature_version = saved_version
            self.is_fitted = True
            logger.info(f"k-NN model loaded from {path} ({self.training_samples} samples, {self.feature_count} features, v{self.feature_version})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_similar_days(
        self,
        bars: list[dict],
        n: int = 5,
        vix_by_date: dict[str, float] | None = None,
        cross_asset_bars: dict[str, list[dict]] | None = None,
        microstructure_by_date: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        """
        Find the N most similar historical days to the current day.

        Returns:
            List of dicts with {date, distance, next_day_return}
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            return []

        calc = FeatureCalculator()
        features = calc.compute_features(bars, len(bars) - 1, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
        if features is None:
            return []

        X_query = self.scaler.transform(features.reshape(1, -1))
        distances, indices = self.model.kneighbors(X_query, n_neighbors=min(n, self.training_samples))

        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            # Map training index back to bar index (training starts at index 200)
            bar_idx = 200 + int(idx)
            if bar_idx + 1 < len(bars):
                today_close = bars[bar_idx]["close"]
                next_close = bars[bar_idx + 1]["close"]
                next_return = (next_close - today_close) / today_close if today_close > 0 else 0
                similar.append({
                    "date": bars[bar_idx].get("date", "unknown"),
                    "distance": round(float(dist), 4),
                    "next_day_return": round(next_return * 100, 2),
                })
        return similar
