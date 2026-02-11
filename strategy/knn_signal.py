"""
k-NN direction prediction as a signal overlay.

Predicts next-day QQQ direction (LONG/SHORT/FLAT) using distance-weighted
k-NN classification on a 16-feature vector computed from daily bar data.

Features (16):
  # Price-based
  1. intraday_return      - open→close return
  2. prior_day_return     - previous day's close→close return
  3. two_day_return       - 2-day cumulative return
  4. five_day_return      - 5-day cumulative return
  # Volatility
  5. intraday_range       - (high-low)/open
  6. atr_ratio            - current ATR vs 20-day avg ATR
  # Trend
  7. distance_from_20ma   - (close-MA20)/MA20
  8. distance_from_50ma   - (close-MA50)/MA50
  9. distance_from_200ma  - (close-MA200)/MA200
  10. ma_20_50_cross      - (MA20-MA50)/close, normalized cross
  # Momentum
  11. rsi_14              - 14-day RSI (normalized 0-1)
  12. rsi_deviation       - |RSI-50|/50, distance from neutral
  13. momentum_10         - 10-day rate of change
  # Volume
  14. volume_ratio        - today's volume vs 20-day average
  # VIX
  15. vix_level           - current VIX / 100 (scaled)
  16. vix_change          - daily VIX change (pct)

Usage:
    knn = KNNSignal()
    knn.fit_from_bars(qqq_bars, vix_by_date=vix_data)
    result = knn.predict(qqq_bars, vix_by_date=vix_data)
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


class FeatureCalculator:
    """Compute the 16-feature vector from daily bar data + VIX."""

    FEATURE_COUNT = 16

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
    def compute_features(
        bars: list[dict], index: int, vix_by_date: dict[str, float] | None = None,
    ) -> np.ndarray | None:
        """
        Compute 16-feature vector at a given bar index.

        Args:
            bars: List of bar dicts with keys: date, open, high, low, close, volume
            index: Index of the bar to compute features for (needs 200+ bars before it)
            vix_by_date: Optional mapping of date→VIX close for features 15-16.
                         Falls back to defaults if None or date not found.

        Returns:
            numpy array of 16 features, or None if insufficient data
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

        # ── Price-based ──

        # Feature 1: intraday return
        intraday_return = (close - open_price) / open_price if open_price > 0 else 0

        # Feature 2: prior day return
        prior_close = prev["close"]
        prior_day_return = (close - prior_close) / prior_close if prior_close > 0 else 0

        # Feature 3: 2-day return
        if index >= 2:
            two_day_ago = bars[index - 2]["close"]
            two_day_return = (close - two_day_ago) / two_day_ago if two_day_ago > 0 else 0
        else:
            two_day_return = 0

        # Feature 4: 5-day return
        if index >= 5:
            five_day_ago = bars[index - 5]["close"]
            five_day_return = (close - five_day_ago) / five_day_ago if five_day_ago > 0 else 0
        else:
            five_day_return = 0

        # ── Volatility ──

        # Feature 5: intraday range
        intraday_range = (high - low) / open_price if open_price > 0 else 0

        # Feature 6: ATR ratio — today's True Range vs 20-day average TR
        if index >= 20:
            today_tr = FeatureCalculator._compute_true_range(bars, index)
            tr_values = [FeatureCalculator._compute_true_range(bars, i) for i in range(index - 19, index + 1)]
            avg_tr = float(np.mean(tr_values))
            atr_ratio = today_tr / avg_tr if avg_tr > 0 else 1.0
        else:
            atr_ratio = 1.0

        # ── Trend ──

        # Feature 7: distance from 20-MA
        if len(closes) >= 20:
            ma20 = float(np.mean(closes[-20:]))
            dist_20ma = (close - ma20) / ma20 if ma20 > 0 else 0
        else:
            ma20 = 0
            dist_20ma = 0

        # Feature 8: distance from 50-MA
        if len(closes) >= 50:
            ma50 = float(np.mean(closes[-50:]))
            dist_50ma = (close - ma50) / ma50 if ma50 > 0 else 0
        else:
            ma50 = 0
            dist_50ma = 0

        # Feature 9: distance from 200-MA
        if len(closes) >= 200:
            ma200 = float(np.mean(closes[-200:]))
            dist_200ma = (close - ma200) / ma200 if ma200 > 0 else 0
        else:
            dist_200ma = 0

        # Feature 10: MA 20/50 cross — (MA20 - MA50) normalized by price
        if ma20 > 0 and ma50 > 0 and close > 0:
            ma_20_50_cross = (ma20 - ma50) / close
        else:
            ma_20_50_cross = 0

        # ── Momentum ──

        # Feature 11: RSI-14 (normalized 0-1)
        rsi = FeatureCalculator.calculate_rsi(closes, period=14)
        rsi_normalized = rsi / 100.0

        # Feature 12: RSI deviation — distance from neutral (50)
        rsi_deviation = abs(rsi - 50.0) / 50.0

        # Feature 13: 10-day momentum (rate of change)
        if index >= 10:
            ten_day_ago = bars[index - 10]["close"]
            momentum_10 = (close - ten_day_ago) / ten_day_ago if ten_day_ago > 0 else 0
        else:
            momentum_10 = 0

        # ── Volume ──

        # Feature 14: volume ratio — today vs 20-day average
        today_volume = bar.get("volume", 0) or 0
        if index >= 20 and today_volume > 0:
            volumes = [bars[i].get("volume", 0) or 0 for i in range(index - 19, index + 1)]
            avg_volume = float(np.mean(volumes))
            volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        # ── VIX ──

        vix_by_date = vix_by_date or {}
        bar_date = bar.get("date", "")
        prev_date = prev.get("date", "")

        vix_today = vix_by_date.get(bar_date, FeatureCalculator.DEFAULT_VIX)
        vix_prev = vix_by_date.get(prev_date, FeatureCalculator.DEFAULT_VIX)

        # Feature 15: VIX level (scaled to ~0-1 range; VIX 20 → 0.20)
        vix_level = vix_today / 100.0

        # Feature 16: VIX daily change (percentage)
        vix_change = (vix_today - vix_prev) / vix_prev if vix_prev > 0 else 0

        return np.array([
            intraday_return,      # 1
            prior_day_return,     # 2
            two_day_return,       # 3
            five_day_return,      # 4
            intraday_range,       # 5
            atr_ratio,            # 6
            dist_20ma,            # 7
            dist_50ma,            # 8
            dist_200ma,           # 9
            ma_20_50_cross,       # 10
            rsi_normalized,       # 11
            rsi_deviation,        # 12
            momentum_10,          # 13
            volume_ratio,         # 14
            vix_level,            # 15
            vix_change,           # 16
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "intraday_return",
            "prior_day_return",
            "two_day_return",
            "five_day_return",
            "intraday_range",
            "atr_ratio",
            "distance_from_20ma",
            "distance_from_50ma",
            "distance_from_200ma",
            "ma_20_50_cross",
            "rsi_14",
            "rsi_deviation",
            "momentum_10",
            "volume_ratio",
            "vix_level",
            "vix_change",
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

    def fit_from_bars(
        self, bars: list[dict], vix_by_date: dict[str, float] | None = None,
    ) -> bool:
        """
        Train the k-NN model from historical bar data.

        Each training sample: features at day i → label is direction at day i+1.
        Label: 1 if next day's close > today's close, else 0.

        Args:
            bars: List of bar dicts with open/high/low/close/volume
            vix_by_date: Optional mapping of date→VIX close for VIX features.

        Returns:
            True if training succeeded, False if insufficient data
        """
        calc = FeatureCalculator()
        X = []
        y = []

        for i in range(200, len(bars) - 1):
            features = calc.compute_features(bars, i, vix_by_date=vix_by_date)
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

        logger.info(f"k-NN model trained: {len(X)} samples, k={self.n_neighbors}, features={self.feature_count}")
        return True

    def predict(self, bars: list[dict], vix_by_date: dict[str, float] | None = None) -> dict:
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
        features = calc.compute_features(bars, len(bars) - 1, vix_by_date=vix_by_date)
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
            adjustment = 0.75  # Low conviction → reduce 25%
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

        LONG predictions: high confidence → no reduction.
          0.55-0.65 → 1.0 (mild conviction, full size)
          0.65+     → 1.0 (strong conviction, full size)

        SHORT predictions: higher confidence → more reduction.
          0.55-0.65 → 0.75 (mild disagreement, reduce 25%)
          0.65-0.80 → linear from 0.75 down to 0.40
          0.80+     → 0.40 (strong disagreement, reduce 60%)
        """
        if bullish:
            return 1.0

        # SHORT: scale reduction with confidence
        if confidence <= 0.65:
            return 0.75
        elif confidence >= 0.80:
            return 0.40
        else:
            # Linear interpolation: 0.65→0.75, 0.80→0.40
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
            }, f)
        logger.info(f"k-NN model saved to {path} ({self.feature_count} features)")

    def load(self, path: Path | str) -> bool:
        """Load a previously fitted model from disk.

        Returns False if the model was trained with a different feature count
        (e.g. old 10-feature model), forcing a retrain.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            saved_features = data.get("feature_count", 10)
            if saved_features != FeatureCalculator.FEATURE_COUNT:
                logger.warning(
                    f"Model feature count mismatch: saved={saved_features}, "
                    f"expected={FeatureCalculator.FEATURE_COUNT}. Retrain required."
                )
                return False

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.n_neighbors = data["n_neighbors"]
            self.min_confidence = data["min_confidence"]
            self.training_samples = data.get("training_samples", 0)
            self.feature_count = saved_features
            self.is_fitted = True
            logger.info(f"k-NN model loaded from {path} ({self.training_samples} samples, {self.feature_count} features)")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_similar_days(
        self, bars: list[dict], n: int = 5, vix_by_date: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Find the N most similar historical days to the current day.

        Useful for Telegram reporting — shows what happened on similar days.

        Returns:
            List of dicts with {date, distance, next_day_return}
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            return []

        calc = FeatureCalculator()
        features = calc.compute_features(bars, len(bars) - 1, vix_by_date=vix_by_date)
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
