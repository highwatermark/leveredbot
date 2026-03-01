"""
XGBoost direction prediction as a signal overlay.

Predicts next-day QQQ direction (LONG/SHORT/FLAT) using XGBClassifier
on the same 20-feature vector as KNNSignal. Handles class imbalance
via scale_pos_weight.

Usage:
    xgb = XGBSignal()
    xgb.fit_from_bars(qqq_bars, vix_by_date=vix, cross_asset_bars=cross,
                      microstructure_by_date=micro)
    result = xgb.predict(qqq_bars, vix_by_date=vix, cross_asset_bars=cross,
                         microstructure_by_date=micro)
"""

import pickle
import logging
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from config import LEVERAGE_CONFIG
from strategy.knn_signal import FeatureCalculator, FEATURE_VERSION, MIN_TRAINING_SAMPLES

logger = logging.getLogger(__name__)


class XGBSignal:
    """
    XGBoost direction prediction signal.

    Predicts whether QQQ will close up or down the next trading day,
    using gradient-boosted trees on the same 20-feature vector as KNNSignal.
    """

    def __init__(
        self,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
        min_confidence: float | None = None,
    ):
        self.n_estimators = n_estimators or LEVERAGE_CONFIG.get("xgb_n_estimators", 200)
        self.max_depth = max_depth or LEVERAGE_CONFIG.get("xgb_max_depth", 4)
        self.learning_rate = learning_rate or LEVERAGE_CONFIG.get("xgb_learning_rate", 0.05)
        self.min_confidence = min_confidence or LEVERAGE_CONFIG.get("xgb_min_confidence", 0.55)
        self.model: XGBClassifier | None = None
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
        Train XGBoost model from historical bar data.

        Uses scale_pos_weight to handle class imbalance (more up days than down).

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

        # Compute class imbalance ratio for scale_pos_weight
        n_neg = int(np.sum(y == 0))
        n_pos = int(np.sum(y == 1))
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.training_samples = len(X)
        self.feature_count = FeatureCalculator.FEATURE_COUNT
        self.feature_version = FEATURE_VERSION

        logger.info(
            f"XGBoost model trained: {len(X)} samples, "
            f"estimators={self.n_estimators}, depth={self.max_depth}, "
            f"scale_pos_weight={scale_pos_weight:.2f}"
        )
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

        Returns same structure as KNNSignal.predict().
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            return self._neutral_prediction("Model not fitted")

        calc = FeatureCalculator()
        features = calc.compute_features(bars, len(bars) - 1, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
        if features is None:
            return self._neutral_prediction("Insufficient feature data")

        X = self.scaler.transform(features.reshape(1, -1))
        probabilities = self.model.predict_proba(X)[0]

        p_up = float(probabilities[1]) if len(probabilities) > 1 else 0.5
        p_down = float(probabilities[0])

        confidence = max(p_up, p_down)

        if confidence < self.min_confidence:
            direction = "FLAT"
            adjustment = 0.75
        elif p_up > p_down:
            direction = "LONG"
            adjustment = 1.0
        else:
            direction = "SHORT"
            # Use same conviction curve as KNNSignal
            if confidence <= 0.65:
                adjustment = 0.75
            elif confidence >= 0.80:
                adjustment = 0.40
            else:
                t = (confidence - 0.65) / (0.80 - 0.65)
                adjustment = 0.75 - t * 0.35

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "adjustment": round(adjustment, 4),
            "probabilities": [round(p_down, 4), round(p_up, 4)],
        }

    def _neutral_prediction(self, reason: str) -> dict:
        logger.warning(f"XGBoost neutral prediction: {reason}")
        return {
            "direction": "FLAT",
            "confidence": 0.5,
            "adjustment": 1.0,
            "probabilities": [0.5, 0.5],
        }

    def get_feature_importance(self) -> list[tuple[str, float]]:
        """Return sorted feature importances (name, importance)."""
        if not self.is_fitted or self.model is None:
            return []
        names = FeatureCalculator.feature_names()
        importances = self.model.feature_importances_
        pairs = list(zip(names, [float(x) for x in importances]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def save(self, path: Path | str) -> None:
        """Persist the fitted model and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_confidence": self.min_confidence,
                "training_samples": self.training_samples,
                "feature_count": self.feature_count,
                "feature_version": FEATURE_VERSION,
            }, f)
        logger.info(f"XGBoost model saved to {path} ({self.feature_count} features, v{FEATURE_VERSION})")

    def load(self, path: Path | str) -> bool:
        """Load a previously fitted model from disk.

        Returns False if feature count or version mismatch.
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
                    f"XGB model feature count mismatch: saved={saved_features}, "
                    f"expected={FeatureCalculator.FEATURE_COUNT}. Retrain required."
                )
                return False

            if saved_version != FEATURE_VERSION:
                logger.warning(
                    f"XGB model feature version mismatch: saved=v{saved_version}, "
                    f"expected=v{FEATURE_VERSION}. Retrain required."
                )
                return False

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.n_estimators = data["n_estimators"]
            self.max_depth = data["max_depth"]
            self.learning_rate = data["learning_rate"]
            self.min_confidence = data["min_confidence"]
            self.training_samples = data.get("training_samples", 0)
            self.feature_count = saved_features
            self.feature_version = saved_version
            self.is_fitted = True
            logger.info(f"XGBoost model loaded from {path} ({self.training_samples} samples, v{self.feature_version})")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False
