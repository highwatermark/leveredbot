#!/usr/bin/env python3
"""
Walk-forward comparison of k-NN vs XGBoost direction prediction.

Expands training window from day 300 to end, predicts day i+1,
records actual outcome. Prints accuracy, precision, recall, F1
for both models side by side.

Usage:
    cd /home/ubuntu/leveraged-etf
    python strategy/compare_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import defaultdict

from strategy.knn_signal import KNNSignal, FeatureCalculator
from strategy.xgb_signal import XGBSignal


def _load_data():
    """Load QQQ bars, VIX data, cross-asset bars, and microstructure from cache."""
    import alpaca_client
    from db.models import get_connection, init_tables
    from db.cache import get_bars_with_cache, get_microstructure_with_cache

    init_tables()
    conn = get_connection()

    qqq_bars = get_bars_with_cache("QQQ", 900, alpaca_client.fetch_bars_for_cache, conn)
    print(f"QQQ bars loaded: {len(qqq_bars)}")

    # Cross-asset bars
    cross_asset_bars = {}
    for sym in ("TLT", "GLD", "IWM"):
        try:
            bars = get_bars_with_cache(sym, 900, alpaca_client.fetch_bars_for_cache, conn)
            cross_asset_bars[sym] = bars
            print(f"{sym} bars loaded: {len(bars)}")
        except Exception as e:
            print(f"Warning: {sym} bars failed: {e}")

    # VIX data
    vix_by_date = {}
    try:
        from strategy.vix_data import get_vix_data
        vix_by_date = get_vix_data()
        print(f"VIX data loaded: {len(vix_by_date)} dates")
    except Exception as e:
        print(f"Warning: VIX data failed: {e}")

    # Microstructure data
    microstructure_by_date = {}
    try:
        microstructure_by_date = get_microstructure_with_cache(
            "QQQ", 900, alpaca_client.get_intraday_bars, conn
        )
        print(f"Microstructure data loaded: {len(microstructure_by_date)} dates")
    except Exception as e:
        print(f"Warning: Microstructure data failed: {e}")

    conn.close()
    return qqq_bars, vix_by_date, cross_asset_bars, microstructure_by_date


def _walk_forward(qqq_bars, vix_by_date, cross_asset_bars, microstructure_by_date=None, start_idx=420, retrain_every=50):
    """
    Walk-forward evaluation for both k-NN and XGBoost.

    Args:
        start_idx: First index to predict (needs 200+ bars for features)
        retrain_every: Retrain models every N predictions
    """
    microstructure_by_date = microstructure_by_date or {}
    results = {"knn": [], "xgb": []}
    calc = FeatureCalculator()

    knn = KNNSignal(n_neighbors=7, min_confidence=0.0)  # 0.0 = never FLAT
    xgb = XGBSignal(n_estimators=200, max_depth=4, learning_rate=0.05, min_confidence=0.0)

    total_predictions = len(qqq_bars) - start_idx - 1
    print(f"\nWalk-forward: {total_predictions} predictions (retrain every {retrain_every})")
    print(f"Training window starts at {start_idx} bars, expanding to {len(qqq_bars)}")

    predictions_since_train = retrain_every  # Force initial train

    for i in range(start_idx, len(qqq_bars) - 1):
        # Retrain periodically with expanding window
        if predictions_since_train >= retrain_every:
            train_bars = qqq_bars[:i + 1]
            knn.fit_from_bars(train_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
            xgb.fit_from_bars(train_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
            predictions_since_train = 0

        if not knn.is_fitted or not xgb.is_fitted:
            continue

        # Compute features for current day
        features = calc.compute_features(qqq_bars, i, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
        if features is None:
            continue

        # Actual next-day direction
        actual_up = qqq_bars[i + 1]["close"] > qqq_bars[i]["close"]
        actual_label = 1 if actual_up else 0

        # k-NN prediction
        knn_result = knn.predict(qqq_bars[:i + 1], vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
        knn_pred = 1 if knn_result["direction"] == "LONG" else 0
        results["knn"].append((knn_pred, actual_label, knn_result["confidence"]))

        # XGBoost prediction
        xgb_result = xgb.predict(qqq_bars[:i + 1], vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)
        xgb_pred = 1 if xgb_result["direction"] == "LONG" else 0
        results["xgb"].append((xgb_pred, actual_label, xgb_result["confidence"]))

        predictions_since_train += 1

        if (i - start_idx) % 50 == 0 and i > start_idx:
            n = len(results["knn"])
            knn_acc = sum(1 for p, a, _ in results["knn"] if p == a) / n
            xgb_acc = sum(1 for p, a, _ in results["xgb"] if p == a) / n
            print(f"  Day {i}: kNN={knn_acc:.1%}, XGB={xgb_acc:.1%} ({n} predictions)")

    return results


def _compute_metrics(predictions: list[tuple[int, int, float]]) -> dict:
    """Compute accuracy, precision, recall, F1 from (pred, actual, conf) tuples."""
    if not predictions:
        return {}

    preds = [p for p, _, _ in predictions]
    actuals = [a for _, a, _ in predictions]
    confs = [c for _, _, c in predictions]

    n = len(predictions)
    correct = sum(1 for p, a in zip(preds, actuals) if p == a)
    accuracy = correct / n

    # For "up" class (label=1)
    tp = sum(1 for p, a in zip(preds, actuals) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(preds, actuals) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds, actuals) if p == 0 and a == 1)
    tn = sum(1 for p, a in zip(preds, actuals) if p == 0 and a == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-class accuracy
    up_total = sum(1 for a in actuals if a == 1)
    down_total = sum(1 for a in actuals if a == 0)
    up_correct = tp
    down_correct = tn

    return {
        "accuracy": accuracy,
        "precision_up": precision,
        "recall_up": recall,
        "f1_up": f1,
        "up_accuracy": up_correct / up_total if up_total > 0 else 0,
        "down_accuracy": down_correct / down_total if down_total > 0 else 0,
        "n": n,
        "class_balance": up_total / n if n > 0 else 0,
        "pred_balance": sum(preds) / n if n > 0 else 0,
        "avg_confidence": float(np.mean(confs)),
    }


def _print_results(results: dict, xgb_model: XGBSignal):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD COMPARISON: k-NN vs XGBoost")
    print("=" * 70)

    knn_m = _compute_metrics(results["knn"])
    xgb_m = _compute_metrics(results["xgb"])

    if not knn_m or not xgb_m:
        print("No predictions to evaluate!")
        return

    print(f"\nPredictions: {knn_m['n']}")
    print(f"Actual class balance: {knn_m['class_balance']:.1%} up / {1 - knn_m['class_balance']:.1%} down")
    print(f"50% baseline accuracy: 50.0%")

    header = f"{'Metric':<25} {'k-NN':>10} {'XGBoost':>10} {'Winner':>10}"
    print(f"\n{header}")
    print("-" * 55)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision (up)", "precision_up"),
        ("Recall (up)", "recall_up"),
        ("F1 (up)", "f1_up"),
        ("Up-day accuracy", "up_accuracy"),
        ("Down-day accuracy", "down_accuracy"),
        ("Pred balance (% up)", "pred_balance"),
        ("Avg confidence", "avg_confidence"),
    ]

    for label, key in metrics:
        kv = knn_m[key]
        xv = xgb_m[key]
        if key in ("pred_balance", "avg_confidence"):
            winner = ""
        else:
            winner = "k-NN" if kv > xv + 0.001 else ("XGBoost" if xv > kv + 0.001 else "TIE")
        print(f"  {label:<23} {kv:>9.1%} {xv:>9.1%} {winner:>10}")

    # XGBoost feature importances
    importances = xgb_model.get_feature_importance()
    if importances:
        print(f"\n{'XGBoost Feature Importances':}")
        print("-" * 40)
        for name, imp in importances:
            bar = "#" * int(imp * 100)
            print(f"  {name:<30} {imp:.3f} {bar}")

    print()


def main():
    print("Loading data...")
    qqq_bars, vix_by_date, cross_asset_bars, microstructure_by_date = _load_data()

    if len(qqq_bars) < 350:
        print(f"Insufficient data: {len(qqq_bars)} bars (need 350+)")
        return

    results = _walk_forward(qqq_bars, vix_by_date, cross_asset_bars, microstructure_by_date=microstructure_by_date)

    # Get a trained XGB model for feature importances
    xgb = XGBSignal(n_estimators=200, max_depth=4, learning_rate=0.05, min_confidence=0.0)
    xgb.fit_from_bars(qqq_bars, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars, microstructure_by_date=microstructure_by_date)

    _print_results(results, xgb)


if __name__ == "__main__":
    main()
