#!/usr/bin/env python3
"""
Train and evaluate the expectancy-based TQQQ/CASH/SQQQ model.

Can use the local cache only, or backfill multi-year OHLCV from Alpaca first.
"""

import json
import sys
from pathlib import Path
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, LEVERAGE_CONFIG
from db.cache import get_cached_bars, get_cached_microstructure, get_bars_with_cache
from db.models import get_connection
from strategy.expectancy_signal import ExpectancySignal, determine_best_action, ACTION_LABELS
from strategy.knn_signal import FeatureCalculator


def _load_local_inputs(start_date: str = "2023-01-01", backfill_days: int | None = None):
    conn = get_connection()
    try:
        if backfill_days:
            import alpaca_client
            qqq_bars = get_bars_with_cache("QQQ", backfill_days, alpaca_client.fetch_bars_for_cache, conn)
            tqqq_bars = get_bars_with_cache("TQQQ", backfill_days, alpaca_client.fetch_bars_for_cache, conn)
            sqqq_bars = get_bars_with_cache("SQQQ", backfill_days, alpaca_client.fetch_bars_for_cache, conn)
            cross = {
                sym: get_bars_with_cache(sym, backfill_days, alpaca_client.fetch_bars_for_cache, conn)
                for sym in ("TLT", "GLD", "IWM")
            }
        else:
            qqq_bars = get_cached_bars("QQQ", 4000, conn, start_date=start_date)
            tqqq_bars = get_cached_bars("TQQQ", 4000, conn, start_date=start_date)
            sqqq_bars = get_cached_bars("SQQQ", 4000, conn, start_date=start_date)
            cross = {
                sym: get_cached_bars(sym, 4000, conn, start_date=start_date)
                for sym in ("TLT", "GLD", "IWM")
            }
        micro = get_cached_microstructure("QQQ", start_date, conn)
    finally:
        conn.close()

    vix_path = DATA_DIR / "vix_cache.json"
    vix = {}
    if vix_path.exists():
        with open(vix_path) as f:
            vix = json.load(f)

    return qqq_bars, tqqq_bars, sqqq_bars, vix, cross, micro


def _walk_forward_expectancy(
    qqq_bars,
    tqqq_bars,
    sqqq_bars,
    vix_by_date,
    cross_asset_bars,
    microstructure_by_date,
    start_idx=420,
    retrain_every=50,
):
    qqq_by_date = {b["date"]: b for b in qqq_bars}
    tqqq_by_date = {b["date"]: b for b in tqqq_bars}
    sqqq_by_date = {b["date"]: b for b in sqqq_bars}
    common_dates = sorted(set(qqq_by_date) & set(tqqq_by_date) & set(sqqq_by_date))
    ordered_qqq = [qqq_by_date[d] for d in common_dates]

    model = ExpectancySignal()
    calc = FeatureCalculator()
    predictions = []
    predictions_since_train = retrain_every

    for i in range(start_idx, len(common_dates) - LEVERAGE_CONFIG.get("expectancy_max_hold_days", 5) - 1):
        if predictions_since_train >= retrain_every:
            train_end = common_dates[i]
            train_dates = common_dates[:i + 1]
            train_qqq = [qqq_by_date[d] for d in train_dates]
            train_tqqq = [tqqq_by_date[d] for d in train_dates]
            train_sqqq = [sqqq_by_date[d] for d in train_dates]
            model.fit_from_aligned_bars(
                train_qqq,
                train_tqqq,
                train_sqqq,
                vix_by_date=vix_by_date,
                cross_asset_bars=cross_asset_bars,
                microstructure_by_date=microstructure_by_date,
            )
            predictions_since_train = 0

        if not model.is_fitted:
            continue

        feat = calc.compute_features(
            ordered_qqq,
            i,
            vix_by_date=vix_by_date,
            cross_asset_bars=cross_asset_bars,
            microstructure_by_date=microstructure_by_date,
        )
        if feat is None:
            continue

        pred = model.predict(
            ordered_qqq[:i + 1],
            vix_by_date=vix_by_date,
            cross_asset_bars=cross_asset_bars,
            microstructure_by_date=microstructure_by_date,
        )
        actual_label, details = determine_best_action(
            i,
            common_dates,
            qqq_by_date,
            tqqq_by_date,
            sqqq_by_date,
        )
        predictions.append({
            "date": common_dates[i],
            "pred": pred["action"],
            "actual": actual_label,
            "confidence": pred["confidence"],
            "tqqq_expectancy": details["TQQQ"].expectancy,
            "sqqq_expectancy": details["SQQQ"].expectancy,
        })
        predictions_since_train += 1

    return predictions, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--backfill-days", type=int, default=0)
    args = parser.parse_args()

    qqq_bars, tqqq_bars, sqqq_bars, vix, cross, micro = _load_local_inputs(
        start_date=args.start_date,
        backfill_days=args.backfill_days or None,
    )
    print(f"Loaded local cache: QQQ={len(qqq_bars)} TQQQ={len(tqqq_bars)} SQQQ={len(sqqq_bars)}")
    print(f"Cross assets: " + ", ".join(f"{k}={len(v)}" for k, v in cross.items()))
    print(f"VIX cache rows: {len(vix)} | microstructure rows: {len(micro)}")
    print(f"Feature set size: {FeatureCalculator.FEATURE_COUNT}")

    model = ExpectancySignal()
    trained = model.fit_from_aligned_bars(
        qqq_bars, tqqq_bars, sqqq_bars,
        vix_by_date=vix,
        cross_asset_bars=cross,
        microstructure_by_date=micro,
    )
    if not trained:
        print("Training failed: insufficient local data.")
        return 1

    model_path = Path(__file__).parent.parent / LEVERAGE_CONFIG["expectancy_model_path"]
    model.save(model_path)
    print(f"Saved expectancy model to {model_path}")

    predictions, wf_model = _walk_forward_expectancy(
        qqq_bars, tqqq_bars, sqqq_bars, vix, cross, micro
    )
    if not predictions:
        print("No walk-forward predictions generated.")
        return 1

    y_true = [p["actual"] for p in predictions]
    y_pred = [p["pred"] for p in predictions]
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nWalk-forward predictions: {len(predictions)}")
    print(f"Multiclass accuracy: {accuracy:.1%}")
    print("\nConfusion matrix (rows=actual, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(ACTION_LABELS)))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=list(ACTION_LABELS), zero_division=0))

    action_counts = {label: y_pred.count(label) for label in ACTION_LABELS}
    print("Predicted action mix:", action_counts)
    avg_tqqq = float(np.mean([p["tqqq_expectancy"] for p in predictions]))
    avg_sqqq = float(np.mean([p["sqqq_expectancy"] for p in predictions]))
    print(f"Average simulated expectancy: TQQQ={avg_tqqq:+.3%} SQQQ={avg_sqqq:+.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
