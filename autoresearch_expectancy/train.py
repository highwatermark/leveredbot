"""
Editable autoresearch model file for expectancy ranking.

The research loop should only modify this file.
Validation is measured on fixed chronological folds from prepare.py.
Final test evaluation is separate in final_eval.py.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from prepare import (
    build_dataset,
    build_validation_folds,
    evaluate_predictions,
)

# ---------------------------------------------------------------------------
# Editable research surface
# ---------------------------------------------------------------------------

N_ESTIMATORS = 800
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2
CLASS_WEIGHT = "balanced"
MAX_FEATURES = "sqrt"
CASH_THRESHOLD = 0.48
SQQQ_THRESHOLD = 0.10
SQQQ_MARGIN_OVER_TQQQ = 0.02


def make_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        max_features=MAX_FEATURES,
        random_state=42,
        n_jobs=1,
    )


def fit_predict_fold(X_train, y_train, X_val):
    model = make_model()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)
    preds = np.argmax(probs, axis=1)

    cash_mask = probs[:, 0] >= CASH_THRESHOLD
    preds[cash_mask] = 0

    sqqq_mask = (
        ~cash_mask
        & (probs[:, 2] >= SQQQ_THRESHOLD)
        & (probs[:, 2] > probs[:, 1] + SQQQ_MARGIN_OVER_TQQQ)
    )
    preds[sqqq_mask] = 2
    return preds, model, None


def run_validation() -> dict:
    ds = build_dataset()
    X = ds["X"]
    y = ds["y"]
    dates = ds["dates"]
    folds = build_validation_folds(dates)

    all_true = []
    all_pred = []
    fold_metrics = []

    for fold in folds:
        preds, _, _ = fit_predict_fold(X[fold.train_idx], y[fold.train_idx], X[fold.val_idx])
        metrics = evaluate_predictions(y[fold.val_idx], preds)
        metrics["fold"] = fold.name
        metrics["n_val"] = int(len(fold.val_idx))
        fold_metrics.append(metrics)
        all_true.extend(y[fold.val_idx].tolist())
        all_pred.extend(preds.tolist())

    overall = evaluate_predictions(np.array(all_true), np.array(all_pred))
    return {
        "overall": overall,
        "folds": fold_metrics,
        "n_total_val": len(all_true),
        "pred_counts": Counter(int(v) for v in all_pred),
    }


def main():
    result = run_validation()
    overall = result["overall"]
    print("---")
    print(f"val_macro_f1:        {overall['macro_f1']:.6f}")
    print(f"val_balanced_acc:    {overall['balanced_accuracy']:.6f}")
    print(f"val_accuracy:        {overall['accuracy']:.6f}")
    print(f"val_weighted_f1:     {overall['weighted_f1']:.6f}")
    print(f"val_samples:         {result['n_total_val']}")
    print(f"pred_cash:           {result['pred_counts'].get(0, 0)}")
    print(f"pred_tqqq:           {result['pred_counts'].get(1, 0)}")
    print(f"pred_sqqq:           {result['pred_counts'].get(2, 0)}")


if __name__ == "__main__":
    main()
