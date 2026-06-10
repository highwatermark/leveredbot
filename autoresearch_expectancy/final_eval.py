"""Final untouched holdout evaluation for the expectancy autoresearch loop."""

from __future__ import annotations

import importlib

import numpy as np

from prepare import ACTION_LABELS, build_dataset, build_test_split, evaluate_predictions


def main():
    train_mod = importlib.import_module("train")
    ds = build_dataset()
    X = ds["X"]
    y = ds["y"]
    dates = ds["dates"]
    train_idx, test_idx = build_test_split(dates)

    preds, _, _ = train_mod.fit_predict_fold(X[train_idx], y[train_idx], X[test_idx])
    metrics = evaluate_predictions(y[test_idx], preds)
    print("---")
    print(f"test_macro_f1:       {metrics['macro_f1']:.6f}")
    print(f"test_balanced_acc:   {metrics['balanced_accuracy']:.6f}")
    print(f"test_accuracy:       {metrics['accuracy']:.6f}")
    print(f"test_weighted_f1:    {metrics['weighted_f1']:.6f}")
    print(f"test_samples:        {len(test_idx)}")


if __name__ == "__main__":
    main()
