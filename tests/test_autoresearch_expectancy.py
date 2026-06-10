"""Tests for the trading-specific autoresearch expectancy harness."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from autoresearch_expectancy.prepare import (
    LABEL_MODE_LIVE,
    LABEL_MODE_SHADOW_BEARISH_V1,
    build_dataset,
    build_validation_folds,
    build_test_split,
)


class TestAutoresearchExpectancySplits:
    def test_validation_and_test_do_not_overlap(self):
        ds = build_dataset()
        folds = build_validation_folds(ds["dates"])
        train_idx, test_idx = build_test_split(ds["dates"])
        test_set = set(test_idx.tolist())

        assert len(test_set) > 0
        for fold in folds:
            assert test_set.isdisjoint(set(fold.train_idx.tolist()))
            assert test_set.isdisjoint(set(fold.val_idx.tolist()))

    def test_validation_folds_are_chronological(self):
        ds = build_dataset()
        folds = build_validation_folds(ds["dates"])
        for fold in folds:
            train_dates = ds["dates"][fold.train_idx].tolist()
            val_dates = ds["dates"][fold.val_idx].tolist()
            assert max(train_dates) < min(val_dates)

    def test_dataset_shapes_match(self):
        ds = build_dataset()
        assert len(ds["X"]) == len(ds["y"]) == len(ds["dates"])
        assert ds["X"].shape[1] > 0

    def test_shadow_bearish_labels_preserve_shape_and_increase_sqqq_count(self):
        live_ds = build_dataset(label_mode=LABEL_MODE_LIVE)
        shadow_ds = build_dataset(label_mode=LABEL_MODE_SHADOW_BEARISH_V1)

        assert live_ds["X"].shape == shadow_ds["X"].shape
        assert len(live_ds["dates"]) == len(shadow_ds["dates"])
        assert np.sum(shadow_ds["y"] == 2) > np.sum(live_ds["y"] == 2)
