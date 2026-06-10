"""
Frozen data prep and evaluation harness for expectancy-model autoresearch.

This file is intentionally read-only during research iterations.
The editable file is `train.py`.

Data policy:
- Features are built from locally cached market data only.
- Labels are generated from the mirrored live-exit path.
- Validation uses fixed chronological folds.
- Test is a final untouched holdout and is NOT used by `train.py`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import DATA_DIR
from db.cache import get_cached_bars, get_cached_microstructure
from db.models import get_connection
from strategy.expectancy_signal import ACTION_LABELS, ACTION_TO_ID, determine_best_action
from strategy.knn_signal import FeatureCalculator

START_DATE = "2018-01-01"
FINAL_TEST_START = "2025-01-01"
VALIDATION_YEARS = ("2022", "2023", "2024")
LABEL_MODE_LIVE = "live"
LABEL_MODE_SHADOW_BEARISH_V1 = "shadow_bearish_v1"


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def load_market_data(start_date: str = START_DATE) -> tuple[list[dict], list[dict], list[dict], dict, dict, dict]:
    conn = get_connection(DATA_DIR / "leveraged_etf.db")
    try:
        qqq = get_cached_bars("QQQ", 4000, conn, start_date=start_date)
        tqqq = get_cached_bars("TQQQ", 4000, conn, start_date=start_date)
        sqqq = get_cached_bars("SQQQ", 4000, conn, start_date=start_date)
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

    return qqq, tqqq, sqqq, vix, cross, micro


def remap_labels(
    base_labels: np.ndarray,
    details: list[dict],
    mode: str = LABEL_MODE_LIVE,
) -> np.ndarray:
    if mode == LABEL_MODE_LIVE:
        return np.array(base_labels, copy=True)
    if mode != LABEL_MODE_SHADOW_BEARISH_V1:
        raise ValueError(f"Unsupported label mode: {mode}")

    remapped = np.array(base_labels, copy=True)
    for idx, (label, trade_details) in enumerate(zip(remapped, details)):
        if int(label) != ACTION_TO_ID["CASH"]:
            continue
        sqqq_expectancy = trade_details["SQQQ"].expectancy
        tqqq_expectancy = trade_details["TQQQ"].expectancy
        if (
            sqqq_expectancy > 0.02
            and tqqq_expectancy < -0.02
            and sqqq_expectancy > tqqq_expectancy + 0.01
        ):
            remapped[idx] = ACTION_TO_ID["SQQQ"]
    return remapped


def build_dataset(start_date: str = START_DATE, label_mode: str = LABEL_MODE_LIVE) -> dict:
    qqq, tqqq, sqqq, vix, cross, micro = load_market_data(start_date)

    qqq_by = {b["date"]: b for b in qqq}
    tqqq_by = {b["date"]: b for b in tqqq}
    sqqq_by = {b["date"]: b for b in sqqq}
    common_dates = sorted(set(qqq_by) & set(tqqq_by) & set(sqqq_by))
    ordered_qqq = [qqq_by[d] for d in common_dates]

    calc = FeatureCalculator()
    rows = []
    labels = []
    dates = []
    details = []

    for i in range(250, len(common_dates) - 6):
        feat = calc.compute_features(
            ordered_qqq,
            i,
            vix_by_date=vix,
            cross_asset_bars=cross,
            microstructure_by_date=micro,
        )
        if feat is None:
            continue
        label, trade_details = determine_best_action(i, common_dates, qqq_by, tqqq_by, sqqq_by)
        rows.append(feat)
        labels.append(ACTION_TO_ID[label])
        dates.append(common_dates[i])
        details.append(trade_details)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    y = remap_labels(y, details, mode=label_mode)
    dates_arr = np.array(dates)
    return {
        "X": X,
        "y": y,
        "dates": dates_arr,
        "details": details,
        "feature_names": FeatureCalculator.feature_names(),
    }


def build_validation_folds(dates: np.ndarray) -> list[Fold]:
    folds: list[Fold] = []
    for year in VALIDATION_YEARS:
        val_mask = np.array([d.startswith(year) for d in dates])
        train_mask = np.array([d < f"{year}-01-01" for d in dates])
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append(
            Fold(
                name=f"val_{year}",
                train_idx=np.where(train_mask)[0],
                val_idx=np.where(val_mask)[0],
            )
        )
    return folds


def build_test_split(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.where(dates < FINAL_TEST_START)[0]
    test_idx = np.where(dates >= FINAL_TEST_START)[0]
    return train_idx, test_idx


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def action_name(y_id: int) -> str:
    return ACTION_LABELS[int(y_id)]
