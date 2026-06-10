# Rolling Walk-Forward Review

Date: 2026-04-26

Scope:
- 32 chronological validation iterations
- 8 shifted validation windows
- 4 model candidates per window
- Final 2025+ holdout left untouched during this pass

Validation windows:
- 2021-01-01 to 2021-06-30
- 2021-07-01 to 2021-12-31
- 2022-01-01 to 2022-06-30
- 2022-07-01 to 2022-12-31
- 2023-01-01 to 2023-06-30
- 2023-07-01 to 2023-12-31
- 2024-01-01 to 2024-06-30
- 2024-07-01 to 2024-12-31

Candidates:
- `xgb_baseline`
- `extratrees_300_raw`
- `extratrees_800_thresh`
- `extratrees_800_conservative`

## Aggregate Summary

| Candidate | Avg Macro F1 | Avg Balanced Acc | Avg Accuracy | Window Wins |
|---|---:|---:|---:|---:|
| `xgb_baseline` | 0.455593 | 0.485932 | 0.619714 | 5 |
| `extratrees_800_conservative` | 0.438485 | 0.470370 | 0.618115 | 0 |
| `extratrees_800_thresh` | 0.437888 | 0.468773 | 0.628164 | 1 |
| `extratrees_300_raw` | 0.432722 | 0.476396 | 0.565366 | 2 |

## Key Takeaways

1. The fixed-fold validation winner was not the most stable rolling candidate.
   - `extratrees_800_thresh` won the fixed 2022-2024 fold set.
   - But across shifted time windows, `xgb_baseline` had the best average macro F1 and the most window wins.

2. The thresholded ExtraTrees variants improved headline accuracy in some windows by collapsing toward `CASH`.
   - That helped in low-opportunity windows.
   - It hurt robustness across regime shifts.

3. The most difficult windows remain 2022 bearish windows.
   - All models struggled to represent rare `SQQQ` outcomes cleanly.
   - Most candidates still predicted almost no `SQQQ`.

4. Oversampling and more conservative thresholding did not help enough.
   - They generally reduced macro F1 or simply pushed more predictions into `CASH`.

## Practical Conclusion

- Best fixed-fold tuner: `extratrees_800_thresh`
- Best rolling-stability model: `xgb_baseline`

This means the current research winner depends on what we optimize for:
- If we optimize for the fixed 2022-2024 folds only, keep the thresholded ExtraTrees.
- If we optimize for stability across shifted train/validation windows, prefer the baseline XGBoost family.

The safer promotion criterion is rolling stability, not one validation slice.
