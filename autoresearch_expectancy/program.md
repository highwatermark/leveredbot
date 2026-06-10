# autoresearch_expectancy

This is a trading adaptation of Karpathy's `autoresearch` loop.

## Scope

Only edit `train.py`.

Do not modify:
- `prepare.py`
- `final_eval.py`
- any file outside this directory

## Objective

Improve the validation score of the expectancy allocator that chooses among:
- `TQQQ`
- `CASH`
- `SQQQ`

The target is generated from the mirrored live exit path in `prepare.py`.

## Non-contamination rules

- `train.py` must only use the validation folds exposed by `prepare.py`.
- `final_eval.py` is the untouched holdout and must not be used during iteration.
- Do not tune against holdout output.
- Do not alter split boundaries in `prepare.py`.

## Metric priority

Primary:
- `val_macro_f1`

Secondary:
- `val_balanced_acc`

Tertiary:
- `val_accuracy`

Why:
- plain accuracy is dominated by `CASH`
- macro-F1 better penalizes ignoring `TQQQ` / `SQQQ`

## Recommended changes

Allowed ideas inside `train.py`:
- hyperparameters
- class weighting
- thresholding
- alternative classifiers already available in repo dependencies
- calibration / post-processing
- ensemble logic inside `train.py`

Not allowed:
- changing the frozen dataset or split policy
- reading or using final test labels during research

## Commands

Validation loop:
```bash
cd autoresearch_expectancy
../.venv/bin/python train.py > run.log 2>&1
grep '^val_' run.log
```

Final holdout, only after a candidate wins on validation:
```bash
cd autoresearch_expectancy
../.venv/bin/python final_eval.py
```
