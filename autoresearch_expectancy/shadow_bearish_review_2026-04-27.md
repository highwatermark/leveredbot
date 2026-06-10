# Shadow Bearish Review

Date: 2026-04-27

## Problem

`SQQQ` opportunities are partially masked in the default expectancy dataset because:
- the live labeler vetoes inverse exposure when regime is not inverse-eligible
- those rows remain labeled `CASH` even when simulated `SQQQ` expectancy is clearly better than `CASH` and much better than `TQQQ`

This makes bearish structure hard to learn from the default labels alone.

## Research Addition

Added a research-only label mode in [prepare.py](./prepare.py):
- `live`
- `shadow_bearish_v1`

`shadow_bearish_v1` remaps a `CASH` row to `SQQQ` only when:
- `SQQQ expectancy > 0.02`
- `TQQQ expectancy < -0.02`
- `SQQQ expectancy > TQQQ expectancy + 0.01`

This is intentionally strict. It only exposes clearly asymmetric bearish cases.

## Effect On Label Distribution

Base labels:
- `CASH = 1187`
- `TQQQ = 573`
- `SQQQ = 45`

Shadow bearish v1:
- `CASH = 1023`
- `TQQQ = 573`
- `SQQQ = 209`

## What Improved

On live-labeled 2022 evaluation:
- current default model:
  - `SQQQ recall = 0.00`
  - `SQQQ f1 = 0.00`
- shadow-bearish-trained model:
  - `SQQQ recall = 0.236842`
  - `SQQQ f1 = 0.197802`

So the shadow mode does recover some non-zero bearish detection.

## What Worsened

On `train <= 2025` and `test = 2026`:
- default thresholded ExtraTrees:
  - `accuracy = 0.708333`
  - `macro_f1 = 0.414634`
- shadow-bearish-trained model:
  - `accuracy = 0.569444`
  - `macro_f1 = 0.264050`

Since 2026 had no true `SQQQ` labels, the shadow mode overexpressed bearishness and degraded current-regime performance.

## Conclusion

`shadow_bearish_v1` is useful as a research tool because it:
- exposes masked bearish structure
- proves that the model can recover some `SQQQ` recall once the label masking is reduced

It is **not** the promoted default model because:
- it degrades the live-like 2026 split too much
- it is better treated as a bearish specialist or analysis mode

## Practical Decision

- Keep the default promoted research candidate unchanged:
  - `extratrees_800_thresholded`
- Keep `shadow_bearish_v1` in the harness as a first-class research mode
- Use bearish-only slices to iterate on inverse-specific features next, rather than replacing the default allocator
