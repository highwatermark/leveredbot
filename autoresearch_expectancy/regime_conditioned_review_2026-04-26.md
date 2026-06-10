# Regime-Conditioned Review

Date: 2026-04-26

Objective:
- test whether regime-conditioned training or regime-conditioned decision rules
  improve validation performance versus the current fixed-fold winner

Reference winner:
- `extratrees_800_thresholded`
- validation metrics:
  - `macro_f1 = 0.383479`
  - `balanced_accuracy = 0.391028`
  - `accuracy = 0.671105`

## Regime definition used in experiments

Derived directly from existing features:
- `distance_from_50ma`
- `ma50_slope_10d`
- `ma_20_50_cross`

Buckets:
- `bull`
- `neutral / transition`
- `bear`

## Tested variants

1. Regime-conditioned XGBoost
- separate XGBoost behavior by regime bucket
- best result:
  - `macro_f1 = 0.336398`
  - `balanced_accuracy = 0.346732`
  - `accuracy = 0.589880`

2. Regime-conditioned ExtraTrees
- separate ExtraTrees behavior by regime bucket
- best result:
  - `macro_f1 = 0.363203`
  - `balanced_accuracy = 0.370732`
  - `accuracy = 0.665779`

3. Regime-conditioned hybrid
- XGBoost in bull bucket
- ExtraTrees elsewhere
- best result:
  - `macro_f1 = 0.372418`
  - `balanced_accuracy = 0.379608`
  - `accuracy = 0.668442`

## Conclusion

None of the regime-conditioned variants beat the current fixed-fold winner.

The best regime-conditioned hybrid came closest, but still trailed:
- `0.372418` macro F1 vs `0.383479`

Practical implication:
- regime conditioning is not yet the lever that improves this model family
- the larger problem is still sparse bearish / `SQQQ` representation and
  class imbalance under the mirrored expectancy label

Decision:
- keep the current research winner unchanged
- do not promote a regime-conditioned version
