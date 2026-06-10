# Two-Stage Review

Date: 2026-04-27

Hypothesis:
- `CASH` is masking some valid `SQQQ` opportunities
- split the problem into:
  1. `trade` vs `cash`
  2. `TQQQ` vs `SQQQ` only when a trade is warranted

## Tested structure

Stage 1:
- ExtraTrees binary classifier
- target: `trade` (`TQQQ` or `SQQQ`) vs `cash`

Stage 2:
- ExtraTrees binary classifier
- target: `SQQQ` vs `TQQQ`
- trained only on non-cash rows

## Results

Aggressive two-stage baseline:
- `macro_f1 = 0.272659`
- `balanced_accuracy = 0.371664`
- `accuracy = 0.398136`
- prediction mix:
  - `CASH = 199`
  - `TQQQ = 551`
  - `SQQQ = 1`

Conservative two-stage variants:
- best observed:
  - `macro_f1 = 0.288376`
  - `balanced_accuracy = 0.337149`
  - `accuracy = 0.713715`
- prediction mix:
  - almost entirely `CASH`

## Conclusion

The two-stage allocator did not solve the masking problem.

What happened:
- if the trade threshold is low enough to expose more trade opportunities,
  the model overtrades and floods into `TQQQ`
- if the trade threshold is raised to control false positives,
  the model collapses back into `CASH`
- neither version materially improved `SQQQ` detection

Decision:
- reject the two-stage branch
- keep the current single-stage thresholded ExtraTrees research winner

Practical takeaway:
- `CASH` masking is real
- but a simple two-stage split is not enough to fix it
- the bigger issue is still sparse / weakly separable bearish expectancy labels
