# Changelog

## 2026-06-10 — Bench SQQQ, regime authority, sleeve engine (live-validation release)

Root-cause review of live trading (Feb 9 – Jun 10) found the deployed system
buying SQQQ every afternoon on k-NN SHORT signals and force-selling it every
morning via the position manager's REGIME_EMERGENCY check — a structural churn
loop that fought a +42% TQQQ rally. Full diagnosis in
`docs/REVIEW_MEMO_2026-04-24.md` and the 2026-06-10 validation below.

### Live validation results (Alpaca fills + decisions DB, Mar–Jun 2026)

- k-NN LONG next-day accuracy 57%, rising to **79% at 3- and 10-day horizons** —
  genuine edge that compounds with holding time.
- k-NN SHORT accuracy **38% at 1 day, 29% at 10 days** — anti-predictive at
  every horizon. QQQ averaged **+3.5% in the 10 days after a SHORT call**.
  Confidence adds nothing (0.75+ bucket: 33% correct).
- SQQQ hold-period simulation over 23 actual entries: next-open exit averaged
  +0.35%; holding 5 days → −4.9%, 10 days → −9.6%. Longer holds lose more;
  the entries themselves were the error.

### Changed

- **SQQQ benched** (`use_sqqq_trading: False`). Re-enable only behind a
  walk-forward-validated short model.
- **k-NN SHORT no longer trims the long sleeve** (`sleeve_model_short_mult: 1.0`).
  LONG signals remain active; SHORT/FLAT are reported in Telegram only.
- **Regime authority is controlling** (`regime_authority: "controlling"`):
  inverse exposure only permitted in RISK_OFF/BREAKDOWN (Gate S9);
  `allow_inverse_in_bull/strong_bull: False`; TQQQ→SQQQ rotation disabled.
- **RSI overbought is no longer a hard entry gate** — handled in sizing as
  `pause_adds` (holds existing position, trims fresh-entry size by 25%).
  The hard gate had blocked every TQQQ entry from Apr 15 to May 27, then
  admitted a full-size buy at the May 28 local top.
- **Rule-based sleeve engine** (`strategy/sleeves.py`): bull/bear sleeve
  targets (trend core, breakout, pullback, mean-reversion) replace single-knob
  sizing; tactical-model overlays applied multiplicatively.
- **Model arbitration**: k-NN primary + XGBoost secondary
  (`prediction_model: "both"`), disagreement applies a 0.82 sizing haircut
  instead of a binary block.
- **Expectancy model scaffolding** (`strategy/expectancy_signal.py`,
  `autoresearch_expectancy/`) — research artifacts, not yet wired into sizing.
- `data/*.db` and `logs/*.log` removed from version control (live DB and logs
  belong to the server; they were force-added in an earlier commit).

### Tests

- 389 passing. SQQQ/sizing unit tests pinned to the legacy (non-sleeve) path
  they exercise; sleeve engine covered by `tests/test_sleeves.py`;
  model policy by `tests/test_model_policy.py`.

## 2026-04-24 — SQQQ trend override entry path

- SQQQ entries allowed on bearish trend (QQQ >3% below SMA-50, ROC-20 < −2%)
  without a k-NN SHORT signal. *(Superseded: SQQQ benched 2026-06-10.)*

## 2026-04-17 — Frozen-position fixes

- Sells bypass entry gates; stale-position force exit after 5 consecutive
  gate-fail days; TQQQ→SQQQ rotation on high-confidence k-NN SHORT.
  *(Rotation superseded: disabled 2026-06-10.)*

## 2026-03-16 — XGBoost reporting

- XGBoost signal added to Telegram reports with context and reasoning.

## 2026-02-09 — Initial live deployment

- 5-state regime detection, 14-gate entry checklist, position manager with
  windowed intraday exits, k-NN overlay (report-only), Alpaca paper account
  shared with momentum-agent (30% allocation cap).
