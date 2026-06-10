# Changelog

## 2026-06-10 (3) — Cash-yield sweep (SGOV)

Idle allocated capital (historically ~74% of the sleeve at the 3x tier) now
sweeps into SGOV (0-3 month T-bills, ~4-5% yield) at the main run and is sold
automatically when TQQQ needs the room. Decade replay estimate: adds ~+3%/yr.

- `strategy/cash_sweep.py`: target math, pre-sell before TQQQ buys
  (1%-padded shortfall), end-of-run sweep; never sells and rebuys same run;
  no same-day round trips (PDT-safe)
- `get_allocated_capital`: sweep ETF classed as strategy capital — does not
  shrink allocation (critical on a dedicated account) or count as "other"
- Config: `use_cash_sweep: True`, `sweep_etf: SGOV`, `sweep_buffer_pct: 0.02`,
  `sweep_min_trade_value: 250`
- Sweep failures are non-fatal: the strategy run never breaks on sweep errors

## 2026-06-10 (2) — Sizing raised to 3x tier after decade stress-test

Replayed the new system through 2018 Q4, COVID 2020, the 2022 bear, and the
full 2017–2026 decade at four sizing tiers. The deployed (1x) sizing maxed out
at 17% deployment — too conservative to compound. Chosen tier: **3x**
(sleeves ×3, regime caps 0.60 STRONG_BULL / 0.45 BULL). Decade replay at this
tier: ~17% CAGR, worst-ever drawdown −16.8%, +17% through the 2022 bear
(regime authority keeps the system in cash below SMA-250).

- `max_position_pct` 0.20 → 0.60, `bull_position_pct` 0.15 → 0.45
- Bull sleeves ×3: trend_core 0.36, breakout 0.15, pullback 0.12,
  mean_reversion 0.12
- `max_portfolio_pct` stays 0.30 on the shared account
- New `docs/PROFILE_2K_DEDICATED.md`: profile for a dedicated $2k account
  (`max_portfolio_pct: 1.0`)
- Legacy sizing tests pin the old regime targets they assert against

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
