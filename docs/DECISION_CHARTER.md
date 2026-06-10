# Leveraged ETF Strategy Decision Charter

Date: 2026-04-24
Scope: Regime/model precedence, TQQQ/SQQQ side selection, risk semantics, and validation rules before code changes.

## Objective

Run the strategy as a coherent two-sided leveraged allocator:

- Hold `TQQQ` when the approved directional thesis is bullish.
- Hold `SQQQ` when the approved directional thesis is bearish.
- Hold `cash` or reduced exposure when conviction is mixed or insufficient.
- Persist with the chosen side while the thesis holds.
- Flip sides only when the transition rule is explicitly satisfied.

## Policy Decisions

### 1. Regime Authority

Decision: `Regime is controlling for default side selection.`

Meaning:

- `BULL` / `STRONG_BULL` defaults to `TQQQ`, not `SQQQ`.
- `RISK_OFF` / `BREAKDOWN` defaults to `cash` or bearish posture, depending on approved bearish-entry rules.
- Tactical models do not silently invert the book against regime.

### 2. Model Role

Decision: `Models are tactical conviction inputs, not unconstrained side-selection controllers.`

Meaning:

- `k-NN` and `XGBoost` may adjust conviction, sizing, or timing.
- They may not force a flip to `SQQQ` in a bullish regime unless a separately approved override contract is satisfied.
- Model disagreement reduces risk; it does not increase aggressiveness.

### 3. Disagreement Policy

Decision: `Model disagreement maps to reduced conviction, not full inversion.`

Default handling:

- Agreement bullish -> eligible for full `TQQQ` policy path.
- Agreement bearish -> eligible for bearish policy path.
- Disagreement -> reduce size or go `cash`.
- Full flips should require either agreement or a stronger override condition than current live logic uses.

### 4. Overbought Semantics

Decision: `Overbought means trim or stop adding, not automatic short.`

Meaning:

- `rsi_overbought` is a long-side throttle.
- It is not sufficient justification to rotate from `TQQQ` to `SQQQ`.

### 5. SQQQ Eligibility

Decision: `SQQQ must be governed by the same directional truth as its exit logic.`

Meaning:

- If the system allows `SQQQ` in `STRONG_BULL`, it must explain why the next morning risk manager would not instantly invalidate it.
- If morning risk logic rejects bearish exposure while QQQ is far above SMA-250, then SQQQ should not be entered on ordinary tactical shorts under those conditions.

### 6. Validation Standard

Decision: `Backtest must mirror the live state machine before performance claims are trusted.`

Required parity:

- end-of-day entries
- morning exits
- midday exits
- TQQQ/SQQQ rotation rules
- stale-position exits
- same transition rules as production

## Governance Contract

This policy should be encoded explicitly in configuration, not left as emergent behavior spread across modules.

Configuration should eventually express:

- whether regime is controlling or advisory
- whether models can override regime
- what disagreement means
- what overbought means
- what conditions allow a bull-to-bear flip
- whether inverse exposure is allowed in `BULL` / `STRONG_BULL`

## Decision Summary

Approved target posture:

- Regime controls side.
- Models control conviction.
- Disagreement reduces risk.
- Overbought trims, not flips.
- SQQQ is not allowed to function as a churn sleeve.
- No model retraining discussion should precede state-machine alignment.
