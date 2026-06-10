# Leveraged ETF Strategy Review Memo

Date: 2026-04-24
Repo state reviewed: `b9644a6`

## Executive Summary

The strategy is intended to behave as a two-sided leveraged allocator:

- bullish thesis -> `TQQQ`
- bearish thesis -> `SQQQ`
- mixed thesis -> reduced size or `cash`
- hold the selected side while the thesis persists
- switch only when the thesis clearly changes

The live system is not behaving that way. It is mixing:

- a medium-horizon regime engine
- a short-horizon prediction model
- a separate intraday risk manager

These layers do not share a unified directional truth. The most visible failure mode is repeated `SQQQ` entry near the close during `BULL` / `STRONG_BULL`, followed by next-morning `REGIME_EMERGENCY` exits.

## Intended State Machine

The intended operating model should be:

1. Determine structural regime.
2. Determine tactical conviction.
3. Select the side that matches the approved thesis.
4. Hold that side while the thesis persists.
5. Reduce, flatten, or flip only when explicit transition rules are met.
6. Validate the exact live state machine in backtest.

Clean intended transitions:

- structural bull + tactical bull -> `TQQQ`
- structural bull + tactical mixed -> reduced `TQQQ` or `cash`
- structural bear + tactical bear -> `SQQQ`
- disagreement -> less risk, not more
- intraday exits protect the active thesis rather than mechanically contradicting it

## Actual State Machine Observed

The live strategy currently behaves more like this:

1. Regime says `BULL` / `STRONG_BULL`.
2. `k-NN` can say `SHORT`.
3. `TQQQ` may be sold or blocked.
4. `SQQQ` can be entered despite bullish regime.
5. Morning position manager exits `SQQQ` if QQQ is far above SMA-250.
6. The process repeats.

Observed recent churn sequence:

- `2026-04-23 15:50` -> buy `SQQQ`
- `2026-04-24 09:35` -> exit `SQQQ` via `REGIME_EMERGENCY`
- `2026-04-24 15:50` -> buy `SQQQ` again

This is not persistent bearish exposure. It is overnight churn.

## Key Findings

### 1. Regime Layer

The regime layer is not the primary defect. In April it correctly identifies a persistent bullish tape:

- QQQ materially above SMA-50
- QQQ materially above SMA-250
- high momentum

The defect is that regime is not clearly controlling downstream side selection.

### 2. Model Layer

`k-NN` is not just reporting.

It is report-only for the normal `TQQQ` sizing path, but it still drives:

- `SQQQ` entry
- `TQQQ -> SQQQ` rotation

`XGBoost` is largely logging-only in the current live configuration.

The core mismatch:

- model target = next-day `QQQ` up/down
- traded action = leveraged ETF side selection plus intraday exits

That is a target/action mismatch even if raw classification is acceptable.

### 3. TQQQ Logic

The `TQQQ` sleeve is not operating as a persistent trend sleeve.

Example:

- `2026-04-15`: `TQQQ` sold while regime was `STRONG_BULL`
- gates failed mainly on `rsi_overbought`
- tactical short signal still triggered rotation

That means “do not add” is drifting into “become short,” which is too large a semantic jump.

### 4. SQQQ Logic

This is the current failure center.

Entry truth:

- tactical bearishness

Exit truth:

- long-term bullishness invalidates bearish thesis

Those are incompatible as currently wired. The entry layer authorizes positions the morning risk layer is structurally prepared to reject.

### 5. Backtest Validity

The backtest is not yet sufficient to validate live behavior because the live strategy includes:

- morning exits
- midday exits
- stale exits
- TQQQ/SQQQ rotations
- leveraged ETF path dependence

Until the backtest reproduces these transitions, it is not approval-grade evidence.

## Reviewer Verdict Matrix

### 1. Regime

- Quant Analyst: not the broken part; being overruled
- Researcher: override policy not researched
- Fund Manager: should dominate side selection
- Architect / Backtest: clean code; contract undefined

### 2. Models

- Quant Analyst: target/action mismatch
- Researcher: under-researched versus influence
- Fund Manager: disagreement should lower risk, not flip
- Architect / Backtest: no governance; primary/veto/log roles unclear

### 3. TQQQ

- Quant Analyst: not acting like a persistent trend sleeve
- Researcher: “do not add” is drifting into “go short”
- Fund Manager: too sensitive to tactical overlays
- Architect / Backtest: transition states ambiguous

### 4. SQQQ

- Quant Analyst: churn sleeve, not investable
- Researcher: falsified by live evidence
- Fund Manager: would not allow live in current form
- Architect / Backtest: entry and exit are based on different truths

### 5. Backtest

- Quant Analyst: not a true PnL estimator
- Researcher: research/production mismatch
- Fund Manager: incomplete diagnostic
- Architect / Backtest: insufficient for approval

## Root Causes

1. No explicit precedence ladder between regime and models.
2. `SQQQ` entry and exit are governed by different theses.
3. Backtest does not reproduce the live state machine.

## Open Decision Questions

1. Is regime advisory or controlling?
2. Under what exact conditions may a short-horizon model override a strong-bullish regime?
3. When models disagree: reduce conviction, go flat, require agreement for flips, or weight by recent accuracy?
4. Does “overbought” map to trim, stop adding, or flip short?
5. Does the backtest reproduce overnight `SQQQ` churn? If not, it is not representative.

## Consensus Decision

The four-role consensus is:

- regime should be controlling for default side selection
- models should govern conviction, not silently invert the book
- disagreement should reduce risk
- overbought should trim, not flip
- no model retraining discussion should precede state-machine alignment
