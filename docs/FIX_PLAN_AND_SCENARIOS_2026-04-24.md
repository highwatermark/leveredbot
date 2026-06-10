# Fix Plan And Scenario Baseline

Date: 2026-04-24
Status: Implemented policy baseline with initial live validation

Purpose:

- define the fix plan before code changes
- establish expected behavior in concrete `TQQQ` / `SQQQ` scenarios
- specify how policy should eventually be expressed in configuration
- create a baseline for implementation review and post-fix backtesting
- record the first post-fix live status readout against the paper account

## Implementation Status

Implemented defaults:

- `regime_authority = controlling`
- `prediction_model = both`
- `use_xgb_signal = True` when the environment supports it
- `model_primary = knn`
- `model_disagreement_action = reduce`
- `model_disagreement_adjustment = 0.75`
- `inverse_allowed_regimes = (RISK_OFF, BREAKDOWN)`
- `allow_inverse_in_bull = False`
- `allow_inverse_in_strong_bull = False`
- `allow_tqqq_to_sqqq_rotation = False`

Operational note:

- The runtime now computes an effective model signal from `k-NN` and `XGBoost`.
- Agreement keeps the directional thesis intact.
- One directional model plus one flat model reduces conviction through the disagreement haircut.
- Opposite directional calls neutralize the effective model to `FLAT`.
- If `xgboost` is unavailable in the environment, `XGBoost` stays neutral and the effective model falls back to the available signal.
- “Overbought means pause adds, not flip short” is now enforced behaviorally by keeping the RSI gate on `TQQQ` adds while disabling the bull-to-inverse rotation path.

Initial live validation after implementation (`job.py status`, 2026-04-24 19:57:57 EDT):

- live regime remained `STRONG_BULL`
- account still held legacy `SQQQ` from the prior logic
- `SQQQ Gate Checklist` now fails with:
  - `execution_window`
  - `regime_not_inverse`
- `SQQQ What-if` now resolves to:
  - target `0`
  - action `EXIT`
  - limiting factor `regime=STRONG_BULL`

That live readout matches the intended state machine: the system no longer wants fresh or persistent inverse exposure in a strong-bull regime.

Follow-up live validation after the TQQQ overbought fix (`job.py status`, 2026-04-24 20:05:58 EDT):

- live regime remained `STRONG_BULL`
- `TQQQ Gate Checklist` now only fails on `execution_window` outside the close window
- `TQQQ What-if` now resolves to:
  - target `328`
  - action `BUY`
  - limiting factor `rsi_overbought_pause_adds`
- `SQQQ What-if` still resolves to `EXIT`

That confirms the intended allocator behavior:

- strong bull -> express the book through `TQQQ`
- overbought -> reduce aggression, not suppress the long sleeve
- strong bull does not permit persistent `SQQQ`

Model-governance live validation after the effective-model pass (`job.py status`, 2026-04-24 21:09:05 EDT):

- raw `k-NN` remained `SHORT 0.72`
- raw `XGBoost` was neutral in the current environment
- effective model resolved to:
  - direction `SHORT`
  - source `knn_only`
  - disagreement `Yes`
  - adjustment `0.4411`
- `TQQQ What-if` still resolved to `BUY`, but with reduced size because:
  - `rsi_overbought_pause_adds`
  - `knn_adj=0.4411`
- `SQQQ What-if` still resolved to `EXIT`

That is the intended mixed-signal behavior in a strong bull regime:

- model disagreement reduces long conviction
- disagreement does not authorize inverse exposure

## Part 1: Fix Plan

### P0. Define Strategy Governance In Configuration

The policy contract should be explicit in config rather than implied across modules.

Proposed governance categories to encode:

- regime authority
  - `controlling` or `advisory`
- model role
  - `conviction_only`, `override_allowed`, `log_only`
- disagreement policy
  - `reduce`, `flat`, `require_agreement_for_flip`, `weighted`
- overbought policy
  - `trim`, `pause_adds`, `cash_reduction`, never `flip_short`
- inverse exposure policy
  - whether `SQQQ` is allowed in `BULL`
  - whether `SQQQ` is allowed in `STRONG_BULL`
  - override conditions if allowed

Reason:

- every other fix is fragile without an explicit governance contract

### P0. Halt Current SQQQ Churn Logic

Do not keep the present `SQQQ` behavior as-is while redesigning.

Reason:

- live evidence already shows repeated next-morning invalidation of entries

### P1. Rebuild Side-Selection State Machine

Required transition vocabulary:

- add
- hold
- reduce
- exit to cash
- rotate to inverse

Each transition must have a separate justification.

Reason:

- current system allows “do not add” to drift into “flip short”

### P1. Define Model Arbitration

Need explicit decision logic for:

- `k-NN` and `XGBoost` agreement
- disagreement
- low-confidence outcomes
- role of log-only models

Reason:

- current middle state is worst-of-both: one model has real influence, the other mostly logs

### P2. Rebuild Backtest To Mirror Live State Machine

Backtest must include:

- end-of-day entries
- morning exits
- midday exits
- stale-position exits
- TQQQ/SQQQ rotation
- same transition rules as production

Reason:

- current backtest is not representative

### P2. Add Model-Attribution Validation

Required slices:

- regime only
- regime + gates
- regime + k-NN
- regime + XGB
- agreement-only trading
- disagreement -> flat

Reason:

- cannot validate the model layer without isolating its contribution

## Part 2: Scenario Baseline

This section defines how the system should behave after the governance contract is adopted.

### Core Rule Set

Default posture:

- `STRONG_BULL` or `BULL` -> long bias through `TQQQ`
- `RISK_OFF` or `BREAKDOWN` -> no long bias; eligible for `cash` or approved bearish posture
- model disagreement -> less risk, not more
- overbought -> trim or stop adding, not inverse

### Daily Variables That Matter

Structural:

- `effective_regime`
- `qqq_close`
- `sma_50`
- `sma_250`
- `pct_above_sma50`
- `pct_above_sma250`

Tactical:

- `knn_direction`
- `knn_confidence`
- `xgb_direction`
- `xgb_confidence`
- model agreement/disagreement

Risk / filters:

- `momentum_score`
- `realized_vol`
- `vol_regime`
- `options_flow_ratio`
- `rsi_overbought`
- execution window

Portfolio state:

- current `TQQQ` shares
- current `SQQQ` shares
- current side

## Part 3: Scenario Matrix

### Scenario A: Strong Bull + Tactical Bull Agreement

Inputs:

- regime = `STRONG_BULL`
- k-NN = `LONG`
- XGB = `LONG` or neutral
- momentum high
- no blocking risk flags

Expected behavior:

- side = `TQQQ`
- action = add / hold `TQQQ`
- no `SQQQ`

### Scenario B: Strong Bull + Tactical Mixed / Disagreement

Inputs:

- regime = `STRONG_BULL`
- k-NN = `SHORT`
- XGB = `LONG` or `FLAT`

Expected behavior:

- no immediate `SQQQ`
- preferred actions:
  - reduce `TQQQ`
  - stop adding
  - or go partially / fully `cash`
- full inversion should require an approved override rule

### Scenario C: Strong Bull + Overbought

Inputs:

- regime = `STRONG_BULL`
- `rsi_overbought = True`

Expected behavior:

- do not add new `TQQQ`
- consider trim / smaller target size
- do not rotate to `SQQQ` based on overbought alone

### Scenario D: Bull + Tactical Bearish But Not Structural Breakdown

Inputs:

- regime = `BULL`
- k-NN = `SHORT`
- XGB mixed or bearish
- QQQ still above SMA-250

Expected behavior:

- either reduce `TQQQ` or go `cash`
- `SQQQ` should only be allowed if the bearish override policy is explicitly satisfied
- if `SQQQ` is allowed, its exit logic must be consistent with that policy

### Scenario E: Structural Breakdown + Tactical Bearish

Inputs:

- regime = `RISK_OFF` or `BREAKDOWN`
- tactical model bearish

Expected behavior:

- eligible for `SQQQ` or bearish expression if approved by design
- this is the natural home for a persistent inverse sleeve

### Scenario F: Structural Breakdown + Tactical Mixed

Inputs:

- regime = `RISK_OFF` / `BREAKDOWN`
- model disagreement

Expected behavior:

- reduced bearish size or `cash`
- avoid forced full-size inverse exposure during disagreement

### Scenario G: Existing SQQQ Position In Strong Bull

Inputs:

- current side = `SQQQ`
- regime = `STRONG_BULL`
- QQQ materially above SMA-250

Expected behavior:

- under the target design, this position should generally not have been initiated on an ordinary tactical short
- if already held, exit should follow the same approved policy that would have governed entry
- avoid a design where entry is authorized by one truth and invalidated by another the next morning

### Scenario H: Existing TQQQ Position + Tactical Bearish Surprise

Inputs:

- current side = `TQQQ`
- regime still bullish
- model turns `SHORT`

Expected behavior:

- first response: reduce, pause adds, or move to cash
- not immediate rotation to `SQQQ` unless override criteria are met

## Part 4: Expected State Behavior Summary

The target system should behave like this:

- bullish structure + bullish evidence -> hold `TQQQ`
- bullish structure + mixed evidence -> reduce risk, not invert
- bearish structure + bearish evidence -> hold `SQQQ`
- overbought -> trim or pause
- disagreement -> reduce conviction
- inverse exposure must not be entered by rules that the next risk layer is guaranteed to reject

## Part 5: Configuration Direction

Yes, this should be baked into configuration, but only for policy expression, not as a substitute for coherent logic.

Recommended config direction:

- policy booleans and enums for authority / arbitration / transition handling
- explicit side-selection mode
- explicit disagreement mode
- explicit overbought mode
- explicit inverse-eligibility mode

Not recommended:

- adding more one-off thresholds without first defining the policy contract

## Part 6: Approval Gate Before Implementation

Do not implement until the following are approved:

1. Is regime controlling?
2. What can models do when regime is bullish?
3. What does disagreement mean?
4. What does overbought mean?
5. When is `SQQQ` allowed?
6. What exact transitions are valid?

Once those are approved, implementation should follow the charter rather than invent policy in code.
