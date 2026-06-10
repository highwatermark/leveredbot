# Leveredbot System Design Review & Implementation Plan

*Generated: 2026-02-09*
*Scope: Holistic review of current system + planned k-NN evolution*

---

## 1. System Overview

### Current Architecture (Live)

A cron-based TQQQ position-trading system executing once daily at 3:50 PM EST.

```
Cron (3:50 PM) → job.py cmd_run
  ├── _fetch_all_data()     → Alpaca API (account, positions, calendar, snapshot, bars)
  ├── _compute_signals()    → Regime detection, momentum, vol, options flow
  ├── run_gate_checklist()  → 16 binary gates (incl. RSI overbought, k-NN disagreement)
  ├── calculate_target_shares() → 6-step sizing pipeline
  ├── execute_rebalance()   → Market order via Alpaca
  ├── log_*()               → SQLite persistence
  └── send_daily_report()   → Telegram notification
```

**Key parameters (updated after Phase 5):**
- Capital isolation: 30% of total account equity
- Regime: Binary mode (STRONG_BULL / BULL / RISK_OFF / BREAKDOWN) -- CAUTIOUS mapped to BULL
- Volatility: realized vol only (no VIX), 4 bands (LOW/NORMAL/HIGH/EXTREME)
- Options flow overlay: Unusual Whales put/call ratio on TQQQ
- Oscillation protection: 10-day minimum regime hold, 5% SMA deadzone
- RSI overbought gate: blocks new buys when RSI-14 > 70
- k-NN signal overlay: 10-feature next-day direction predictor (report-only mode, continuous conviction scoring)

### Live Performance (Aug 2024 - Feb 2026)

| Metric | Strategy | QQQ B&H | TQQQ B&H |
|--------|----------|---------|----------|
| Return | -0.4% | +28.4% | -26.0% |
| Max Drawdown | 6.3% | -- | -- |
| Trades | 306 | 0 | 0 |
| Regime Flips | 37 | -- | -- |

**Root causes of underperformance:**
1. **Whipsaw is the primary drag** -- 37 regime transitions in 18 months
2. **CAUTIOUS 25% positions bleed value** -- partial positions eat decay without capturing upside
3. **TQQQ volatility decay is structural** -- 3x leverage in choppy sideways markets destroys value
4. **Capital preservation works** -- 6.3% max DD vs TQQQ's raw -26% validates safety thesis

### Planned Evolution: k-NN Signal Integration

`leveretf_prompt.md` and `knnsignal.md` are **reference documents** describing the Adaptive Investments strategy (Collective2 #148705494: 130% annual, 2.17 Sharpe, 51.6% win rate). They contain hypothetical code and aspirational architecture -- not a concrete system to build as-is.

The useful concepts to evolve into the current system:

| Concept from Reference | How It Fits the Current System |
|------------------------|-------------------------------|
| k-NN next-day prediction (16 features) | Additional signal in `_compute_signals()`, runs at 3:50 PM alongside SMA regime |
| Confidence-based sizing (0.55 threshold) | Modulates position sizing like options flow does today |
| Feature engineering (price, vol, trend, momentum, volume, VIX) | Calculated from data the system already fetches (QQQ bars, snapshots) |
| Entry at 15:59 | Current system already runs at 15:50 -- compatible timing |
| Windowed exits (morning stops, midday catastrophic) | Future work -- requires additional cron jobs or polling, not needed for v1 |
| SQQQ short positions | Future consideration after k-NN signal is validated on paper |

**Key insight:** k-NN prediction is fundamentally an EOD calculation. It fits naturally into the current cron architecture as a signal layer, not a replacement architecture. The windowed position management from the reference is a separate evolution that can be added incrementally later.

---

## 2. Code Quality Issues (16 Agreed Changes)

### 2A. Architecture Changes

#### ARCH-1: Dataclasses for Pipeline Data (Conviction: 95%)

**Problem:** `job.py` manually copies fields between 4 raw dicts (`data`, `signals`, `gate_data`, `sizing_data`) across 1060 lines. Field mismatches are undetectable until runtime.

**Solution:** Replace raw dicts with typed dataclasses.

```python
@dataclass
class MarketData:
    account: dict
    positions: list
    tqqq_position: dict | None
    qqq_closes: list[float]
    qqq_current: float | None
    tqqq_price: float | None
    daily_loss_pct: float
    trading_days_fetched: int
    is_half_day: bool
    calendar: dict | None

@dataclass
class StrategySignals:
    regime: str
    raw_regime: str
    previous_regime: str | None
    momentum_score: float
    realized_vol: float
    vol_regime: str
    options_flow_adjustment: float
    options_flow_bearish: bool
    options_flow_ratio: float
    allocated_capital: float
    current_shares: int
    tqqq_price: float
    qqq_close: float
    sma_50: float
    sma_250: float
    qqq_closes: list[float]
    daily_loss_pct: float
    day_trades_remaining: int
    trading_days_fetched: int
```

**Files:** `job.py` (refactor `_fetch_all_data` and `_compute_signals` return types), `strategy/sizing.py` (accept dataclass instead of dict)

#### ARCH-2: Single DB Connection Per Pipeline Run (Conviction: 95%)

**Problem:** `db/models.py` opens and closes a new SQLite connection for each of 14 functions. In a single `cmd_run`, this creates 5-8 separate connections.

**Solution:** Pass a single connection through the pipeline, close once at the end.

**Files:** `db/models.py` (add `conn` parameter to all functions), `job.py` (create connection at start, pass through, close in finally)

#### ARCH-3: Singleton Alpaca Clients (Conviction: 90%)

**Problem:** `alpaca_client.py` creates new `TradingClient` and `StockHistoricalDataClient` instances on every API call via `_get_trading_client()` and `_get_data_client()`.

**Solution:** Module-level lazy singletons.

```python
_trading_client = None
_data_client = None

def _get_trading_client():
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
    return _trading_client
```

**Files:** `alpaca_client.py`

#### ARCH-4: Atomic Transaction for Trade Pipeline (Conviction: 75%)

**Problem:** If the process crashes between `execute_rebalance()` and `log_daily_decision()`, the trade executes but isn't logged. No recovery mechanism.

**Solution:** Wrap the decision-execution-logging sequence in a DB transaction. Log intent before execution, update with result after.

```python
# 1. Log intent (status=PENDING)
decision_id = log_daily_decision({...status: "PENDING"})
# 2. Execute
result = execute_rebalance(...)
# 3. Update with result (status=EXECUTED or FAILED)
update_decision(decision_id, result)
# All in a single DB transaction (committed after step 3)
```

**Files:** `job.py` (cmd_run), `db/models.py` (add update_decision, transaction support)

---

### 2B. Code Quality Changes

#### CQ-1: DB Context Manager (Conviction: 95%)

**Problem:** Connection open/close boilerplate repeated 14 times in `db/models.py`.

**Solution:**
```python
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
```

**Files:** `db/models.py`

#### CQ-2: Extract Gate/Sizing Data Builders (Conviction: 90%)

**Problem:** `cmd_run()` and `cmd_status()` both manually construct identical `gate_data` and `sizing_data` dicts (20+ lines duplicated).

**Solution:** Extract `build_gate_data(signals)` and `build_sizing_data(signals)` functions.

**Files:** `job.py` (extract functions, call from both cmd_run and cmd_status)

#### CQ-3: Resilient Pregame Polling (Conviction: 85%)

**Problem:** `cmd_pregame()` has a 20-minute polling loop (`time.sleep(300)` x 4 polls) with no error recovery. If any poll crashes, the entire pregame aborts.

**Solution:** Wrap each poll iteration in try/except, continue on failure, log which polls succeeded.

**Files:** `job.py` (cmd_pregame polling loop)

#### CQ-4: Classify Retryable Exceptions (Conviction: 90%)

**Problem:** `_retry()` in `alpaca_client.py` catches bare `Exception`, retrying non-retryable errors like `ValueError` (bad data) or `AuthenticationError`.

**Solution:**
```python
RETRYABLE = (ConnectionError, TimeoutError, httpx.HTTPStatusError)

def _retry(fn, label, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            return fn()
        except RETRYABLE as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
        # Non-retryable exceptions propagate immediately
```

**Files:** `alpaca_client.py`

---

### 2C. Test Improvements

#### TEST-1: Integration Test for Full cmd_run Pipeline (Conviction: 90%)

**Problem:** No test exercises the complete `cmd_run` pipeline end-to-end. Individual units are tested but integration points (data flow between stages, dict key consistency) are not.

**Solution:** Mock Alpaca and UW APIs at the HTTP level, run `cmd_run()`, verify: DB has a decision logged, correct regime detected, correct order submitted (or not).

**Files:** `tests/test_integration.py` (new)

#### TEST-2: Test Real Backtest Code (Conviction: 85%)

**Problem:** `test_backtest.py` reimplements a parallel `_simulate_backtest()` function instead of testing the actual `cmd_backtest()` in `job.py`. Tests pass but don't validate production code.

**Solution:** Replace with tests that call `cmd_backtest()` with mocked Alpaca data, then query the backtest_results DB table.

**Files:** `tests/test_backtest.py` (rewrite)

#### TEST-3: Notification Tests (Conviction: 80%)

**Problem:** `notifications.py` (224 lines) has zero test coverage. Division by zero risk at line 211 (`days_in_market/total_days*100`).

**Solution:** Test each notification function with mock httpx, verify message formatting, test edge cases (zero values, missing fields).

**Files:** `tests/test_notifications.py` (new)

#### TEST-4: Pregame Override Bug Fix + Tests (Conviction: 95%)

**Problem (BUG):** `job.py:465` hardcodes `options_flow_adjustment = 0.5` when pregame detects bearish flow. But `config.py` sets `options_flow_reduction_pct = 0.25`, meaning the adjustment factor should be `1.0 - 0.25 = 0.75`. Pregame bearish flow causes a **50%** position reduction while live bearish flow causes only **25%** reduction.

**Fix:**
```python
# job.py line 465, change:
signals["options_flow_adjustment"] = 0.5
# to:
signals["options_flow_adjustment"] = 1.0 - LEVERAGE_CONFIG["options_flow_reduction_pct"]
```

**Files:** `job.py` (line 465), `tests/test_pregame.py` (new, verify pregame overrides use config values)

---

### 2D. Performance Improvements

#### PERF-1: Parallel API Calls in _fetch_all_data (Conviction: 90%)

**Problem:** 5 sequential API calls (`get_account`, `get_positions`, `get_calendar`, `get_snapshot`, `get_bars_with_cache`) take ~3-5 seconds total.

**Solution:** `get_account`, `get_positions`, and `get_calendar` are independent -- run them concurrently with `concurrent.futures.ThreadPoolExecutor`.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=3) as pool:
    f_account = pool.submit(alpaca_client.get_account)
    f_positions = pool.submit(alpaca_client.get_positions)
    f_calendar = pool.submit(alpaca_client.get_calendar, today_str)
    account = f_account.result()
    positions = f_positions.result()
    calendar = f_calendar.result()
```

**Files:** `job.py` (_fetch_all_data)

#### PERF-2: SQL Date Filter for Bar Cache (Conviction: 85%)

**Problem:** `get_bars_with_cache()` at `db/cache.py:158` fetches `days * 2` bars from Alpaca then filters in Python.

**Solution:** Use SQL `WHERE date >= ?` to fetch only the needed range, and only request missing dates from Alpaca.

**Files:** `db/cache.py`

#### PERF-3: Batched DB Reads in _compute_signals (Conviction: 80%)

**Problem:** `_compute_signals` makes 3 separate DB reads (`get_last_regime`, `get_regime_duration_days`, `get_consecutive_losing_days`), each opening and closing its own connection.

**Solution:** Single query returning all three values.

```python
def get_strategy_state(conn) -> dict:
    """Single query for regime, duration, and losing days."""
    row = conn.execute("""
        SELECT regime,
               julianday('now') - julianday(date) as duration,
               ...
        FROM regimes ORDER BY date DESC LIMIT 1
    """).fetchone()
    ...
```

**Files:** `db/models.py`, `job.py` (_compute_signals)

#### PERF-4: Interruptible Pregame Polling (Conviction: 75%)

**Problem:** `cmd_pregame` uses `time.sleep(300)` with no way to interrupt or signal the process.

**Solution:** Use `threading.Event.wait(timeout=300)` so the process can be cleanly interrupted.

**Files:** `job.py` (cmd_pregame)

---

## 3. System Design Gap Analysis

### 3A. Current System vs. Expansion Phase (Tier 1 Changes)

The EXPANSION_PHASE.md documents concrete improvements that showed +1.3% return in backtests (vs -0.4% current):

| Change | Status | Impact |
|--------|--------|--------|
| Binary mode (remove CAUTIOUS) | Not implemented | Reduces decay drag |
| Wider deadzone (2% -> 5-7%) | Not implemented | Reduces flips from 37 to ~15 |
| Longer hold period (2d -> 10-15d) | Not implemented | Prevents oscillation |
| RSI overbought filter (>70-75) | Not implemented | Prevents buying tops |

**Recommendation:** These should be implemented as config toggles (feature flags in `LEVERAGE_CONFIG`), not structural changes. The current architecture supports all four via parameter changes:

```python
# In LEVERAGE_CONFIG, add:
"use_binary_mode": True,        # Remove CAUTIOUS regime
"sma_deadzone_pct": 0.05,       # Widen from 0.02
"min_regime_hold_days": 10,     # Increase from 2
"rsi_overbought_threshold": 70, # New gate
```

**Files to change:** `config.py`, `strategy/regime.py` (binary mode toggle), `strategy/sizing.py` (RSI gate)

### 3B. Evolving k-NN Concepts into the Current System

`knnsignal.md` is reference code -- not production code to be integrated directly. The concepts that matter for evolution:

**What to adopt (fits current architecture):**

| Concept | Integration Point | Effort |
|---------|-------------------|--------|
| 16-feature vector (MarketFeatures) | New `strategy/knn_signal.py` module | Medium |
| k-NN prediction (KNNSignalGenerator) | Called in `_compute_signals()` at 3:50 PM | Medium |
| Confidence-based sizing | New field in sizing pipeline (like `options_flow_adjustment`) | Low |
| Feature calculation from daily bars | `FeatureCalculator` adapted to use existing QQQ bar cache | Medium |
| Model training on historical data | Run once on startup or cache fitted model to disk | Low |

**What to defer (requires architectural changes):**

| Concept | Why Defer | Prerequisite |
|---------|-----------|-------------|
| Continuous polling / windowed exits | Needs always-on process, not cron | Validate k-NN signal accuracy first |
| SQQQ short positions | Different risk profile, capital management | Proven directional accuracy on paper |
| Partial profit taking / trailing stops | Intraday monitoring required | Morning cron job or polling process |
| 95% capital allocation | Dramatically higher risk | Extended paper trading validation |
| Position scaling (scale-ins) | Intraday re-entry logic | Windowed exit infrastructure |

**How k-NN slots into the current pipeline:**

```
Current:  _compute_signals() → regime + momentum + vol + flow → sizing
                                                                  ↓
Evolved:  _compute_signals() → regime + momentum + vol + flow + k-NN signal → sizing
                                                                               ↓
          k-NN confidence modulates target_pct (like vol_adjustment does today)
          k-NN disagreement with regime triggers caution (reduce allocation)
          k-NN agreement with regime boosts conviction (allow full allocation)
```

**Concrete integration design:**

```python
# strategy/knn_signal.py (new module)
class KNNSignal:
    """k-NN direction prediction as a signal overlay."""

    def __init__(self, n_neighbors=7, min_confidence=0.55):
        ...

    def fit_from_bars(self, qqq_bars: list[dict]) -> None:
        """Train from the same bar cache the system already uses."""
        ...

    def predict(self, qqq_bars: list[dict]) -> dict:
        """Returns {direction: 'LONG'|'SHORT'|'FLAT', confidence: float, adjustment: float}"""
        ...

# In _compute_signals(), add:
knn = KNNSignal()
knn.fit_from_bars(closes)  # Or load cached model
knn_result = knn.predict(closes)
signals["knn_direction"] = knn_result["direction"]
signals["knn_confidence"] = knn_result["confidence"]
signals["knn_adjustment"] = knn_result["adjustment"]

# In calculate_target_shares(), add:
# If k-NN disagrees with regime direction, reduce by 50%
# If k-NN agrees with high confidence (>0.65), no reduction
# If k-NN is FLAT (low confidence), reduce by 25%
```

**New gate for sizing.py:**

```python
# Gate: knn_disagreement
# If regime says STRONG_BULL but k-NN predicts SHORT with >0.6 confidence → block entry
```

### 3C. Reference Code Concepts Worth Extracting

From `knnsignal.md`, the following patterns are worth adapting (not copying):

| Pattern | Adapt For |
|---------|-----------|
| `MarketFeatures` dataclass with `to_array()` | Feature vector construction from existing bar data |
| `FeatureCalculator._calculate_indicators()` | Reuse RSI, ATR, MA distance calculations (some already exist in `signals.py`) |
| Distance-weighted k-NN with sklearn | Core prediction engine -- keep simple, avoid over-engineering |
| `analyze_prediction()` similar-days analysis | Useful for Telegram reporting (show what similar days did) |
| `build_training_data()` loop from day 200 | Training data builder -- adapt to use bar cache |

**Not worth adopting:**
- `AlpacaSignalProvider`, `AlpacaMarketData`, `AlpacaOrderExecutor` -- the current system already has `alpaca_client.py`
- `create_leveraged_etf_system()` factory -- over-engineered for cron-based execution
- Async patterns -- current system is synchronous, no benefit to async for a cron job

---

## 4. Implementation Order

### Phase 1: Bug Fix + Quick Wins -- COMPLETE

| # | Change | Status |
|---|--------|--------|
| 1 | TEST-4: Fix pregame 0.5 bug → uses `1.0 - LEVERAGE_CONFIG["options_flow_reduction_pct"]` | DONE |
| 2 | CQ-4: Classify retryable exceptions → `RETRYABLE_EXCEPTIONS` tuple | DONE |
| 3 | ARCH-3: Singleton Alpaca clients → lazy module-level globals | DONE |
| 4 | CQ-1: DB context manager → `get_db()` + refactored 11 functions | DONE |
| -- | New test: `tests/test_pregame.py` (7 tests) | DONE |
| -- | All 82 tests passing | VERIFIED |

### Phase 2: Infrastructure Cleanup -- COMPLETE

| # | Change | Status |
|---|--------|--------|
| 5 | ARCH-2: Single DB connection → `cmd_run` passes `conn` through full pipeline | DONE |
| 6 | PERF-3: Batched DB reads → `get_strategy_state()` replaces 3 separate calls | DONE |
| 7 | CQ-2: Extract gate/sizing builders → `_build_gate_data()`, `_build_sizing_data()` | DONE |
| 8 | PERF-1: Parallel API calls → `ThreadPoolExecutor` in `_fetch_all_data()` | DONE |
| 9 | PERF-2: SQL date filter for cache → `get_cached_bars()` uses `WHERE date >= ?` | DONE |
| -- | All 134 tests passing (52 new since Phase 1) | VERIFIED |

### Phase 3: Resilience + Tests -- COMPLETE

| # | Change | Status |
|---|--------|--------|
| 10 | CQ-3: Resilient pregame polling → try/except per poll, neutral fallback on all-fail | DONE |
| 11 | PERF-4: Interruptible pregame sleep → 10s intervals instead of 300s block | DONE |
| 12 | TEST-3: Notification tests → 16 tests in `test_notifications.py`, fixed div-by-zero bug | DONE |
| 13 | TEST-1: Integration test for cmd_run → 4 tests in `test_integration.py` (full pipeline, closed market, insufficient data, halfday) | DONE |
| 14 | TEST-2: Test real backtest code → `TestCmdBacktestIntegration` exercises actual `cmd_backtest()` | DONE |
| -- | All 155 tests passing | VERIFIED |

### Phase 4: Architecture -- COMPLETE

| # | Change | Status |
|---|--------|--------|
| 15 | ARCH-1: Dataclasses for pipeline data → `MarketData` and `StrategySignals` replace raw dicts, attribute access throughout | DONE |
| 16 | ARCH-4: Atomic transaction for trade pipeline → intent-first logging (PENDING→EXECUTED/FAILED), `update_decision()` | DONE |
| -- | All 155 tests passing | VERIFIED |

### Phase 5: Strategy Evolution -- COMPLETE

| # | Change | Status |
|---|--------|--------|
| 17 | Expansion Tier 1: Binary mode toggle → `use_binary_mode: True` maps CAUTIOUS→BULL | DONE |
| 18 | Expansion Tier 1: Wider deadzone (5%) + longer hold (10d) | DONE |
| 19 | Expansion Tier 1: RSI overbought gate → Gate 15 (`rsi_overbought_threshold: 70`) | DONE |
| 20 | k-NN signal module → `strategy/knn_signal.py` with 10-feature FeatureCalculator, KNNSignal class, model persistence | DONE |
| 21 | k-NN integration → wired into `_compute_signals`, Gate 16 (knn_disagreement), sizing Step 7, DB logging, status display | DONE |
| 22 | k-NN backtest validation → walk-forward split, accuracy/regime agreement tracking in `cmd_backtest` | DONE |
| 23 | k-NN report-only mode → `knn_report_only: True` (default), Telegram daily report shows k-NN signal | DONE |
| 24 | k-NN conviction scoring → continuous `_conviction_adjustment()`: SHORT 0.65→0.75 to 0.80→0.40 linear scaling | DONE |
| -- | All 185 tests passing | VERIFIED |

---

## 5. k-NN Evolution Roadmap (Integrated into Current System)

The k-NN concepts from `knnsignal.md` and `leveretf_prompt.md` evolve into the current system incrementally. No separate service needed.

### Phase 5a: k-NN as Signal Overlay (first)

Add k-NN prediction as another signal in `_compute_signals()`, alongside SMA regime, momentum, vol, and flow. It modulates sizing but does **not** override regime decisions.

**New files:**
- `strategy/knn_signal.py` -- KNNSignal class, FeatureCalculator, model persistence
- `tests/test_knn_signal.py` -- Unit tests for prediction, feature calc, edge cases

**Changes to existing files:**
- `config.py` -- Add k-NN config: `use_knn_signal`, `knn_neighbors`, `knn_min_confidence`, `knn_lookback_years`, `knn_disagreement_reduction`
- `job.py` (`_compute_signals`) -- Call `knn.predict()`, add `knn_direction`, `knn_confidence`, `knn_adjustment` to signals
- `strategy/sizing.py` (`calculate_target_shares`) -- Apply `knn_adjustment` to `target_pct` (same pattern as `options_flow_adjustment`)
- `strategy/sizing.py` (`run_gate_checklist`) -- Add `knn_disagreement` gate (k-NN predicts opposite direction with high confidence)

**Integration logic:**

```
SMA regime says STRONG_BULL + k-NN says LONG (>0.55 confidence)
  → Full allocation (agreement boosts conviction)

SMA regime says STRONG_BULL + k-NN says FLAT (<0.55 confidence)
  → Reduce by 25% (low conviction, proceed cautiously)

SMA regime says STRONG_BULL + k-NN says SHORT (>0.60 confidence)
  → Block entry via gate OR reduce by 50% (disagreement = caution)

SMA regime says RISK_OFF + k-NN says anything
  → RISK_OFF always wins (safety override unchanged)
```

**Feature subset for v1** (10 of the 16 from reference -- skip VIX and volume initially):
1. `intraday_return` -- from snapshot (open to current)
2. `prior_day_return` -- from bar cache
3. `two_day_return` -- from bar cache
4. `five_day_return` -- from bar cache
5. `intraday_range` -- from snapshot (high - low) / open
6. `distance_from_20ma` -- calculated from bar cache (new indicator)
7. `distance_from_50ma` -- already have SMA-50
8. `distance_from_200ma` -- already have SMA-250 (close enough)
9. `rsi_14` -- new calculation from bar cache
10. `momentum_10` -- similar to existing ROC-20, adapt

**Model persistence:** Pickle the fitted sklearn model + scaler to `data/knn_model.pkl`. Retrain weekly (Sunday cron) or when bar cache updates significantly.

**Validation before going live:**
- Backtest k-NN signal accuracy against historical QQQ bars (already cached)
- Compare: regime-only decisions vs regime+k-NN decisions
- Paper trade for 2 weeks with k-NN in **report-only mode** (logged to Telegram, no sizing impact)

### Phase 5b: k-NN Conviction Scoring

After validating signal accuracy, evolve from binary (agree/disagree) to continuous conviction:

- k-NN confidence 0.55-0.65 → low conviction → 75% of regime allocation
- k-NN confidence 0.65-0.75 → moderate conviction → 100% of regime allocation
- k-NN confidence 0.75+ → high conviction → 100% + report "high conviction" flag
- k-NN confidence <0.55 → FLAT signal → reduce allocation by 25%

This replaces the stepped logic with continuous scaling, similar to how momentum_score already scales between `min_position_pct` and `max_position_pct`.

### Phase 5c: Enhanced Features (after paper validation)

Add remaining features from reference if they improve accuracy:
- `volume_ratio` (today vs 20-day avg) -- needs volume data in bar cache
- `vix_level` + `vix_change` -- needs VIX data source (Alpaca or CBOE)
- `atr_ratio` -- ATR calculation from bar cache
- `ma_20_50_cross` -- 20MA-50MA spread (crossover signal)
- `rsi_deviation` -- RSI distance from 50 (already have RSI from 5b)

### Phase 5d: Windowed Exits (future, requires arch change)

Only pursue after k-NN signal is validated and consistently profitable:
- Add morning cron job (9:35 AM) for stop-loss evaluation
- Add midday cron job (12:00 PM) for catastrophic stop only
- Current 3:50 PM cron remains the entry/rebalance window
- This is the bridge toward the continuous polling model from the reference

**Not planned:**
- SQQQ short positions (validate TQQQ-long accuracy first)
- 95% capital allocation (keep 30% isolation until extended track record)
- Continuous polling loop (cron jobs provide adequate coverage for now)

---

## 6. Risk Assessment

### Current System Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Whipsaw in choppy markets | HIGH (proven) | Tier 1 changes (binary, wider dz, RSI) |
| Pregame bug (0.5 vs 0.75) | MEDIUM | Fix immediately (TEST-4) |
| Non-atomic trade+log | MEDIUM | ARCH-4 |
| No integration tests | MEDIUM | TEST-1 |
| Notification failures silent | LOW | TEST-3 |

### k-NN Integration Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting on historical data | HIGH | Walk-forward validation, out-of-sample testing, start report-only |
| k-NN conflicts with regime signal | MEDIUM | Regime always wins on safety (RISK_OFF/BREAKDOWN), k-NN only modulates sizing |
| sklearn dependency in cron job | LOW | Model cached to disk, retrain weekly, fallback to neutral if load fails |
| Feature calculation adds latency | LOW | Features computed from bar cache (already in memory), no extra API calls |
| Insufficient training data (Alpaca free tier ~2.5 yr) | MEDIUM | Start with available data, upgrade data source if accuracy warrants it |

---

## 7. Success Metrics

### Current System (after Phases 1-5)

- Regime flips reduced (wider deadzone 5%, 10-day hold period, binary mode)
- Max drawdown bounded (RSI overbought gate, k-NN disagreement gate)
- Zero unlogged trades (ARCH-4 atomic transaction)
- **185 tests passing** (up from ~60 at start)

### k-NN Signal Integration (paper trading)

- k-NN directional accuracy > 52% on out-of-sample data
- k-NN agreement with regime on >60% of trading days
- Regime+k-NN combined return beats regime-only return in backtest
- Report-only mode runs for 2+ weeks without errors
- No increase in max drawdown vs regime-only system

---

## 8. Files Changed Summary

| File | Changes |
|------|---------|
| `config.py` | Binary mode, RSI overbought, wider deadzone (5%), longer hold (10d), k-NN configs |
| `job.py` | Dataclasses (`MarketData`/`StrategySignals`), parallel API, pregame resilience, bug fix L465, extract builders, k-NN signal integration, k-NN backtest validation, atomic transaction |
| `alpaca_client.py` | Singleton clients, classify retries |
| `db/models.py` | Context manager, batched reads, single conn, atomic transactions, k-NN columns + migration |
| `db/cache.py` | SQL date filter |
| `pipeline_types.py` | NEW: `MarketData` and `StrategySignals` dataclasses (incl. k-NN fields) |
| `strategy/sizing.py` | RSI gate (15), k-NN disagreement gate (16), k-NN sizing adjustment (Step 7) |
| `strategy/regime.py` | Binary mode toggle (`use_binary_mode`) |
| `strategy/signals.py` | `calculate_rsi()`, `check_rsi_overbought()` |
| `strategy/knn_signal.py` | NEW: KNNSignal, FeatureCalculator, continuous conviction scoring, model persistence |
| `notifications.py` | k-NN signal in daily Telegram report |
| `tests/test_integration.py` | NEW: full pipeline test (4 tests) |
| `tests/test_notifications.py` | NEW: notification tests (16 tests) |
| `tests/test_pregame.py` | NEW: pregame override tests (7 tests) |
| `tests/test_knn_signal.py` | NEW: k-NN prediction, features, conviction scoring (23 tests) |
| `tests/test_backtest.py` | REWRITE: test real backtest code |
| `tests/test_regime.py` | Binary mode test, CAUTIOUS tests updated |
| `tests/test_signals.py` | RSI tests (5 tests) |
| `tests/test_sizing.py` | RSI gate test, adjusted gate data seed |
