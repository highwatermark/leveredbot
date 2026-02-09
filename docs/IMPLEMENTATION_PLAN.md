# Leveraged ETF Strategy — Implementation Plan

## Context

Build a fully automated leveraged ETF position-trading system that trades TQQQ (3x Bull NASDAQ) and goes to cash during Risk-Off periods. Based on the detailed specification at `/home/ubuntu/momentum-agent/leveretf/leveraged_etf_build_prompt.md`.

This is a **standalone repo** (not part of v1 or v2), sharing only the Alpaca paper account. Runs as a daily cron job at 3:50 PM EST (12:45 PM on half days).

---

## Decisions Made (from review)

| # | Issue | Decision |
|---|-------|----------|
| 1 | Repo coupling | Fully standalone. Queries Alpaca API for capital isolation. |
| 2 | Execution model | Cron-based. Morning status cron as watchdog. |
| 3 | UW flow gate | Implement with robust fallback to neutral. |
| 4 | PDT tracking | Use Alpaca account-level `daytrade_count`. |
| 5 | Capital isolation DRY | Single `get_allocated_capital()` function. |
| 6 | File structure | Modular: regime, signals, sizing, executor, models, notifications, config, job. |
| 7 | Regime edge cases | All four: cold start (CAUTIOUS), oscillation (2-day hold), gap alerts, staleness. |
| 8 | Backtest scope | Full backtest with P&L simulation, fills, benchmarks, CSV export. |
| 9 | Test strategy | Comprehensive 60+ tests across all modules. |
| 10 | Backtest assertions | Concrete pytest assertions (cash before crashes, max DD <45%). |
| 11 | API mocking | Record/replay — record actual API responses, replay in tests. |
| 12 | Failure modes | All 9 failure modes tested. |
| 13 | API calls | Cache bars locally for review/backtest; daily job fetches only latest day. |
| 14 | Database | Raw SQL with `executemany()` for batch backtest writes. |
| 15 | Backtest engine | SQLite-based — store intermediate results in DB for queryability. |
| 16 | Telegram output | Summary message + CSV file for full backtest results. |

---

## Repo Structure

```
~/leveraged-etf/
├── .env.example
├── .gitignore
├── pyproject.toml
│
├── config.py                 # LEVERAGE_CONFIG dict + env loading
│
├── strategy/
│   ├── __init__.py
│   ├── regime.py             # Regime detection (SMA, deadzone, state machine)
│   ├── signals.py            # Momentum scoring, realized vol, options flow sentiment
│   ├── sizing.py             # Position sizing, capital isolation, gate checklist
│   └── executor.py           # Order submission, fill handling, force exit
│
├── db/
│   ├── __init__.py
│   ├── models.py             # Schema creation, raw SQL helpers
│   └── cache.py              # Local bar cache (QQQ/TQQQ daily bars)
│
├── notifications.py          # Telegram message formatting and sending
├── alpaca_client.py          # Alpaca API wrapper (trading + data)
├── uw_client.py              # Unusual Whales API wrapper (TQQQ flow)
│
├── job.py                    # CLI entry point: run, status, backtest, force_exit
│
├── data/                     # SQLite DB + CSV exports (gitignored)
│   └── .gitkeep
├── logs/                     # Log files (gitignored)
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures, DB setup
│   ├── fixtures/             # Recorded API responses
│   │   ├── qqq_bars_bull.json
│   │   ├── qqq_bars_bear.json
│   │   ├── account_with_positions.json
│   │   ├── tqqq_snapshot.json
│   │   └── calendar_halfday.json
│   ├── test_regime.py        # Regime detection tests
│   ├── test_signals.py       # Momentum, vol, flow tests
│   ├── test_sizing.py        # Position sizing + gate checklist tests
│   ├── test_executor.py      # Order submission tests
│   ├── test_backtest.py      # Backtest assertions (crash exits, drawdown)
│   ├── test_cache.py         # Bar caching tests
│   └── test_failures.py      # All 9 failure mode tests
```

---

## Implementation Phases

### Phase 1: Foundation (config, DB, clients) — [ ] Pending

**Files:** `config.py`, `db/models.py`, `db/cache.py`, `alpaca_client.py`, `uw_client.py`, `notifications.py`

1. **`config.py`** — `LEVERAGE_CONFIG` dict from the prompt + `.env` loading (dotenv)
   - All thresholds, regime parameters, execution times
   - Add `min_regime_hold_days: 2` for oscillation protection

2. **`db/models.py`** — Raw SQL schema creation + helper functions
   - `init_tables()` — CREATE IF NOT EXISTS for 3 tables (decisions, regimes, performance)
   - `log_daily_decision(data: dict)` — INSERT into decisions
   - `log_regime_change(old, new, data)` — INSERT into regimes
   - `log_daily_performance(data: dict)` — INSERT into performance
   - `get_last_regime() -> str | None` — For cold start detection
   - `get_regime_duration_days() -> int` — For oscillation protection
   - `get_consecutive_losing_days() -> int` — For holding period check
   - `get_position_entry_date() -> str | None` — For holding period
   - `get_performance_summary(days=30) -> dict`

3. **`db/cache.py`** — Local bar cache for QQQ/TQQQ
   - SQLite table `bar_cache(symbol, date, open, high, low, close, volume)`
   - `get_cached_bars(symbol, days) -> list` — Read from cache
   - `update_cache(symbol, bars)` — Upsert new bars
   - `get_bars_with_cache(symbol, days) -> list` — Check cache, fetch only missing days from Alpaca, update cache
   - Used by backtest (load full history once) and daily run (fetch only latest)

4. **`alpaca_client.py`** — Thin wrapper around alpaca-py
   - `get_account() -> dict` — Equity, cash, PDT count
   - `get_positions() -> list[dict]` — All positions (for capital isolation)
   - `get_tqqq_position() -> dict | None` — Current TQQQ holding
   - `get_snapshot(symbols) -> dict` — Multi-symbol snapshot
   - `get_bars(symbol, start, end) -> list` — Historical daily bars
   - `get_calendar(date) -> dict` — Half-day detection
   - `submit_market_order(symbol, qty, side) -> dict` — Market order
   - Retry logic: 3 retries with 30s delay on HTTP errors

5. **`uw_client.py`** — Unusual Whales TQQQ flow
   - `get_tqqq_flow(lookback_hours=4) -> dict` — Put/call premium totals
   - Best-effort with timeout + retry, returns neutral on failure

6. **`notifications.py`** — Telegram via httpx
   - `send_daily_report(data: dict)` — Formatted daily summary
   - `send_regime_alert(old, new, data: dict)` — Regime change alert
   - `send_halfday_alert()` — Half-day detection alert
   - `send_error(title, detail)` — Error notification
   - `send_backtest_summary(stats: dict)` — Backtest results

**Tests:** `test_cache.py` (cache read/write/update), basic model tests

---

### Phase 2: Strategy Logic (regime, signals, sizing) — [ ] Pending

**Files:** `strategy/regime.py`, `strategy/signals.py`, `strategy/sizing.py`

1. **`strategy/regime.py`** — Regime state machine
   - `detect_regime(qqq_close, sma_50, sma_250, deadzone_pct) -> str`
   - 5 states: STRONG_BULL, BULL, CAUTIOUS, RISK_OFF, BREAKDOWN
   - `get_effective_regime(detected, previous, hold_days, min_hold) -> str`
     - Cold start: if no previous regime, return CAUTIOUS
     - Oscillation: if regime changed < `min_regime_hold_days` ago AND new regime is not RISK_OFF/BREAKDOWN, keep previous regime
     - RISK_OFF/BREAKDOWN always take effect immediately
   - `get_regime_target_pct(regime) -> float` — Maps regime to allocation %

2. **`strategy/signals.py`** — Momentum, volatility, flow
   - `calculate_momentum(closes, roc_fast=5, roc_slow=20) -> dict` — ROC values + blended momentum score (0-1)
   - `calculate_realized_vol(closes, window=20) -> float` — Annualized realized vol %
   - `classify_vol_regime(vol) -> str` — LOW/NORMAL/HIGH/EXTREME
   - `get_vol_adjustment(vol_regime) -> float` — 1.0/1.0/0.5/0.0
   - `check_options_flow(uw_client) -> tuple[bool, float]` — (is_bearish, adjustment_factor)
   - `check_consecutive_down_days(closes, max_days=5) -> bool`
   - `check_overextended(close, sma_50, threshold=0.15) -> bool`
   - `check_sideways(closes, days=30, range_pct=0.05) -> bool`

3. **`strategy/sizing.py`** — Position sizing + entry gates
   - `get_allocated_capital(equity, positions, max_portfolio_pct) -> dict`
     - Returns: `{total_equity, other_positions_value, allocated_capital, cash_available}`
     - Single function, used everywhere (DRY)
   - `run_gate_checklist(data: dict) -> tuple[bool, list[str]]`
     - Returns (all_passed, list_of_failed_gates)
     - 14 individual gate checks, each returns (passed, reason)
     - Gates: regime, trend_strength, momentum, vol, daily_loss, sideways, holding_days, overextended, consecutive_down, execution_window, capital, pdt, flow_sentiment, data_quality
   - `calculate_target_shares(data: dict) -> dict`
     - Full pipeline: regime target → momentum scaling → vol adjustment → flow adjustment → overextension reduction → share count
     - Returns: `{target_shares, target_value, current_shares, delta_shares, delta_value, action, limiting_factors}`

**Tests:** `test_regime.py` (15+ tests), `test_signals.py` (15+ tests), `test_sizing.py` (15+ tests including all 14 gate checks)

---

### Phase 3: Execution + CLI (executor, job) — [ ] Pending

**Files:** `strategy/executor.py`, `job.py`

1. **`strategy/executor.py`** — Order handling
   - `execute_rebalance(target_shares, current_shares, price, client) -> dict`
     - Calculates delta, checks min_trade_value threshold
     - Submits market order (buy or sell)
     - Returns fill details
   - `force_exit(client) -> dict` — Sell all TQQQ immediately
   - PDT check integrated: skip non-emergency rebalances if day_trades < 2

2. **`job.py`** — CLI entry point with 4 commands
   - `run` — Full daily pipeline (fetch → regime → gates → size → execute → log → notify)
     - `--halfday-check` flag: exits immediately if not a half day
   - `status` — Diagnostic (no trading): regime, SMAs, position, vol, flow, gates, what-if
   - `backtest` — Full historical simulation (Phase 4)
   - `force_exit` — Emergency sell all TQQQ

**Tests:** `test_executor.py` (order submission, PDT check, min trade value)

---

### Phase 4: Backtest — [ ] Pending

**Files:** `job.py` (backtest command), `test_backtest.py`

1. **Backtest engine** — SQLite-based
   - Fetch 2+ years QQQ + TQQQ daily bars (use cache)
   - TQQQ split check: verify pre-2022 prices are in $20-80 range (adjusted)
   - Simulate day-by-day:
     - Calculate SMAs, momentum, vol for each day
     - Run regime detection + gate checklist
     - Calculate target position
     - Track simulated fills (market orders at close price)
     - Record P&L, drawdown, regime transitions
   - Store results in `backtest_results` SQLite table
   - Track 3 strategies: this strategy, buy-hold QQQ, buy-hold TQQQ
   - Output: summary stats to Telegram, full results to CSV

2. **Backtest assertions** (in `test_backtest.py`)
   - Strategy is 100% cash by March 1, 2020
   - Strategy is 100% cash by January 15, 2022
   - Max drawdown < 45%
   - Zero buy trades during RISK_OFF periods
   - Total return > QQQ buy-and-hold (risk-adjusted)

---

### Phase 5: Failure Mode Tests + Deploy — [ ] Pending

**Files:** `test_failures.py`, cron setup, GitHub push

1. **Failure mode tests** (9 tests):
   - Alpaca API 500 → retry then alert
   - Insufficient bars (<250) → abort, no trade
   - Zero equity → no trade, alert
   - Stale TQQQ price → no trade, alert
   - Market closed (holiday) → exit early
   - UW API timeout → skip flow gate, proceed
   - Negative share calculation → cap at 0
   - Order rejected → log, alert, no crash
   - Half-day missed (3:50 PM but market closed) → handle gracefully

2. **Cron setup:**
   ```
   50 20 * * 1-5  ~/leveraged-etf/.venv/bin/python ~/leveraged-etf/job.py run >> ~/leveraged-etf/logs/leveraged_etf.log 2>&1
   45 17 * * 1-5  ~/leveraged-etf/.venv/bin/python ~/leveraged-etf/job.py run --halfday-check >> ~/leveraged-etf/logs/leveraged_etf.log 2>&1
   35 14 * * 1-5  ~/leveraged-etf/.venv/bin/python ~/leveraged-etf/job.py status >> ~/leveraged-etf/logs/leveraged_etf.log 2>&1
   ```

3. **GitHub:** Create private repo, push

---

## Dependencies

```toml
[project]
name = "leveraged-etf"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "alpaca-py>=0.33.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

---

## Verification

1. `pytest tests/` — All 60+ tests pass
2. `python job.py status` — Shows regime, SMAs, vol, gates, position
3. `python job.py backtest` — Validates crash exits, exports CSV
4. `python job.py run` — Executes on paper account (during market hours)
5. Telegram receives daily report and any regime alerts
6. Cron entries installed and verified
7. Pushed to GitHub as private repo
