# Leveredbot Audit Fixes

Audit of `leveraged_etf_build_prompt.md` against implementation — 2026-02-09

## BUG

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 1 | Gate count hardcoded as 14 in daily report — should be 16 | `job.py:992`, `sizing.py` docstrings | [x] FIXED |

## MEDIUM

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 2 | Gate 10 (execution window) always returns True — not dynamically checked against market hours/half-days | `job.py` new `_is_execution_window()` | [x] FIXED |
| 3 | Momentum normalization uses fixed bounds (-0.10, +0.10) instead of historical percentile ranking | `signals.py:calculate_momentum()` | [x] FIXED |
| 4 | Exit rules (vol spike >35, losing >15 days) only gate entries — spec says they should force exit existing positions | `position_manager.py` | [x] ALREADY COVERED — position_manager handles forced exits at morning/midday checks |

## LOW

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 5 | Missing `get_regime_history(days=30)` DB function | `db/models.py` | [x] FIXED |
| 6 | Missing benchmark QQQ comparison in daily Telegram report | `notifications.py`, `job.py` | [x] FIXED |
| 7 | No error alert sent after API retries exhausted | `alpaca_client.py:_retry()` | [x] FIXED |
| 8 | No data staleness check (spec wants alert if data >1 day old) | `job.py:cmd_run()` | [x] FIXED |

## MISSING

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 9 | `bot.py` Telegram bot commands completely absent — `/leverage`, `/leverageperf`, `/leverageregime`, `/leverageexit`, `/leveragebacktest`, `/leveragevol`, `/leverageflow` | `bot.py` (new file) | [x] FIXED |

## INTENTIONAL (no fix needed)

| # | Issue | Notes |
|---|-------|-------|
| 10 | `sma_deadzone_pct`: spec says 0.02, code uses 0.05 | Intentionally widened for less whipsaw |
