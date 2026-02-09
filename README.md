# Leveraged ETF Trading System

Fully automated position-trading system that trades **TQQQ** (3x Bull NASDAQ) and goes to 100% cash during Risk-Off periods. Runs as a daily cron job at 3:50 PM EST.

## Strategy Overview

Uses regime detection on **QQQ** (unleveraged NASDAQ 100) to determine positioning in TQQQ:

| Regime | Condition | Target Allocation |
|--------|-----------|-------------------|
| **STRONG_BULL** | QQQ > SMA-50 + SMA-250, golden cross | 70% of allocated capital |
| **BULL** | QQQ > both SMAs | 50% |
| **CAUTIOUS** | QQQ > SMA-250 but near/below SMA-50 | 25% |
| **RISK_OFF** | QQQ < SMA-250 - deadzone | 0% (full cash) |
| **BREAKDOWN** | QQQ < SMA-250, death cross | 0% (full cash) |

Allocations are further scaled by momentum, realized volatility, and options flow sentiment.

## Architecture

```
config.py                 # All thresholds and settings
alpaca_client.py          # Alpaca API wrapper (trading + data)
uw_client.py              # Unusual Whales API (TQQQ flow sentiment)
notifications.py          # Telegram notifications

strategy/
  regime.py               # 5-state regime detection with oscillation protection
  signals.py              # Momentum, realized vol, options flow, guardrails
  sizing.py               # Capital isolation, 14-gate checklist, position sizing
  executor.py             # Order submission, PDT checks, force exit

db/
  models.py               # SQLite schema (decisions, regimes, performance, backtest)
  cache.py                # Local bar cache (QQQ/TQQQ daily bars)

job.py                    # CLI: run, status, backtest, force_exit
```

## Safety Features

- **14-gate entry checklist** — all must pass before any buy order
- **Capital isolation** — shares the Alpaca account with a momentum agent; only uses 30% of equity
- **PDT awareness** — skips non-emergency rebalances if <2 day trades remain
- **Emergency exits** — RISK_OFF/BREAKDOWN regime shifts always execute regardless of PDT
- **Realized volatility** — reduces to 50% at high vol, 0% at extreme vol (no VIX/VIXY dependency)
- **Oscillation protection** — 2-day minimum regime hold to prevent whipsaws
- **Cold start** — defaults to CAUTIOUS on first run

## Setup

```bash
# Clone and create venv
cd ~/leveraged-etf
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys (Alpaca, Telegram, Unusual Whales)
```

## Usage

```bash
# Daily strategy execution (runs at 3:50 PM EST via cron)
python job.py run

# Half-day check (exits immediately if not a half day)
python job.py run --halfday-check

# Show current regime, signals, gates, what-if (no trading)
python job.py status

# Run historical backtest with P&L simulation
python job.py backtest

# Emergency sell all TQQQ
python job.py force_exit
```

## Cron Setup

```cron
# Daily execution at 3:50 PM EST (20:50 UTC)
50 20 * * 1-5  ~/leveraged-etf/.venv/bin/python ~/leveraged-etf/job.py run

# Half-day safety net at 12:45 PM EST (17:45 UTC)
45 17 * * 1-5  ~/leveraged-etf/.venv/bin/python ~/leveraged-etf/job.py run --halfday-check

# Morning status check at 9:35 AM EST (14:35 UTC)
35 14 * * 1-5  ~/leveraged-etf/.venv/bin/python ~/leveraged-etf/job.py status
```

## Tests

```bash
pytest tests/ -v    # 127 tests across 7 test files
```

Test coverage:
- **test_regime.py** — regime detection, oscillation protection, cold start
- **test_signals.py** — momentum, realized vol, vol regimes, overextension, sideways
- **test_sizing.py** — capital isolation, all 14 gate checks, position sizing pipeline
- **test_executor.py** — order submission, PDT checks, force exit
- **test_backtest.py** — crash exits, drawdown limits, risk-off compliance
- **test_cache.py** — bar caching read/write/upsert
- **test_failures.py** — all 9 failure modes (API errors, stale data, order rejections)

## Key Settings

| Setting | Value |
|---------|-------|
| Max portfolio allocation | 30% of total equity |
| Max position (STRONG_BULL) | 70% of allocated capital |
| SMA periods | 50-day / 250-day |
| Deadzone | 2% band around SMAs |
| Min trend strength | 2% above SMA |
| Min momentum score | 0.3 |
| Realized vol thresholds | Low <15, Normal <25, High <35, Extreme 35+ |
| Options flow bearish ratio | Put premium > 2x call premium |
| Min trade value | $100 (avoid churning) |
| Execution time | 3:50 PM EST (12:45 PM half days) |
