# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run tests (skip sklearn-dependent tests that fail in this venv)
.venv/bin/python -m pytest tests/ --ignore=tests/test_knn_signal.py -v

# Run a single test file
.venv/bin/python -m pytest tests/test_regime.py -v

# Run a single test class or function
.venv/bin/python -m pytest tests/test_sizing.py::TestGateChecklist -v
.venv/bin/python -m pytest tests/test_sizing.py::TestGateChecklist::test_all_gates_pass -v

# Strategy execution (production)
.venv/bin/python job.py run
.venv/bin/python job.py status
.venv/bin/python job.py morning
.venv/bin/python job.py midday
.venv/bin/python job.py pregame
.venv/bin/python job.py backtest
.venv/bin/python job.py force_exit
```

**Important:** Always use `.venv/bin/python`, not `python`. `scikit-learn` is listed in pyproject.toml but not fully installed in the venv — tests importing `knn_signal.py` or `xgb_signal.py` directly will fail with ImportError. The pre-existing `test_cmd_backtest_writes_to_db` test also fails for this reason.

## Architecture

This is an automated TQQQ/SQQQ position-trading system that runs as daily cron jobs on an Alpaca brokerage account shared with a separate momentum-agent.

### Data Pipeline (job.py `cmd_run`)

```
_fetch_all_data() → MarketData dataclass
    ↓
_compute_signals() → StrategySignals dataclass
    ↓
run_gate_checklist() → 14 boolean gates (ALL must pass for entry)
    ↓
calculate_target_shares() → share delta
    ↓
execute_rebalance() → market order via Alpaca
    ↓
DB logging + Telegram notification
```

`MarketData` and `StrategySignals` are typed dataclasses in `pipeline_types.py` that flow through the entire pipeline.

### Strategy Modules (strategy/)

- **regime.py** — 5-state regime detection (STRONG_BULL → BREAKDOWN) based on QQQ vs SMA-50/SMA-250. Oscillation protection enforces 10-day minimum hold. RISK_OFF/BREAKDOWN always take effect immediately.
- **signals.py** — Computes momentum (blended 5d/20d ROC), realized volatility (20-day rolling, annualized), and options flow sentiment. Cross-asset bars (TLT, GLD, IWM) feed the ML models.
- **sizing.py** — 14-gate entry checklist + position sizing. `get_allocated_capital()` enforces 30% max portfolio share so this strategy never competes with momentum-agent for capital.
- **executor.py** — Market order submission with PDT awareness. Intent-first logging: writes DB record before order, updates after fill.
- **position_manager.py** — Intraday exit checks (morning at 9:35 AM, midday at 12:30 PM): stop-loss, trailing stop, gap-down, regime emergency, daily loss limit, partial profit-taking.
- **knn_signal.py / xgb_signal.py** — ML overlays (k-NN and XGBoost) using a 20-feature vector. Currently in report-only mode (`knn_report_only: True`).

### Config

All tunable parameters live in `LEVERAGE_CONFIG` dict in `config.py`. Binary mode is enabled (`use_binary_mode: True`) which maps CAUTIOUS→BULL. SQQQ trading is enabled for k-NN SHORT signals.

### Database (db/)

SQLite at `data/leveraged_etf.db`. Four tables: `decisions` (one row per execution), `regimes` (transition log), `performance` (daily P&L), `pregame` (pre-execution intel). `cache.py` provides append-only bar caching to minimize API calls.

### Testing Patterns

- In-memory SQLite via `db_conn` fixture from `conftest.py` — always use this, never real DB
- Mock `alpaca_client` with `MagicMock`; for multi-symbol snapshot tests, use side_effect routing on `get_snapshot`
- Options flow: use `mock_uw_flow_neutral` / `mock_uw_flow_bearish` fixtures
- Position manager: use `mock_tqqq_position_state` / `mock_sqqq_position_state` fixtures
- External APIs (Alpaca, Unusual Whales, Telegram) are always mocked in tests
