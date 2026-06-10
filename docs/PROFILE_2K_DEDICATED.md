# Dedicated $2,000 Account Profile

Profile for running this strategy on its own small account (not shared with
momentum-agent). Chosen 2026-06-10 after the decade stress-test
(see `CHANGELOG.md` and `docs/SIM_2K_90D_2026-06-10.md`).

## Config overrides vs the shared-account config

```python
"max_portfolio_pct": 1.0,   # whole account is working capital (shared acct: 0.30)
# Everything else identical — the 3x-tier sleeves/caps are already in config.py:
#   max_position_pct 0.60, bull_position_pct 0.45
#   sleeve_trend_core_pct 0.36, breakout 0.15, pullback 0.12, mean_reversion 0.12
```

## How to activate

1. Create the dedicated Alpaca account; put its keys in this repo's `.env`
   (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`).
2. Set `max_portfolio_pct: 1.0` in `config.py`.
3. Deploy + set up the same cron schedule (see CLAUDE.md / server crontab).

## Cash sweep at this size

The SGOV sweep (`use_cash_sweep: True`) works unchanged: idle capital
(~$1,000–1,400 typically) holds ~10–14 SGOV shares. `sweep_min_trade_value:
250` means the sweep adjusts in ≥$250 steps — at this account size it will
rebalance the sweep every few weeks, which is correct (don't lower it).

## Expectations at this tier (from the 2017–2026 replay, $2k start)

- CAGR ≈ 17% → $2k ≈ $8.5k over a decade
- Worst-ever drawdown −16.8%; worst single day −5.5%
- 2022-bear outcome: +17% (system mostly in cash below SMA-250)
- Longest underwater stretch: ~1 year (system flat in cash during bears —
  this is expected behavior, not malfunction)

## Real-account cautions

- Under $25k equity = PDT rules. The system holds overnight and reserves
  2 day trades (`pm_min_day_trades_reserve`), but intraday exits can be
  skipped when day trades are exhausted — gaps fill the role of stops then.
- Whole shares only at this account size: at TQQQ ≈ $75, a 60% target on
  $2k is ~16 shares; granularity error ≈ 3% of target. Acceptable.
- `min_trade_value: 100` means rebalances under $100 are skipped — at $2k
  this prevents pointless churn; do not lower it.
