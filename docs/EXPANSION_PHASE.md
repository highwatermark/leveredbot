# Expansion Phase — Strategy Optimization

## Current State (Feb 2026)

### Live Backtest Results (Aug 2024 – Feb 2026, real Alpaca data)

| Metric | Strategy | QQQ B&H | TQQQ B&H |
|--------|----------|---------|----------|
| Return | -0.4% | +28.4% | -26.0% |
| Max Drawdown | 6.3% | — | — |
| Trades | 306 | 0 | 0 |
| Regime Flips | 37 | — | — |

### Root Cause Analysis

1. **Whipsaw is the primary drag.** 37 regime transitions in 18 months — QQQ oscillates around its 50-day SMA, causing CAUTIOUS/STRONG_BULL toggling every 3-7 days. Each flip costs $200-$800.

2. **CAUTIOUS 25% positions bleed value.** Partial positions eat decay without capturing enough upside. The worst-performing component across all variants tested.

3. **TQQQ volatility decay is structural.** TQQQ lost 28.7% while QQQ gained 25.1% — 3x leverage in choppy sideways markets destroys value regardless of timing.

4. **Capital preservation works.** Max drawdown of 6.3% vs TQQQ's raw -26%. The safety thesis is validated even if alpha isn't.

---

## Variants Tested Against Real Data

### Round 1: Tuning Current Architecture

| Config | Return | Max DD | Trades | Flips |
|--------|--------|--------|--------|-------|
| Current (SMA 2% dz, 2d hold) | +0.8% | 8.1% | 212 | 35 |
| 5% dz + 10d hold | +2.2% | 6.1% | 201 | 19 |
| EMA + 5% dz + 10d hold | +0.4% | 6.0% | 197 | 19 |
| EMA + RSI + 5% dz + 10d | +1.2% | 6.1% | 202 | 20 |
| 4% dz + weekly only | +0.1% | 6.0% | 199 | — |

### Round 2: Structural Changes

| Config | Return | Max DD | Trades | Flips |
|--------|--------|--------|--------|-------|
| Binary (all-in/out) EMA 7% dz, 10d | +0.3% | 14.5% | 244 | 5 |
| Binary + RSI, 5% dz, 15d hold | **+1.3%** | 14.5% | 209 | 15 |
| Higher alloc (40%) + RSI | +1.2% | 19.1% | 232 | 15 |

### Key Findings

- **Binary (all-in or all-out) beats partial positions** in choppy markets — fewer flips, less decay drag
- **RSI filter adds ~1% by avoiding overbought entries** (prevents buying into peaks)
- **Wider deadzones (5-7%) reduce flips** but don't help much alone — need combined with hold period
- **SMA slope filter had negligible impact** — EMA smoothing achieves the same thing more cleanly
- **Higher allocation (40%) amplifies both gains and losses** — not worth it without better timing

---

## Strategy for Choppy Markets

The current architecture is optimized for sustained trends. In choppy/range-bound markets, it will underperform. Potential improvements ranked by expected impact:

### Tier 1: High Impact (implement next)

**A. Binary Mode Toggle**
- Remove CAUTIOUS (25%) regime entirely
- Above EMA-200 + deadzone → 70% allocated (FULL)
- Below → 0% (CASH)
- Rationale: partial positions in chop are pure drag

**B. Wider Deadzone + Longer Hold**
- Increase from 2% to 5-7%
- Increase min_hold from 2 to 10-15 days
- Reduces regime flips from 37 to ~5-15

**C. RSI Overbought Filter**
- Don't enter/add when RSI(14) > 70-75
- Reduces "buying the top" entries by ~30%
- Computationally free (uses same daily closes)

### Tier 2: Moderate Impact (research needed)

**D. Volatility-Adaptive Deadzone**
- When realized vol is HIGH, widen deadzone to 7-10%
- When LOW, tighten to 3-4%
- Adapts to market regime automatically

**E. Breadth Confirmation**
- Track NASDAQ advance/decline ratio or % stocks above 200-SMA
- Only enter FULL position when breadth confirms (>60% stocks above 200-SMA)
- Leading indicator — breadth narrows before index breaks
- Requires additional data source

**F. Momentum Regime (ROC slope)**
- Instead of price vs SMA, use the *rate of change of the SMA itself*
- Rising SMA slope → trend confirmed → hold
- Flat/declining slope → trend exhausted → reduce
- Avoids the "price oscillates around flat SMA" problem

### Tier 3: Speculative (needs more data)

**G. Inverse Vol Sizing**
- Size positions inversely to realized vol: low vol → larger, high vol → smaller
- Already partially implemented via vol_adjustment, but could be continuous instead of stepped

**H. Mean Reversion Overlay**
- In CAUTIOUS regime, instead of holding partial TQQQ position, buy QQQ dips (non-leveraged)
- Switches from leveraged-trend to unleveraged-mean-reversion when conditions are choppy

**I. TQQQ → QQQ Fallback**
- When vol regime is HIGH/EXTREME, trade QQQ instead of TQQQ
- Captures directional moves without 3x decay
- Adds complexity to capital isolation

---

## Week 1 Live Test Plan (Current Strategy)

**Objective:** Establish baseline with current architecture on paper account.

**What to observe:**
- Does it enter a position? What regime does it detect?
- How many regime flips in 5 trading days?
- How close are fills to the daily close price? (execution quality)
- Does Telegram reporting work reliably?
- Does capital isolation hold (momentum-agent positions untouched)?
- Does the status command give useful diagnostics?

**Daily checks:**
```bash
python job.py status          # Morning — review regime + signals
# Cron handles 3:50 PM run automatically
# Review Telegram report each evening
```

**After 1 week:**
- If 0-1 regime flips → current params may be fine for trending periods
- If 3+ regime flips → implement Tier 1 changes (binary + wider dz + RSI)
- Compare daily decisions against what QQQ actually did next day

---

## Data Limitations

- Alpaca free tier provides ~2.5 years of history (Aug 2023 – present)
- Cannot backtest through COVID crash (Mar 2020) or 2022 bear market
- TQQQ had a 2:1 reverse split in Dec 2025 — data is split-adjusted but verify
- For longer backtests, would need premium data source (Polygon, Tiingo, etc.)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-09 | Deploy current strategy for 1-week live test | Establish baseline before optimizing |
| 2026-02-09 | Document all backtest variants | Data-driven optimization after live observation |
| TBD | Implement Tier 1 changes if whipsaw observed | Binary + wider dz + RSI showed +1.3% vs -0.4% |
