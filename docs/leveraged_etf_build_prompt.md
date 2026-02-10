## Prompt

```
Build a fully automated leveraged ETF position-trading system that trades TQQQ (3x Bull NASDAQ) 
and goes to cash during Risk-Off periods. The system runs as a daily cron job on an existing 
momentum-agent server (Ubuntu, Python 3.11+, Alpaca paper account).

This system operates ALONGSIDE an existing equity momentum trading bot in the same Alpaca account.
It must respect shared capital and not interfere with existing positions.

## REPOSITORY CONTEXT

This lives in the existing momentum-agent repo at ~/momentum-agent/
The repo already has:
- Alpaca API keys in .env (ALPACA_API_KEY, ALPACA_SECRET_KEY)
- Anthropic API key in .env (ANTHROPIC_API_KEY)
- SQLite database at data/trades.db
- Telegram bot integration (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID in .env)
- Python virtualenv at ./venv/
- Existing equity positions that must NOT be touched (currently BRK.B, TSM, VMC — but this changes)

## STRATEGY OVERVIEW

Mathematical Momentum & Trend Filtering using regime detection on QQQ 
(unleveraged NASDAQ 100) to determine positioning in TQQQ (3x leveraged).

Core thesis: Large sustained rallies require price above key moving averages.
By identifying regime shifts using the 50-day and 250-day SMA on QQQ,
we capture leveraged upside during confirmed bull trends and preserve 
capital by going to cash during confirmed breakdowns.

Execution window: 3:50 PM EST daily (or 12:45 PM EST on half days). 
One decision, one trade, deterministic.

## FILES TO CREATE

### 1. `leveraged_etf.py` - Main Strategy Engine

This is the core strategy file. All logic lives here.

#### Configuration Block

```python
LEVERAGE_CONFIG = {
    # Instruments
    "bull_etf": "TQQQ",           # 3x Bull NASDAQ 100
    "bear_etf": "SQQQ",           # 3x Inverse NASDAQ 100 (reference only, we do NOT trade this)
    "underlying": "QQQ",           # Regime detection runs on QQQ (unleveraged)
    
    # Capital Allocation
    "max_portfolio_pct": 0.30,     # Max 30% of TOTAL account equity for this strategy
    "max_position_pct": 0.70,      # Max 70% of ALLOCATED capital in TQQQ at any time
    "min_position_pct": 0.10,      # Minimum position if Risk-On (10% of allocated)
    
    # Regime Detection (calculated on QQQ, not TQQQ)
    "sma_fast": 50,                # 50-day SMA
    "sma_slow": 250,               # 250-day SMA  
    "sma_deadzone_pct": 0.02,      # 2% band around SMA to prevent whipsaws
    
    # Historical Data
    # CRITICAL: Need 400 calendar days (~280 trading days) to guarantee
    # a full 250-bar SMA. The old value of 300 calendar days (~210 trading
    # days) was 40 bars short. Always use calendar days >= 400 or set an
    # explicit start date 14 months prior.
    "history_calendar_days": 400,
    
    # Momentum / Rate of Change
    "roc_period": 20,              # 20-day rate of change for momentum scoring
    "roc_fast": 5,                 # 5-day rate of change for short-term momentum
    
    # Volatility Guardrails — REALIZED VOLATILITY ONLY
    # We do NOT use VIXY or ^VIX. VIXY is a futures-based ETF whose price
    # does not correspond 1:1 with VIX levels (e.g., VIXY ~$27 when VIX ~18).
    # The UW SPIKE endpoint also returns empty. Instead we compute 20-day
    # realized vol on QQQ returns, annualized (* sqrt(252)).
    #
    # Realized vol thresholds (annualized):
    #   < 15  → "Low vol"     → full allocation OK
    #   15-25 → "Normal vol"  → standard allocation
    #   25-35 → "High vol"    → reduce allocation 50%
    #   > 35  → "Extreme vol" → go to 100% cash
    "vol_low_threshold": 15,
    "vol_normal_threshold": 25,
    "vol_high_threshold": 35,
    
    "max_daily_loss_pct": 0.08,    # If TQQQ drops 8%+ intraday, do not enter/add
    
    # Expected Value Guardrails
    "min_trend_strength": 0.02,    # QQQ must be at least 2% above SMA to enter
    "min_momentum_score": 0.3,     # Minimum momentum score (0-1) to hold position
    "mean_reversion_threshold": 0.15, # If QQQ is >15% above 50-SMA, reduce (overextended)
    "consecutive_down_days_max": 5, # If QQQ has 5+ consecutive red days, reduce to minimum
    
    # Volatility Decay Protection
    "max_holding_days_losing": 15, # If position is losing for 15+ days, close and reassess
    "sideways_detection_days": 30, # If QQQ range < 5% over 30 days, reduce exposure
    "sideways_range_pct": 0.05,    # Range threshold for sideways detection
    
    # Options Flow Sentiment (Unusual Whales integration)
    # Before entering or adding, check TQQQ options flow from UW.
    # If recent put premium > 2x call premium, reduce target allocation by 25%.
    # This is an additional gate, not a replacement for regime detection.
    "use_options_flow_gate": True,
    "options_flow_bearish_ratio": 2.0,     # put_premium / call_premium threshold
    "options_flow_reduction_pct": 0.25,    # Reduce allocation by this much if bearish flow
    "options_flow_lookback_hours": 4,      # Look at flow from the last 4 hours
    
    # Execution
    "execution_time_normal": "15:50",  # 3:50 PM EST on normal days
    "execution_time_halfday": "12:45", # 12:45 PM EST on half days (market closes 1 PM)
    "order_type": "market",            # Market orders (EOD, liquid ETF, tight spread)
    "min_trade_value": 100,            # Don't bother rebalancing if delta < $100
    
    # PDT Safety
    # The Alpaca account is flagged as PDT. This strategy trades at most
    # once daily, but the momentum agent may also trade. Track combined
    # day-trade count and skip non-critical rebalances if < 2 day trades
    # remain. Emergency exits (regime shift to RISK_OFF) always execute.
    "min_day_trades_for_rebalance": 2,  # Reserve day trades for emergencies
}
```

#### Volatility Calculation

```python
def calculate_realized_volatility(qqq_closes: list[float], window: int = 20) -> float:
    """
    Calculate annualized realized volatility from QQQ daily closes.
    
    This replaces all VIXY/VIX-based approaches. Reasons:
    - VIXY is a futures ETF, its price does NOT equal the VIX index level
    - Alpaca does not provide ^VIX directly
    - The Unusual Whales SPIKE endpoint returns empty data
    - Realized vol from QQQ returns is self-contained, no external deps
    
    Method:
    1. Compute daily log returns: ln(close_t / close_{t-1})
    2. Take rolling standard deviation over `window` days
    3. Annualize: stdev * sqrt(252)
    
    Returns:
        Annualized realized volatility as a percentage (e.g., 18.2)
    """
```

#### Market Calendar & Half-Day Detection

```python
def get_execution_time() -> str:
    """
    Determine the correct execution time for today.
    
    CRITICAL: On half days (early close), the market closes at 1:00 PM EST.
    If we try to execute at 3:50 PM, the market is already closed and no
    trade will go through. We detect half days by comparing today's close
    time from Alpaca's calendar API against the standard 16:00.
    
    Returns:
        "15:50" for normal days, "12:45" for half days
    """
    # Use Alpaca get_calendar for today
    # If close_time != "16:00" → it's a half day
    # Return execution_time_halfday
```

#### Options Flow Sentiment Gate

```python
def check_options_flow_sentiment() -> tuple[bool, float]:
    """
    Query Unusual Whales TQQQ flow alerts to gauge institutional sentiment.
    
    TQQQ has extremely active options flow (70+ alerts/day observed).
    We use the put/call premium ratio as a sentiment overlay:
    
    - Sum total_premium for puts vs calls from recent flow alerts
    - If put_premium > 2x call_premium → bearish signal → reduce allocation 25%
    - If call_premium > 2x put_premium → bullish confirmation → no reduction
    - Otherwise → neutral → no adjustment
    
    This is NOT a standalone signal. It modifies the target allocation
    calculated by the regime + momentum logic. It can only reduce, never
    increase, the allocation.
    
    Returns:
        (is_bearish: bool, adjustment_factor: float)
        e.g., (True, 0.75) means bearish, multiply target by 0.75
        e.g., (False, 1.0) means neutral/bullish, no change
    """
```

#### Regime Detection Logic

```
REGIME STATES:
  
  STRONG_BULL:  QQQ > SMA_50 + deadzone AND QQQ > SMA_250 + deadzone AND SMA_50 > SMA_250
                → Target: max_position_pct of allocated capital (70%)
  
  BULL:         QQQ > SMA_50 + deadzone AND QQQ > SMA_250
                → Target: 50% of allocated capital
  
  CAUTIOUS:     QQQ > SMA_250 BUT below or near SMA_50 (within deadzone)
                → Target: 25% of allocated capital
  
  RISK_OFF:     QQQ < SMA_250 - deadzone
                → Target: 0% (full cash, sell all TQQQ)
  
  BREAKDOWN:    QQQ < SMA_250 AND SMA_50 < SMA_250 (death cross)
                → Target: 0% (full cash, absolutely no TQQQ)
```

#### Momentum Scaling Within Regime

Within each regime, scale the actual allocation by a momentum factor:

```
momentum_factor calculation:
  1. roc_20 = (QQQ_close - QQQ_close_20d_ago) / QQQ_close_20d_ago
  2. roc_5 = (QQQ_close - QQQ_close_5d_ago) / QQQ_close_5d_ago
  3. momentum_score = (roc_20 * 0.6) + (roc_5 * 0.4)  # Blend long and short momentum
  4. Normalize to 0-1 range based on historical percentile
  5. If momentum_score < min_momentum_score → reduce to min_position_pct
  6. If momentum_score > 0.8 → full regime allocation
  7. If momentum_score between → linear interpolation
```

#### Expected Positive Outcome Guardrails

CRITICAL: Only enter or add to position if expected outcome is positive.
Before ANY buy order, ALL of the following must be true:

```
ENTRY GATE CHECKLIST (ALL must pass):

  □ Regime is STRONG_BULL, BULL, or CAUTIOUS (not RISK_OFF or BREAKDOWN)
  □ QQQ is at least min_trend_strength (2%) above the relevant SMA
  □ Momentum score >= min_momentum_score (0.3)
  □ Realized vol < vol_high_threshold (35) — if 25-35, halve the target
  □ Realized vol < vol_high_threshold (35) — if above, force 100% cash
  □ QQQ has NOT dropped more than max_daily_loss_pct (8%) today
  □ QQQ is NOT in sideways range (< 5% range over last 30 days)
  □ TQQQ current position is NOT in a losing streak > max_holding_days_losing (15 days)
  □ QQQ is NOT overextended (> 15% above 50-SMA) — if so, reduce, don't add
  □ QQQ does NOT have 5+ consecutive down days — if so, hold or reduce
  □ It is a regular trading day AND correct execution window (half-day aware)
  □ There is sufficient allocated capital available
  □ PDT day trades remaining >= min_day_trades_for_rebalance (2) — unless emergency exit
  □ Options flow sentiment is NOT strongly bearish — if so, reduce 25%
  
  If ANY gate fails → DO NOT BUY. Log the reason. Send Telegram alert.
  
EXIT RULES (sell TQQQ if ANY is true — these ALWAYS execute regardless of PDT):
  □ Regime shifts to RISK_OFF or BREAKDOWN
  □ Realized vol spikes above vol_high_threshold (35)
  □ Position has been losing for max_holding_days_losing (15 days)
  □ QQQ drops below SMA_250 - deadzone (confirmed breakdown)
```

#### Data Fetching — Use Snapshots for Efficiency

```python
def fetch_market_data() -> dict:
    """
    Fetch all required market data in minimum API calls.
    
    Use Alpaca's get_stock_snapshot for real-time data (returns latest quote,
    latest trade, minute bar, daily bar, and previous daily bar in ONE call).
    This is more efficient than separate bar + quote calls.
    
    API Calls:
    1. get_stock_snapshot(["TQQQ", "QQQ"]) → current prices, daily bars
    2. get_stock_bars("QQQ", days=400, timeframe="1Day") → historical for SMAs
    3. get_calendar(today, today) → half-day detection
    4. get_account() → equity, cash, positions
    5. get_all_positions() → isolate non-TQQQ positions
    
    For options flow (optional, best-effort):
    6. Unusual Whales get_stock_flow_alerts("TQQQ") → sentiment overlay
    
    Returns:
        Dict with all data needed for regime detection and position sizing
    """
```

#### Position Sizing Calculation

```python
def calculate_target_position():
    """
    Calculate exact number of TQQQ shares to hold.
    
    Flow:
    1. Get total account equity from Alpaca
    2. Subtract value of existing non-TQQQ positions (momentum agent's positions)
    3. Available capital = min(equity * max_portfolio_pct, equity - other_positions_value)
    4. Apply regime target percentage to available capital
    5. Apply momentum scaling factor
    6. Apply realized vol adjustment (halve if vol 25-35, zero if >35)
    7. Apply options flow sentiment adjustment (reduce 25% if bearish)
    8. Apply overextension reduction if applicable
    9. Calculate target_shares = target_dollar_value / TQQQ_current_price
    10. Calculate delta = target_shares - current_shares
    11. If abs(delta * price) < min_trade_value → no trade (avoid churning)
    12. If rebalance (not emergency) and day_trades_remaining < 2 → skip
    13. Return order details (buy X shares or sell Y shares)
    
    SANITY CHECK (at current prices ~$50.64/share):
    - Account equity ~$129K → 30% allocation = ~$38.7K
    - At 70% max position = ~$27.1K = ~535 shares max
    - At 50% (BULL regime) = ~$19.3K = ~382 shares
    - At 25% (CAUTIOUS) = ~$9.7K = ~191 shares
    Log these intermediate values for verification.
    """
```

#### Main Daily Execution Function

```python
def run_daily_strategy():
    """
    Called at execution time daily (3:50 PM or 12:45 PM on half days).
    
    Steps:
    1. Check market calendar — is today a trading day? Is it a half day?
    2. Verify we're in the correct execution window
    3. Fetch QQQ daily bars (400 calendar days for 250-SMA calculation)
    4. Verify we have >= 250 trading-day bars (abort if not)
    5. Calculate SMA_50, SMA_250 on QQQ
    6. Calculate momentum (ROC_5, ROC_20)
    7. Calculate 20-day realized volatility on QQQ
    8. Fetch TQQQ options flow from Unusual Whales (best-effort, don't fail if unavailable)
    9. Determine regime state
    10. Run entry gate checklist (including PDT check)
    11. Calculate target position size
    12. Determine order (buy/sell/hold)
    13. Execute order on Alpaca
    14. Log everything to database
    15. Send Telegram summary
    
    Returns: Dict with regime, signals, order details, portfolio state
    """
```

### 2. `leveraged_etf_db.py` - Database Schema & Functions

Add these tables to the existing SQLite database:

```sql
-- Daily strategy decisions
CREATE TABLE IF NOT EXISTS leveraged_etf_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- QQQ Data
    qqq_close REAL,
    qqq_sma_50 REAL,
    qqq_sma_250 REAL,
    qqq_pct_above_sma50 REAL,
    qqq_pct_above_sma250 REAL,
    qqq_roc_5 REAL,
    qqq_roc_20 REAL,
    
    -- Volatility Data (realized vol, NOT VIX)
    realized_vol_20d REAL,           -- 20-day realized vol, annualized
    vol_regime TEXT,                  -- LOW, NORMAL, HIGH, EXTREME
    
    -- Options Flow Sentiment
    options_flow_put_premium REAL,    -- Total put premium from UW flow
    options_flow_call_premium REAL,   -- Total call premium from UW flow
    options_flow_ratio REAL,          -- put/call premium ratio
    options_flow_bearish INTEGER DEFAULT 0,  -- 1 if ratio > threshold
    options_flow_adjustment REAL DEFAULT 1.0, -- Multiplier applied to target
    
    -- Regime
    regime TEXT,  -- STRONG_BULL, BULL, CAUTIOUS, RISK_OFF, BREAKDOWN
    regime_changed INTEGER DEFAULT 0,
    previous_regime TEXT,
    
    -- Momentum
    momentum_score REAL,
    momentum_factor REAL,
    
    -- Gate Results
    gates_passed INTEGER,
    gates_failed TEXT,  -- JSON list of failed gate names
    
    -- Position Decision
    target_allocation_pct REAL,
    target_dollar_value REAL,
    target_shares INTEGER,
    current_shares INTEGER,
    order_action TEXT,  -- BUY, SELL, HOLD, REBALANCE
    order_shares INTEGER,
    order_value REAL,
    
    -- Execution
    order_id TEXT,
    fill_price REAL,
    fill_time TEXT,
    execution_window TEXT,  -- NORMAL or HALFDAY
    
    -- Portfolio State
    account_equity REAL,
    allocated_capital REAL,
    tqqq_position_value REAL,
    tqqq_pnl_pct REAL,
    other_positions_value REAL,
    cash_balance REAL,
    day_trades_remaining INTEGER,
    
    -- Data Quality
    trading_days_fetched INTEGER,  -- Verify >= 250
    is_half_day INTEGER DEFAULT 0
);

-- Track regime transitions
CREATE TABLE IF NOT EXISTS leveraged_etf_regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    old_regime TEXT,
    new_regime TEXT,
    qqq_close REAL,
    qqq_sma_50 REAL,
    qqq_sma_250 REAL,
    trigger_reason TEXT
);

-- Performance tracking
CREATE TABLE IF NOT EXISTS leveraged_etf_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    tqqq_shares INTEGER,
    tqqq_avg_cost REAL,
    tqqq_current_price REAL,
    tqqq_position_value REAL,
    tqqq_pnl_day REAL,
    tqqq_pnl_total REAL,
    tqqq_pnl_pct REAL,
    regime TEXT,
    allocated_capital REAL,
    realized_vol REAL,
    benchmark_qqq_pct REAL,  -- Compare performance vs just holding QQQ
    strategy_total_return_pct REAL
);
```

Functions to implement:
- init_leveraged_etf_tables()
- log_daily_decision(decision_dict)
- log_regime_change(old, new, data)
- log_daily_performance(perf_dict)
- get_regime_history(days=30)
- get_performance_summary()
- get_consecutive_losing_days()
- get_last_regime()
- get_position_entry_date()

### 3. `leveraged_etf_job.py` - Cron Job Runner

```python
"""
Cron job entry point for leveraged ETF strategy.

Usage:
    python leveraged_etf_job.py run       # Execute daily strategy
    python leveraged_etf_job.py status    # Show current state without trading
    python leveraged_etf_job.py backtest  # Run historical backtest
    python leveraged_etf_job.py force_exit # Emergency exit all TQQQ
"""
```

The `run` command:
1. Check if market is open today
2. Detect half-day schedule and adjust execution time
3. Run the full strategy pipeline
4. Log results to DB
5. Send Telegram notification with daily summary

The `status` command (no trading, just diagnostics):
1. Show current regime
2. Show QQQ vs SMAs (verify >= 250 bars available)
3. Show current TQQQ position
4. Show realized volatility (not VIX)
5. Show options flow sentiment from UW
6. Show gate checklist status (including PDT remaining)
7. Show what the strategy WOULD do if executed now

The `backtest` command:
1. Pull 2+ years of QQQ daily data from Alpaca
2. **CRITICAL: Use split-adjusted prices for TQQQ.** TQQQ had a 10:1 reverse split
   in January 2022. If using raw Alpaca bars, the data IS already split-adjusted.
   But verify by checking that TQQQ prices pre-2022 are in the correct range
   (should be ~$20-80, NOT $2-8). If data looks wrong, log a warning and skip
   pre-split period.
3. Simulate the strategy day by day
4. Track hypothetical P/L, max drawdown, win rate
5. Compare vs buy-and-hold QQQ and buy-and-hold TQQQ
6. Output summary statistics and save to CSV
7. Verify: strategy correctly exits before COVID crash (Mar 2020) and 2022 bear market

### 4. Add Telegram Commands to `bot.py`

Add these commands to the existing Telegram bot:

```
/leverage         - Show current leveraged ETF strategy status
/leverageperf     - Show performance history
/leverageregime   - Show regime history and transitions
/leverageexit     - Force exit all TQQQ (emergency)
/leveragebacktest - Run backtest and send results
/leveragevol      - Show current realized volatility breakdown
/leverageflow     - Show TQQQ options flow sentiment from Unusual Whales
```

### 5. Crontab Entries

```bash
# Leveraged ETF Strategy - Normal day execution at 3:50 PM EST (20:50 UTC)
# The script itself checks for half days and adjusts, but we also add a
# half-day cron entry as a safety net.
50 20 * * 1-5 cd /home/ubuntu/momentum-agent && ./venv/bin/python leveraged_etf_job.py run >> logs/leveraged_etf.log 2>&1

# Half-day safety net - 12:45 PM EST (17:45 UTC)
# The script checks if today is a half day. If not, it exits immediately.
# If yes, it executes the strategy early.
45 17 * * 1-5 cd /home/ubuntu/momentum-agent && ./venv/bin/python leveraged_etf_job.py run --halfday-check >> logs/leveraged_etf.log 2>&1

# Morning status check - 9:35 AM EST (14:35 UTC) - no trades, just regime check
35 14 * * 1-5 cd /home/ubuntu/momentum-agent && ./venv/bin/python leveraged_etf_job.py status >> logs/leveraged_etf.log 2>&1
```

## CRITICAL IMPLEMENTATION NOTES

### Capital Isolation
The momentum agent and leveraged ETF strategy share the same Alpaca account.
TQQQ positions belong to this strategy. All other positions belong to the momentum agent.
When calculating available capital:
- Get total equity
- Subtract all non-TQQQ position values
- Allocated capital = min(equity * 0.30, remaining_cash)
This prevents the two strategies from competing for capital.

Current account state for reference (will change):
- Equity: ~$129K
- Momentum positions: BRK.B (~$12.7K), TSM (~$13K), VMC (~$12.3K) = ~$38K
- Cash: ~$91K
- 30% cap = ~$38.7K → comfortably fits alongside momentum positions

### PDT Day Trade Management
The account is flagged as Pattern Day Trader with day trades tracked.
- This strategy trades at most once daily (end of day), which is fine.
- BUT the momentum agent may also use day trades.
- Before any NON-EMERGENCY rebalance, check `daytrade_count` from account info.
- If fewer than 2 day trades remain, skip the rebalance (it can wait until tomorrow).
- EXIT trades (regime shift to RISK_OFF/BREAKDOWN) ALWAYS execute regardless of PDT.

### Order of Operations
1. Always fetch fresh account data before ANY calculation
2. Always check what positions exist before calculating targets
3. Always handle the case where TQQQ is already held (rebalance vs new entry)
4. Always handle partial fills and order rejections gracefully
5. Never place an order if the market is closed
6. Always check market calendar for half days BEFORE deciding execution time

### Historical Data Validation
CRITICAL: Before calculating SMAs, verify the data quality:
```python
bars = fetch_qqq_bars(calendar_days=400)
trading_days = len(bars)
if trading_days < 250:
    log_error(f"Only {trading_days} trading days fetched, need 250 for SMA_250")
    send_telegram_alert("⚠️ Insufficient data for SMA calculation. Skipping today.")
    return  # DO NOT TRADE with incomplete data
```
The old 300 calendar days ≈ 210 trading days, which was 40 bars short of the 
250 needed. Using 400 calendar days ≈ 280 trading days provides adequate buffer.

### Alpaca API Usage
Use the existing alpaca-py library already installed in the repo.
- StockHistoricalDataClient for bars
- TradingClient for orders and positions
- **Use get_stock_snapshot(["TQQQ", "QQQ"]) for current data** — returns latest 
  quote, trade, minute bar, daily bar, and previous daily bar in ONE call. This is
  more efficient than separate get_stock_latest_quote + get_stock_bars calls.
- Get historical bars with timeframe=TimeFrame.Day, start=today-400days for SMA calculations
- Use get_calendar(today, today) to detect half-day schedules

### Volatility Measurement — Realized Vol ONLY
Do NOT use VIXY, ^VIX, or any external VIX data source. Reasons validated:
1. VIXY is a futures-based ETF. Its price ($26.71 currently) does NOT equal the VIX 
   index level. Using VIXY price with VIX-scale thresholds (30, 40) would be wrong.
2. Alpaca does not provide ^VIX as a tradable/quotable symbol.
3. The Unusual Whales SPIKE endpoint returns empty data.
4. Realized volatility from QQQ returns is self-contained and requires no external deps.

Implementation:
```python
import numpy as np

def calc_realized_vol(closes: list[float], window: int = 20) -> float:
    """20-day realized vol, annualized."""
    returns = np.diff(np.log(closes[-window-1:]))  # log returns
    return float(np.std(returns) * np.sqrt(252) * 100)  # annualized %
```

Map to vol regimes:
- realized_vol < 15 → "LOW" (full allocation OK)
- 15 <= realized_vol < 25 → "NORMAL" (standard allocation)
- 25 <= realized_vol < 35 → "HIGH" (reduce allocation 50%)
- realized_vol >= 35 → "EXTREME" (go to 100% cash)

### Options Flow Integration (Unusual Whales)
TQQQ has extremely active options flow — 70+ alerts per day observed.
The data is available via get_stock_flow_alerts("TQQQ") and includes:
- type (call/put)
- total_premium
- total_bid_side_prem / total_ask_side_prem
- volume, open_interest, strike, expiry

Use as a SENTIMENT OVERLAY, not a primary signal:
```python
def get_flow_sentiment():
    alerts = uw_client.get_stock_flow_alerts("TQQQ")
    # Filter to recent alerts (last 4 hours)
    # Sum put premium vs call premium
    # If put_premium > 2x call_premium → bearish → reduce allocation 25%
    # This is additive to regime + momentum + vol logic
```

IMPORTANT: This is best-effort. If the UW API is unavailable or returns an error,
skip the flow gate entirely and proceed with regime + momentum + vol logic.
Do NOT let a UW API failure prevent trading.

### Backtest — Split-Adjusted Prices
TQQQ underwent a 10:1 reverse split in January 2022.
- Alpaca's historical bars API returns split-adjusted data by default.
- BUT verify: pre-2022 TQQQ prices should be in the ~$20-80 range (adjusted).
  If they show $2-8, the data is NOT adjusted and SMA/P&L calculations will be wrong.
- Add a sanity check in the backtest:
```python
if any(bar.close < 5 for bar in tqqq_bars if bar.timestamp.year < 2022):
    logger.warning("TQQQ pre-2022 prices appear unadjusted for reverse split!")
    # Either skip pre-split period or apply 10x multiplier
```

### Error Handling
- If Alpaca API fails → retry 3 times with 30s delay, then alert via Telegram
- If data is stale (bars older than 1 trading day) → do NOT trade, alert
- If insufficient bars for SMA_250 (<250 trading days) → do NOT trade, alert
- If position calculation results in negative shares → cap at 0
- If market closes early (half day) → use half-day execution time
- If UW API fails → skip options flow gate, proceed with core logic
- Log every decision, every gate check, every order attempt

### Telegram Notifications
Send a daily summary after each run:

```
📊 Leveraged ETF Daily Report
━━━━━━━━━━━━━━━━━━━━━━━

Regime: 🟢 STRONG_BULL (Day 15)
QQQ: $609.65 (+2.1%)
  └ SMA50: $585.12 (+4.2% above)
  └ SMA250: $542.30 (+12.4% above)

Momentum: 0.72 (Strong)
Realized Vol: 18.2% (Normal)
Options Flow: Neutral (P/C ratio: 0.9)

Data Quality: ✅ 278 trading days loaded
Gate Check: ✅ All 14 gates passed
PDT Status: 6 day trades remaining

Action: REBALANCE → Buy 5 shares TQQQ
Target: 65% of $38.7K allocated ($25,155)
Current: $24,850 → $25,155

Position: 497 shares TQQQ @ $50.64
P/L: +$1,240 (+5.1%) since entry
Benchmark: QQQ +2.3% (outperforming)
━━━━━━━━━━━━━━━━━━━━━━━
```

When regime changes, send an ALERT:

```
🚨 REGIME CHANGE ALERT
━━━━━━━━━━━━━━━━━━━━━━━
BULL → RISK_OFF

QQQ broke below 250-day SMA ($542.30)
Current: $535.80 (-1.2% below)

Realized Vol: 28.5% (High)
Options Flow: Bearish (P/C ratio: 2.4)

ACTION: Selling all 497 shares TQQQ
Est. proceeds: ~$25,168

Strategy going to 100% CASH
Will re-enter when QQQ reclaims SMA250 + 2%
━━━━━━━━━━━━━━━━━━━━━━━
```

Half-day alert:

```
⏰ HALF DAY DETECTED
━━━━━━━━━━━━━━━━━━━━━━━
Market closes at 1:00 PM EST today
Execution moved to 12:45 PM EST
━━━━━━━━━━━━━━━━━━━━━━━
```

## TESTING REQUIREMENTS

Before deploying:
1. Run `leveraged_etf_job.py status` and verify:
   - >= 250 trading days loaded for SMA calculation
   - Realized vol is in a sensible range (10-30 typically)
   - No reference to VIX or VIXY anywhere in output
   - Options flow data loads (or gracefully skips if UW unavailable)
   - Half-day detection works (test against Alpaca calendar)
2. Run `leveraged_etf_job.py backtest` and verify:
   - Strategy beats QQQ buy-and-hold on risk-adjusted basis
   - Max drawdown < 50% (if higher, tighten guardrails)
   - Strategy correctly exits before major crashes (2020 COVID, 2022 bear)
   - No trades executed during RISK_OFF periods
   - TQQQ prices look correct pre/post Jan 2022 reverse split
   - Backtest uses 400-day lookback for SMAs, not 300
3. Run `leveraged_etf_job.py run` on paper account during market hours
4. Verify Telegram notifications arrive correctly
5. Verify database logging captures all fields (including new vol/flow fields)
6. Verify capital isolation (momentum agent positions untouched)
7. Verify PDT day trade tracking works

## DEPENDENCIES

Should already be installed:
- alpaca-py
- python-telegram-bot  
- python-dotenv
- sqlite3 (stdlib)
- numpy (for realized vol calculation)

May need to install:
- pandas (for backtest and data manipulation) — pip install pandas

## STYLE GUIDE

Follow the existing codebase patterns:
- print() with timestamps for logging
- try/except with descriptive error messages
- Type hints on all functions
- Docstrings explaining the "why" not just the "what"
- Config at top of file, not buried in functions
- Telegram messages with emoji and clean formatting
- Never reference VIX, VIXY, or ^VIX in code — use realized vol exclusively