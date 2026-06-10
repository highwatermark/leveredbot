# Leveraged ETF Trading System — Claude Code Session

## Project Overview

Build a production-ready TQQQ/SQQQ trading system based on analysis of the **Adaptive Investments** strategy (Collective2 #148705494), which achieved 130% annual returns with a 2.17 Sharpe ratio.

### Core Philosophy
> "Do what would've worked best, given historically similar conditions."

The system uses k-NN (k-Nearest Neighbors) machine learning to predict next-day Nasdaq-100 direction, then trades leveraged ETFs (TQQQ for long, SQQQ for short) with windowed position management.

---

## Key Insight: Windowed Exit Timing

**The 9:30-10:30 AM exit window is NOT arbitrary** — it's optimized through backtesting:

1. **Overnight gaps resolve by 10:30** — Morning session prices in all overnight news
2. **Midday is noise** — Trading 11:00-3:00 destroys the close-to-close edge  
3. **k-NN trained on close-to-close** — Exiting mid-day breaks the statistical foundation

### Trading Windows

```
PREMARKET (4:00-9:30)
├── Monitor overnight gaps
├── Poll: 30 seconds
└── Actions: None

MORNING CHECK (9:30-10:30) ← PRIMARY EXIT WINDOW
├── Full exit evaluation
├── Stop loss, profit taking, trailing stops
├── Scaling into winners
├── Poll: 5 seconds
└── This is when overnight moves "resolve"

MIDDAY HOLD (10:30-15:55) ← DISCIPLINE ZONE
├── ONLY catastrophic stop (6% for black swans)
├── Trailing stop updates silently
├── Poll: 30 seconds
└── Preserves close-to-close edge

EOD SIGNAL (15:55-16:00) ← ENTRY WINDOW
├── k-NN prediction at 15:58
├── New entries at 15:59
├── Position flips if signal reverses
└── Poll: 1 second
```

---

## Reference Strategy: Adaptive Investments

### Performance Metrics (Verified on Collective2)
- **Annual Return:** 130.2% (compounded)
- **Max Drawdown:** 19.58%
- **Sharpe Ratio:** 2.17
- **Sortino Ratio:** 4.75
- **Calmar Ratio:** 11.19
- **Win Rate:** 51.6%
- **Profit Factor:** 2.3:1
- **Win Months:** 85% (17 of 20)
- **Avg Win:** $1,395 | **Avg Loss:** $662 (2.1:1 ratio)

### Observed Trade Patterns
From AutoTrade data analysis:

| Pattern | Observation |
|---------|-------------|
| Entry Time | Always 15:59 ET (no exceptions) |
| Morning Exits | 9:32, 9:34, 9:42, 9:46, 10:01, 10:22, 10:34 |
| EOD Exits | 15:59 (flipping to opposite direction) |
| Position Sizing | 342-2,731 shares (dynamic based on conviction) |
| Holding Period | 1-9 days |
| Scaling | Multiple fills to build/exit positions |

### Monthly Returns (Notable)
- **April 2025:** +50.7% (tariff shock — system caught regime shift)
- **2025 Full Year:** +181.9%
- **2026 YTD:** +25.0%

---

## Technical Architecture

### 1. K-NN Signal Generator

**16 Features for similarity matching:**

```python
# Price-based
intraday_return      # Open to current
prior_day_return     # Yesterday close-to-close
two_day_return       # 2-day cumulative
five_day_return      # 5-day cumulative

# Volatility
intraday_range       # (High - Low) / Open
atr_ratio            # Current ATR vs 20-day avg

# Trend
distance_from_20ma   # % from 20-day MA
distance_from_50ma   # % from 50-day MA
distance_from_200ma  # % from 200-day MA
ma_20_50_cross       # 20MA - 50MA normalized

# Momentum
rsi_14               # RSI 14-period
rsi_deviation        # RSI distance from 50
momentum_10          # 10-day momentum

# Volume
volume_ratio         # Today vs 20-day avg

# VIX
vix_level            # Current VIX
vix_change           # Daily VIX change
```

**Prediction Flow:**
```
15:58 ET → Calculate today's features
         → Find k most similar historical days (k=7)
         → Weight by distance (closer = more weight)
         → Predict direction + confidence (0-1)
         → If confidence > 0.55: Take position
         → If confidence < 0.55: Stay flat
```

### 2. Position Manager (Windowed)

**Configuration Profiles:**

| Profile | Stop Loss | Profit Targets | Catastrophic | Scaling |
|---------|-----------|----------------|--------------|---------|
| conservative | 1.5% | 2/4/6% | 4.5% | 15% × 1 |
| moderate | 2% | 2.5/4.5/7% | 6% | 20% × 2 |
| aggressive | 2.5% | 3/5.5/8.5% | 7.5% | 25% × 3 |
| adaptive_clone | 2% | 2.5/5/8% | 6% | 25% × 2 |

**Exit Logic by Window:**

```python
# MORNING WINDOW (9:30-10:30)
if pnl <= -stop_loss:
    exit_full("stop_loss")
elif trailing_stop_active and price <= trailing_stop_price:
    exit_full("trailing_stop")
elif pnl >= profit_threshold_1 and partial_exits < 1:
    exit_partial(25%, "profit_take_1")
elif pnl >= profit_threshold_2 and partial_exits < 2:
    exit_partial(25%, "profit_take_2")
elif pnl >= profit_threshold_3 and partial_exits < 3:
    exit_partial(25%, "profit_take_3")

# MIDDAY HOLD (10:30-15:55)
if pnl <= -catastrophic_threshold:  # e.g., -6%
    exit_full("catastrophic_stop")
else:
    update_trailing_stop_silently()
    hold_position()

# EOD SIGNAL (15:55-16:00)
new_signal = knn.predict(current_features)
if new_signal != current_direction:
    exit_full("signal_flip")
    enter_new_position(new_signal)
```

### 3. Alpaca Integration

**Required Components:**
- `StockHistoricalDataClient` — Historical bars for k-NN training
- `StockLatestQuoteRequest` — Real-time prices
- `TradingClient` — Order execution, account info
- Market orders for urgent exits, limit orders for normal

---

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Set up project structure
- [ ] Implement data classes (Position, ExitDecision, ScaleDecision)
- [ ] Build abstract interfaces (MarketDataProvider, SignalProvider, OrderExecutor)
- [ ] Create configuration management with profiles

### Phase 2: K-NN Signal Generator
- [ ] Implement FeatureCalculator with all 16 features
- [ ] Build KNNSignalGenerator with sklearn
- [ ] Add historical data loading from Alpaca
- [ ] Create prediction analysis tools (similar days, confidence)
- [ ] Backtest feature selection

### Phase 3: Position Manager
- [ ] Implement TradingWindow enum and WindowConfig
- [ ] Build windowed polling loop
- [ ] Create morning window exit evaluation
- [ ] Implement midday hold with catastrophic stop
- [ ] Add EOD signal check and entry logic
- [ ] Implement partial profit taking
- [ ] Add trailing stop management
- [ ] Build scaling logic

### Phase 4: Alpaca Integration
- [ ] Implement AlpacaMarketData provider
- [ ] Implement AlpacaSignalProvider with k-NN
- [ ] Implement AlpacaOrderExecutor
- [ ] Add account/position tracking
- [ ] Handle market hours detection

### Phase 5: Testing & Validation
- [ ] Unit tests for each component
- [ ] Paper trading integration tests
- [ ] Backtest against historical data
- [ ] Compare results to Adaptive Investments benchmarks
- [ ] Stress test with volatile periods (e.g., April 2025 tariffs)

### Phase 6: Production Deployment
- [ ] AWS Lightsail or similar always-on hosting
- [ ] Logging and monitoring (CloudWatch)
- [ ] Alerting (SMS/Discord for trades, errors)
- [ ] Dashboard for performance tracking
- [ ] Graceful shutdown handling

---

## File Structure

```
leveraged-etf-system/
├── src/
│   ├── __init__.py
│   ├── config.py              # PositionConfig, profiles
│   ├── models.py              # Position, ExitDecision, etc.
│   ├── interfaces.py          # Abstract base classes
│   ├── position_manager.py    # Windowed position management
│   ├── knn_signal.py          # K-NN signal generator
│   ├── features.py            # Feature calculation
│   ├── alpaca/
│   │   ├── __init__.py
│   │   ├── market_data.py     # AlpacaMarketData
│   │   ├── signal_provider.py # AlpacaSignalProvider
│   │   └── executor.py        # AlpacaOrderExecutor
│   └── utils/
│       ├── __init__.py
│       ├── volatility.py      # VolatilityTracker
│       └── logging.py         # Logging configuration
├── tests/
│   ├── test_position_manager.py
│   ├── test_knn_signal.py
│   └── test_features.py
├── scripts/
│   ├── backtest.py
│   ├── train_model.py
│   └── run_paper.py
├── config/
│   └── profiles.yaml
├── requirements.txt
├── README.md
└── run.py                     # Main entry point
```

---

## Key Code Snippets

### Trading Window Detection

```python
from datetime import time
from enum import Enum

class TradingWindow(Enum):
    PREMARKET = "premarket"
    MORNING_CHECK = "morning_check"
    MIDDAY_HOLD = "midday_hold"
    EOD_SIGNAL = "eod_signal"
    AFTER_HOURS = "after_hours"

WINDOWS = {
    TradingWindow.PREMARKET: (time(4, 0), time(9, 30)),
    TradingWindow.MORNING_CHECK: (time(9, 30), time(10, 30)),
    TradingWindow.MIDDAY_HOLD: (time(10, 30), time(15, 55)),
    TradingWindow.EOD_SIGNAL: (time(15, 55), time(16, 0)),
    TradingWindow.AFTER_HOURS: (time(16, 0), time(20, 0)),
}

def get_current_window() -> TradingWindow:
    now = datetime.now().time()
    for window, (start, end) in WINDOWS.items():
        if start <= now < end:
            return window
    return TradingWindow.AFTER_HOURS
```

### K-NN Prediction

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class KNNSignalGenerator:
    def __init__(self, n_neighbors=7, min_confidence=0.55):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance'
        )
        self.scaler = StandardScaler()
        self.min_confidence = min_confidence
    
    def fit(self, features: np.ndarray, next_day_returns: np.ndarray):
        labels = (next_day_returns > 0).astype(int)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, labels)
    
    def predict(self, current_features: np.ndarray) -> tuple[int, float]:
        scaled = self.scaler.transform(current_features.reshape(1, -1))
        prediction = self.model.predict(scaled)[0]
        probabilities = self.model.predict_proba(scaled)[0]
        confidence = probabilities[prediction]
        return prediction, confidence
```

### Morning Exit Evaluation

```python
async def evaluate_morning_exit(
    self, 
    current_price: float,
    pnl_pct: float
) -> ExitDecision:
    # 1. Hard stop loss
    if pnl_pct <= -self.config.initial_stop_loss:
        return ExitDecision(
            should_exit=True,
            reason=ExitReason.STOP_LOSS,
            shares_to_exit=self.position.shares,
            urgency="immediate"
        )
    
    # 2. Trailing stop
    if self.position.trailing_stop_active:
        if current_price <= self.position.trailing_stop_price:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                shares_to_exit=self.position.shares,
                urgency="immediate"
            )
    
    # 3. Partial profit taking
    if pnl_pct >= self.config.partial_profit_threshold_1:
        if self.position.partial_exits < 1:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.PARTIAL_PROFIT,
                shares_to_exit=self.position.shares // 4,
                urgency="normal"
            )
    
    return ExitDecision(should_exit=False)
```

---

## Environment Setup

### Requirements

```
alpaca-py>=0.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
asyncio
```

### Environment Variables

```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_PAPER=true  # Set to false for live trading
```

---

## Risk Management Rules

1. **Position Sizing:** Max 95% of account in single position
2. **Stop Loss:** 2% initial, 4% max (volatility-adjusted)
3. **Catastrophic Stop:** 6% — only midday protection
4. **Partial Profits:** 25% off at each threshold (2.5%, 4.5%, 7%)
5. **Trailing Stop:** Activates at 2% profit, trails at 1.2%
6. **Max Scale-ins:** 2 per position, 20% each
7. **Daily Loss Limit:** Consider adding 5% daily loss halt

---

## Monitoring & Alerts

### Key Metrics to Track
- Daily P/L ($ and %)
- Win rate (rolling 20 trades)
- Profit factor (rolling 20 trades)
- Max drawdown (from peak)
- Sharpe ratio (rolling 30 days)
- Signal confidence distribution

### Alert Conditions
- Trade executed (entry/exit)
- Stop loss triggered
- Catastrophic stop triggered
- System error
- Market data stale (>60s)
- Position size exceeds limits

---

## Session Commands

Start the session with:

```bash
claude --project leveraged-etf-system
```

### Useful Prompts

**Initialize project:**
> "Set up the project structure with all the directories and files outlined in the plan. Create placeholder files with docstrings explaining their purpose."

**Implement k-NN:**
> "Implement the KNNSignalGenerator class with all 16 features. Include methods for training, prediction, and analyzing similar historical days."

**Build position manager:**
> "Implement the windowed PositionManager with morning check, midday hold, and EOD signal windows. Include all exit logic and partial profit taking."

**Add Alpaca integration:**
> "Implement the Alpaca providers: AlpacaMarketData, AlpacaSignalProvider, and AlpacaOrderExecutor. Handle authentication and error cases."

**Run backtest:**
> "Create a backtest script that simulates the strategy on 2 years of historical data. Calculate Sharpe ratio, max drawdown, and monthly returns."

**Deploy to production:**
> "Set up the system for AWS Lightsail deployment with systemd service, CloudWatch logging, and Discord webhook alerts."

---

## Success Criteria

The system is successful when:

1. **Sharpe > 1.5** on paper trading over 30 days
2. **Max drawdown < 25%** 
3. **Win rate > 45%** with profit factor > 1.5
4. **No missed signals** — 100% execution at 15:59
5. **No midday exits** except catastrophic stops
6. **Clean logs** — all trades documented with reasoning

---

## Reference Materials

- **Adaptive Investments C2:** https://collective2.com/details-list/148705494
- **Alpaca API Docs:** https://docs.alpaca.markets/
- **scikit-learn k-NN:** https://scikit-learn.org/stable/modules/neighbors.html
- **TQQQ/SQQQ Info:** ProShares UltraPro QQQ / Short QQQ

---

*Last updated: February 2026*
*Author: Hari Lakshmanan*
