"""Typed dataclasses for the strategy pipeline.

Replaces raw dicts flowing between _fetch_all_data(), _compute_signals(),
gate checklist, and sizing. Catches field mismatches at dev time.
"""

from dataclasses import dataclass, field


@dataclass
class MarketData:
    """Output of _fetch_all_data() — all raw market data for the strategy."""
    account: dict
    positions: list[dict]
    tqqq_position: dict | None
    calendar: dict | None
    is_half_day: bool
    snapshots: dict
    qqq_bars: list[dict]
    qqq_closes: list[float]
    qqq_current: float | None
    tqqq_price: float | None
    daily_loss_pct: float
    trading_days_fetched: int
    # SQQQ data (optional, only when use_sqqq_trading enabled)
    sqqq_position: dict | None = None
    sqqq_price: float | None = None


@dataclass
class StrategySignals:
    """Output of _compute_signals() — all derived signals for gate/sizing."""
    qqq_close: float
    sma_50: float
    sma_250: float
    pct_above_sma50: float
    pct_above_sma250: float
    momentum: dict
    momentum_score: float
    realized_vol: float
    vol_regime: str
    vol_adjustment: float
    flow: dict
    options_flow_bearish: bool
    options_flow_adjustment: float
    options_flow_ratio: float
    raw_regime: str
    effective_regime: str
    previous_regime: str | None
    regime_hold_days: int
    regime_changed: bool
    capital: dict
    allocated_capital: float
    current_shares: int
    tqqq_price: float
    daily_loss_pct: float
    qqq_closes: list[float]
    trading_days_fetched: int
    day_trades_remaining: int
    account_equity: float
    cash_balance: float
    consecutive_losing_days: int
    # SQQQ state
    sqqq_price: float = 0.0
    sqqq_current_shares: int = 0
    has_tqqq_position: bool = False
    # k-NN signal overlay
    knn_direction: str = "FLAT"
    knn_confidence: float = 0.5
    knn_adjustment: float = 1.0
    knn_probabilities: list[float] = field(default_factory=lambda: [0.5, 0.5])
