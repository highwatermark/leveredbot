"""
K-NN Signal Generator & Alpaca Integration
==========================================
Implements the nearest-neighbor prediction algorithm
inspired by Adaptive Investments strategy.

Features:
- k-NN based next-day direction prediction
- Feature engineering for market similarity
- Alpaca integration for real-time data and execution
- Continuous polling integration

Author: Hari Lakshmanan
Date: February 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import asyncio

logger = logging.getLogger('KNN_Signal')


@dataclass
class MarketFeatures:
    """Features used for k-NN similarity matching"""
    timestamp: datetime
    
    # Price-based features
    intraday_return: float        # Open to current (or close)
    prior_day_return: float       # Yesterday's close-to-close
    two_day_return: float         # 2-day cumulative return
    five_day_return: float        # 5-day cumulative return
    
    # Volatility features
    intraday_range: float         # (High - Low) / Open
    atr_ratio: float              # Current ATR vs 20-day average ATR
    
    # Trend features
    distance_from_20ma: float     # % distance from 20-day MA
    distance_from_50ma: float     # % distance from 50-day MA
    distance_from_200ma: float    # % distance from 200-day MA
    ma_20_50_cross: float         # 20MA - 50MA (normalized)
    
    # Momentum features
    rsi_14: float                 # RSI 14-period
    rsi_deviation: float          # RSI distance from 50
    momentum_10: float            # 10-day momentum
    
    # Volume features
    volume_ratio: float           # Today's volume vs 20-day average
    
    # VIX features (if available)
    vix_level: float = 0.0
    vix_change: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for k-NN"""
        return np.array([
            self.intraday_return,
            self.prior_day_return,
            self.two_day_return,
            self.five_day_return,
            self.intraday_range,
            self.atr_ratio,
            self.distance_from_20ma,
            self.distance_from_50ma,
            self.distance_from_200ma,
            self.ma_20_50_cross,
            self.rsi_14,
            self.rsi_deviation,
            self.momentum_10,
            self.volume_ratio,
            self.vix_level,
            self.vix_change
        ])


class KNNSignalGenerator:
    """
    K-Nearest Neighbors signal generator for next-day prediction.
    
    Core philosophy: "Do what would've worked best, given historically similar conditions."
    """
    
    def __init__(
        self,
        n_neighbors: int = 7,
        weight_scheme: str = 'distance',
        lookback_years: int = 10,
        min_confidence: float = 0.55
    ):
        self.n_neighbors = n_neighbors
        self.weight_scheme = weight_scheme
        self.lookback_years = lookback_years
        self.min_confidence = min_confidence
        
        self.model: Optional[KNeighborsClassifier] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Historical data storage
        self.features_history: List[MarketFeatures] = []
        self.returns_history: List[float] = []  # Next-day returns
        
        # Feature importance tracking
        self.feature_names = [
            'intraday_return', 'prior_day_return', 'two_day_return', 'five_day_return',
            'intraday_range', 'atr_ratio',
            'distance_from_20ma', 'distance_from_50ma', 'distance_from_200ma', 'ma_20_50_cross',
            'rsi_14', 'rsi_deviation', 'momentum_10',
            'volume_ratio', 'vix_level', 'vix_change'
        ]
    
    def fit(self, features: np.ndarray, next_day_returns: np.ndarray):
        """
        Fit the k-NN model on historical data.
        
        Args:
            features: Array of shape (n_samples, n_features)
            next_day_returns: Array of shape (n_samples,) with next-day returns
        """
        # Convert returns to binary classification (up/down)
        labels = (next_day_returns > 0).astype(int)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit k-NN
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weight_scheme,
            metric='euclidean'
        )
        self.model.fit(features_scaled, labels)
        self.is_fitted = True
        
        logger.info(f"K-NN model fitted with {len(features)} samples")
    
    def predict(self, current_features: MarketFeatures) -> Tuple[int, float]:
        """
        Predict next-day direction.
        
        Returns:
            Tuple of (direction, confidence)
            direction: 1 for up (LONG TQQQ), 0 for down (LONG SQQQ)
            confidence: probability of the predicted direction
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        features = current_features.to_array().reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence
    
    def get_similar_days(self, current_features: MarketFeatures, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get the n most similar historical days and their outcomes.
        
        Returns:
            List of (index, distance) tuples for similar days
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = current_features.to_array().reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        distances, indices = self.model.kneighbors(features_scaled, n_neighbors=n)
        
        return list(zip(indices[0], distances[0]))
    
    def analyze_prediction(self, current_features: MarketFeatures) -> Dict:
        """
        Detailed analysis of prediction including similar days.
        """
        prediction, confidence = self.predict(current_features)
        similar_days = self.get_similar_days(current_features, n=self.n_neighbors)
        
        # Get outcomes of similar days
        similar_returns = [self.returns_history[idx] for idx, _ in similar_days]
        
        analysis = {
            'prediction': 'LONG' if prediction == 1 else 'SHORT',
            'confidence': confidence,
            'similar_days_count': len(similar_days),
            'similar_days_avg_return': np.mean(similar_returns),
            'similar_days_win_rate': sum(1 for r in similar_returns if r > 0) / len(similar_returns),
            'similar_days_returns': similar_returns,
            'recommendation': self._get_recommendation(prediction, confidence)
        }
        
        return analysis
    
    def _get_recommendation(self, prediction: int, confidence: float) -> str:
        """Generate trading recommendation based on prediction and confidence"""
        
        if confidence < self.min_confidence:
            return "FLAT - Confidence below threshold"
        
        if confidence >= 0.70:
            size = "FULL"
        elif confidence >= 0.60:
            size = "PARTIAL (70%)"
        else:
            size = "SMALL (50%)"
        
        direction = "LONG TQQQ" if prediction == 1 else "LONG SQQQ"
        
        return f"{direction} - {size} position ({confidence:.1%} confidence)"


class FeatureCalculator:
    """
    Calculates market features from price data.
    """
    
    def __init__(self):
        self.price_history: pd.DataFrame = None
        self.vix_history: pd.DataFrame = None
    
    def load_price_data(self, df: pd.DataFrame):
        """
        Load historical price data.
        
        Expected columns: open, high, low, close, volume
        Index should be datetime
        """
        self.price_history = df.copy()
        self._calculate_indicators()
    
    def load_vix_data(self, df: pd.DataFrame):
        """Load VIX data for volatility features"""
        self.vix_history = df.copy()
    
    def _calculate_indicators(self):
        """Pre-calculate technical indicators"""
        df = self.price_history
        
        # Moving averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['ma_200'] = df['close'].rolling(200).mean()
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_20_avg'] = df['atr_14'].rolling(20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Volume average
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        
        # Returns
        df['return_1d'] = df['close'].pct_change()
        df['return_2d'] = df['close'].pct_change(2)
        df['return_5d'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        self.price_history = df
    
    def calculate_features(
        self, 
        date: datetime,
        current_price: float = None,
        current_high: float = None,
        current_low: float = None,
        current_volume: float = None
    ) -> MarketFeatures:
        """
        Calculate features for a specific date/time.
        
        If current_* values provided, calculates intraday features.
        Otherwise uses EOD data from history.
        """
        df = self.price_history
        
        # Find the relevant row
        if isinstance(date, datetime):
            date_only = date.date()
        else:
            date_only = date
        
        # Get historical data up to this date
        mask = df.index.date <= date_only
        hist = df[mask]
        
        if len(hist) < 200:
            raise ValueError("Insufficient historical data (need 200+ days)")
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        # Use provided intraday values or historical
        if current_price is None:
            current_price = latest['close']
        if current_high is None:
            current_high = latest['high']
        if current_low is None:
            current_low = latest['low']
        if current_volume is None:
            current_volume = latest['volume']
        
        # Calculate features
        features = MarketFeatures(
            timestamp=date if isinstance(date, datetime) else datetime.combine(date, time(15, 58)),
            
            # Price returns
            intraday_return=(current_price - latest['open']) / latest['open'],
            prior_day_return=latest['return_1d'],
            two_day_return=latest['return_2d'],
            five_day_return=latest['return_5d'],
            
            # Volatility
            intraday_range=(current_high - current_low) / latest['open'],
            atr_ratio=latest['atr_14'] / latest['atr_20_avg'] if latest['atr_20_avg'] > 0 else 1.0,
            
            # Trend
            distance_from_20ma=(current_price - latest['ma_20']) / latest['ma_20'],
            distance_from_50ma=(current_price - latest['ma_50']) / latest['ma_50'],
            distance_from_200ma=(current_price - latest['ma_200']) / latest['ma_200'],
            ma_20_50_cross=(latest['ma_20'] - latest['ma_50']) / latest['ma_50'],
            
            # Momentum
            rsi_14=latest['rsi_14'],
            rsi_deviation=(latest['rsi_14'] - 50) / 50,
            momentum_10=latest['momentum_10'],
            
            # Volume
            volume_ratio=current_volume / latest['volume_ma_20'] if latest['volume_ma_20'] > 0 else 1.0
        )
        
        # Add VIX features if available
        if self.vix_history is not None:
            vix_mask = self.vix_history.index.date <= date_only
            vix_hist = self.vix_history[vix_mask]
            if len(vix_hist) > 0:
                features.vix_level = vix_hist.iloc[-1]['close']
                if len(vix_hist) > 1:
                    features.vix_change = (vix_hist.iloc[-1]['close'] - vix_hist.iloc[-2]['close']) / vix_hist.iloc[-2]['close']
        
        return features
    
    def build_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training data from historical prices.
        
        Returns:
            Tuple of (features, next_day_returns)
        """
        df = self.price_history.dropna()
        
        features_list = []
        returns_list = []
        
        # Start from day 200 to have enough history for all indicators
        for i in range(200, len(df) - 1):
            date = df.index[i]
            try:
                features = self.calculate_features(date)
                next_day_return = df.iloc[i + 1]['return_1d']
                
                features_list.append(features.to_array())
                returns_list.append(next_day_return)
            except Exception as e:
                logger.warning(f"Error calculating features for {date}: {e}")
                continue
        
        return np.array(features_list), np.array(returns_list)


class AlpacaSignalProvider:
    """
    Alpaca-integrated signal provider using k-NN predictions.
    
    Implements the SignalProvider interface for the PositionManager.
    """
    
    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        paper: bool = True,
        index_symbol: str = "QQQ"  # Use QQQ as proxy for NDX
    ):
        from alpaca.data import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        
        self.data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        self.trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=paper)
        
        self.index_symbol = index_symbol
        
        # Initialize components
        self.feature_calculator = FeatureCalculator()
        self.signal_generator = KNNSignalGenerator()
        
        self._is_initialized = False
        self._last_prediction: Optional[Tuple[int, float]] = None
        self._last_prediction_time: Optional[datetime] = None
    
    async def initialize(self):
        """Load historical data and train the model"""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        logger.info("Initializing Alpaca Signal Provider...")
        
        # Load historical data (10 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)
        
        request = StockBarsRequest(
            symbol_or_symbols=self.index_symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = self.data_client.get_stock_bars(request)
        df = bars.df.reset_index()
        
        # Prepare DataFrame
        df = df.rename(columns={
            'timestamp': 'date',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.set_index('date')
        
        # Load into feature calculator
        self.feature_calculator.load_price_data(df)
        
        # Build training data
        features, returns = self.feature_calculator.build_training_data()
        
        # Train the model
        self.signal_generator.fit(features, returns)
        self.signal_generator.returns_history = returns.tolist()
        
        self._is_initialized = True
        logger.info(f"Signal Provider initialized with {len(features)} training samples")
    
    async def get_current_signal(self) -> 'Direction':
        """Get the current directional signal"""
        from leveraged_etf_position_manager import Direction
        
        if not self._is_initialized:
            await self.initialize()
        
        # Get current features
        features = await self._get_current_features()
        
        # Get prediction
        prediction, confidence = self.signal_generator.predict(features)
        
        self._last_prediction = (prediction, confidence)
        self._last_prediction_time = datetime.now()
        
        if confidence < self.signal_generator.min_confidence:
            return Direction.FLAT
        
        return Direction.LONG if prediction == 1 else Direction.SHORT
    
    async def get_signal_confidence(self) -> float:
        """Get confidence level 0-1"""
        if self._last_prediction is None:
            await self.get_current_signal()
        
        return self._last_prediction[1]
    
    async def _get_current_features(self) -> MarketFeatures:
        """Get current market features from Alpaca"""
        from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
        from alpaca.data.timeframe import TimeFrame
        
        # Get latest quote for current price
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=self.index_symbol)
        quote = self.data_client.get_stock_latest_quote(quote_request)
        current_price = quote[self.index_symbol].ask_price
        
        # Get today's OHLV
        today = datetime.now().date()
        bars_request = StockBarsRequest(
            symbol_or_symbols=self.index_symbol,
            timeframe=TimeFrame.Day,
            start=today,
            end=today + timedelta(days=1)
        )
        
        try:
            today_bars = self.data_client.get_stock_bars(bars_request)
            if len(today_bars.df) > 0:
                today_data = today_bars.df.iloc[-1]
                current_high = today_data['high']
                current_low = today_data['low']
                current_volume = today_data['volume']
            else:
                current_high = current_price
                current_low = current_price
                current_volume = 0
        except:
            current_high = current_price
            current_low = current_price
            current_volume = 0
        
        # Calculate features
        features = self.feature_calculator.calculate_features(
            date=datetime.now(),
            current_price=current_price,
            current_high=current_high,
            current_low=current_low,
            current_volume=current_volume
        )
        
        return features
    
    def get_detailed_analysis(self) -> Dict:
        """Get detailed analysis of current signal"""
        if not self._is_initialized:
            raise ValueError("Provider not initialized")
        
        # This is synchronous - would need async wrapper for live use
        features = self.feature_calculator.calculate_features(datetime.now())
        return self.signal_generator.analyze_prediction(features)


class AlpacaMarketData:
    """
    Alpaca market data provider implementing MarketDataProvider interface.
    """
    
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str):
        from alpaca.data import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        
        self.data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        self.trading_client = TradingClient(alpaca_api_key, alpaca_secret_key)
    
    async def get_current_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestQuoteRequest
        
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data_client.get_stock_latest_quote(request)
        return (quote[symbol].ask_price + quote[symbol].bid_price) / 2
    
    async def get_bid_ask(self, symbol: str) -> Tuple[float, float]:
        from alpaca.data.requests import StockLatestQuoteRequest
        
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data_client.get_stock_latest_quote(request)
        return (quote[symbol].bid_price, quote[symbol].ask_price)
    
    async def get_atr(self, symbol: str, period: int) -> float:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        end = datetime.now()
        start = end - timedelta(days=period * 2)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        bars = self.data_client.get_stock_bars(request)
        df = bars.df
        
        # Calculate ATR
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        return tr.rolling(period).mean().iloc[-1]
    
    def is_market_open(self) -> bool:
        clock = self.trading_client.get_clock()
        return clock.is_open
    
    def time_to_close(self) -> timedelta:
        clock = self.trading_client.get_clock()
        if not clock.is_open:
            return timedelta(hours=0)
        return clock.next_close - clock.timestamp


class AlpacaOrderExecutor:
    """
    Alpaca order executor implementing OrderExecutor interface.
    """
    
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str, paper: bool = True):
        from alpaca.trading.client import TradingClient
        
        self.trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=paper)
    
    async def market_sell(self, symbol: str, shares: int) -> dict:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        request = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(request)
        return {"order_id": order.id, "status": order.status}
    
    async def market_buy(self, symbol: str, shares: int) -> dict:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        request = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(request)
        return {"order_id": order.id, "status": order.status}
    
    async def limit_sell(self, symbol: str, shares: int, price: float) -> dict:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        request = LimitOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=price
        )
        
        order = self.trading_client.submit_order(request)
        return {"order_id": order.id, "status": order.status}
    
    async def limit_buy(self, symbol: str, shares: int, price: float) -> dict:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        request = LimitOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=price
        )
        
        order = self.trading_client.submit_order(request)
        return {"order_id": order.id, "status": order.status}


# =============================================================================
# Complete System Integration
# =============================================================================

async def create_leveraged_etf_system(
    alpaca_api_key: str,
    alpaca_secret_key: str,
    paper: bool = True
) -> 'PositionManager':
    """
    Factory function to create a fully configured leveraged ETF trading system.
    """
    from leveraged_etf_position_manager import PositionManager, PositionConfig, Direction
    
    # Create configuration
    config = PositionConfig(
        # Profit taking - more aggressive for leveraged ETFs
        partial_profit_threshold_1=0.025,  # 2.5% take 25% off
        partial_profit_threshold_2=0.045,  # 4.5% take another 25%
        partial_profit_threshold_3=0.070,  # 7% take another 25%
        
        # Trailing stop
        trailing_stop_activation=0.02,    # Activate at 2% profit
        trailing_stop_distance=0.012,     # 1.2% trail
        
        # Stop loss
        initial_stop_loss=0.02,           # 2% initial
        max_stop_loss=0.04,               # 4% max in high vol
        
        # Scaling
        scale_in_threshold=0.015,         # Scale at 1.5% profit
        scale_in_size_pct=0.20,           # Add 20% of original
        max_scale_ins=2,
        
        # Polling
        poll_interval_normal=3.0,         # 3 second normal polling
        poll_interval_urgent=0.5,         # 500ms when near triggers
    )
    
    # Create providers
    market_data = AlpacaMarketData(alpaca_api_key, alpaca_secret_key)
    signal_provider = AlpacaSignalProvider(alpaca_api_key, alpaca_secret_key, paper)
    order_executor = AlpacaOrderExecutor(alpaca_api_key, alpaca_secret_key, paper)
    
    # Initialize signal provider (loads historical data, trains model)
    await signal_provider.initialize()
    
    # Create position manager
    manager = PositionManager(
        config=config,
        market_data=market_data,
        signal_provider=signal_provider,
        order_executor=order_executor,
        symbols={
            Direction.LONG: "TQQQ",
            Direction.SHORT: "SQQQ"
        }
    )
    
    # Add logging callbacks
    async def on_entry(position):
        logger.info(f"🟢 ENTERED: {position.symbol} | {position.shares} shares @ ${position.entry_price:.2f}")
    
    async def on_exit(position, decision):
        logger.info(f"🔴 EXITED: {decision.reason.value} | {decision.shares_to_exit} shares @ ${decision.exit_price:.2f}")
    
    async def on_scale(position, decision):
        logger.info(f"📈 SCALED: {decision.reason} | +{decision.shares_to_add} shares")
    
    manager.on_entry = on_entry
    manager.on_exit = on_exit
    manager.on_scale = on_scale
    
    return manager


# =============================================================================
# Main entry point
# =============================================================================

async def main():
    """
    Main entry point for the leveraged ETF trading system.
    """
    import os
    
    # Get credentials from environment
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.error("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        return
    
    # Create and start the system
    manager = await create_leveraged_etf_system(api_key, secret_key, paper=True)
    
    logger.info("=" * 60)
    logger.info("Leveraged ETF Trading System Started")
    logger.info("=" * 60)
    logger.info(f"Symbols: TQQQ (long) / SQQQ (short)")
    logger.info(f"Poll Interval: {manager.config.poll_interval_normal}s normal, {manager.config.poll_interval_urgent}s urgent")
    logger.info("=" * 60)
    
    try:
        await manager.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
