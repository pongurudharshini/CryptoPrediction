"""
Technical Feature Engineering Module.
Calculates momentum, volatility, volume, and trend indicators.
"""

import pandas as pd
import ta
import numpy as np

class FeatureEngineer:
    """Engineers financial features from raw OHLCV data."""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and appends technical indicators to the dataframe.
        Complexity: O(N) where N is the number of rows.
        """
        df = df.copy()
        
        # 1. Momentum Indicators
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['stoch_k'] = stoch.stoch()
        
        # 2. Trend Indicators
        # MACD
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Moving Averages (EMA, SMA)
        df['sma_20'] = ta.trend.sma_indicator(close=df['close'], window=20)
        df['ema_20'] = ta.trend.ema_indicator(close=df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(close=df['close'], window=50)
        # Moving Average Crossover (1 if EMA20 > EMA50 else 0)
        df['ma_crossover'] = (df['ema_20'] > df['ema_50']).astype(int)
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        
        # 3. Volatility Indicators
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']  # Normalized width
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # 4. Volume Indicators
        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14
        ).volume_weighted_average_price()
        
        return df

    @staticmethod
    def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds lag features and rolling statistics to capture temporal dependencies."""
        df = df.copy()
        
        # Daily Returns (Momentum)
        df['return_1d'] = df['close'].pct_change(1)
        
        # Lag features (past prices)
        for lag in [1, 3, 7]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            
        # Rolling Statistics (Volatility/Mean)
        df['rolling_mean_7d'] = df['close'].rolling(window=7).mean()
        df['rolling_std_7d'] = df['close'].rolling(window=7).std()
        
        return df

    @staticmethod
    def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts cyclical date features."""
        df = df.copy()
        # Cyclical encoding for day of week to maintain temporal continuity
        day_of_week = df['timestamp'].dt.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        df['month'] = df['timestamp'].dt.month.astype(str) # Kept as categorical for TFT
        return df