"""
Historical Data Ingestion Pipeline.
Downloads OHLCV data from Yahoo Finance and Binance REST API.
"""

import yfinance as yf
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """Class to fetch historical data from multiple sources."""
    
    @staticmethod
    def fetch_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical data using Yahoo Finance.
        
        Args:
            symbol (str): Ticker symbol (e.g., 'BTC-USD').
            period (str): Data period ('1y', '2y', 'max').
            interval (str): Granularity ('1d', '1h').
            
        Returns:
            pd.DataFrame: Cleaned OHLCV data.
        """
        logger.info(f"Fetching {symbol} data from Yahoo Finance...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
            
        df = df.reset_index()
        # Normalize column names
        df.rename(columns={
            "Date": "timestamp", "Datetime": "timestamp", "index": "timestamp",
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        }, inplace=True)
        
        # Ensure timestamp is datetime in UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Keep only relevant columns
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    @staticmethod
    def fetch_binance(symbol: str, interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """
        Fetches historical klines (candlesticks) from Binance REST API.
        
        Args:
            symbol (str): Binance pair (e.g., 'BTCUSDT').
            interval (str): Kline interval ('1h', '1d').
            limit (int): Max records (up to 1000).
            
        Returns:
            pd.DataFrame: Cleaned OHLCV data.
        """
        logger.info(f"Fetching {symbol} data from Binance...")
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        
        # Convert types and normalize timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]