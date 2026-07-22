"""
Data Preprocessing Pipeline.
Handles missing values, outlier winsorization, and feature scaling.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Cleans and scales data for PyTorch consumption."""
    
    def __init__(self, target_col: str = 'close'):
        self.target_col = target_col
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fills gaps, then backward fills initial NaN from rolling windows."""
        logger.info(f"Handling missing values. Initial NaNs: {df.isna().sum().sum()}")
        df = df.copy()
        
        # Interpolate ONLY numeric columns to prevent string/datetime interpolation errors
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=3)
        
        # Forward fill and backward fill remaining NaNs for all columns
        df = df.ffill().bfill()
        return df
        
    def handle_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Clips outliers at 1st and 99th percentiles (Winsorization)."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df
        
    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Fits scalers and transforms the data."""
        df = df.copy()
        
        # Scale features
        df[feature_cols] = self.feature_scaler.fit_transform(df[feature_cols])
        
        # Scale target separately so we can inverse_transform predictions later
        df[[self.target_col]] = self.target_scaler.fit_transform(df[[self.target_col]])
        
        # Time index required by PyTorch Forecasting (must be continuous integer)
        df['time_idx'] = range(len(df))
        
        # Group ID required by PyTorch Forecasting
        if 'symbol' not in df.columns:
            df['symbol'] = 'BTC'
            
        return df