"""
Tests the entire Feature Engineering and Preprocessing pipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_pipeline.ingest.historical import HistoricalDataFetcher
from data_pipeline.ingest.alternative import FearAndGreedFetcher
from data_pipeline.features.technical_indicators import FeatureEngineer
from data_pipeline.process.preprocessor import DataPreprocessor
from data_pipeline.process.sequence_generator import SequenceGenerator
import pandas as pd

def main():
    print("=== 1. Fetching Raw Data ===")
    df_btc = HistoricalDataFetcher.fetch_yfinance("BTC-USD", period="1y")
    df_btc['symbol'] = "BTC"
    
    df_fng = FearAndGreedFetcher.fetch_historical()
    
    # Merge Price with Fear & Greed on Date
    df_btc['date_only'] = df_btc['timestamp'].dt.date
    df_fng['date_only'] = df_fng['timestamp'].dt.date
    df_merged = pd.merge(df_btc, df_fng, on='date_only', how='left').drop(columns=['date_only'])
    
    print("=== 2. Feature Engineering ===")
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df_merged)
    df_features = engineer.add_lag_and_rolling_features(df_features)
    df_features = engineer.add_date_features(df_features)
    
    print(f"Columns generated: {len(df_features.columns)}")
    
    print("=== 3. Preprocessing (Clean & Scale) ===")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df_features)
    
    # Define features to scale
    numerical_features = ['volume', 'rsi', 'macd', 'bb_width', 'fng_value', 'atr']
    df_clean = preprocessor.handle_outliers(df_clean, numerical_features)
    df_scaled = preprocessor.fit_transform(df_clean, numerical_features)
    
    print(f"Final Data Shape: {df_scaled.shape}")
    print(f"Any NaNs remaining?: {df_scaled.isna().any().any()}")
    
    print("=== 4. PyTorch Dataset Generation ===")
    train_ds, val_ds = SequenceGenerator.create_datasets(df_scaled, max_encoder_length=30, max_prediction_length=7)
    
    # Get a sample batch
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=4, num_workers=0)
    x, y = next(iter(train_dataloader))
    
    print("\nPyTorch Dataloader Test Successful!")
    print(f"Encoder Input Shape (Batch, Time, Features): {x['encoder_cont'].shape}")
    print(f"Target Output Shape (Batch, Time): {y[0].shape}")

if __name__ == "__main__":
    main()