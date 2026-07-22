"""
TFT Architecture Test Script.
Validates the construction and training loop of the Temporal Fusion Transformer.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from data_pipeline.ingest.historical import HistoricalDataFetcher
from data_pipeline.ingest.alternative import FearAndGreedFetcher
from data_pipeline.features.technical_indicators import FeatureEngineer
from data_pipeline.process.preprocessor import DataPreprocessor
from data_pipeline.process.sequence_generator import SequenceGenerator

from models.tft.builder import TFTBuilder
from training.trainer import TFTTrainer


def main():
    print("=== 1. Preparing Data for TFT ===")
    df_btc = HistoricalDataFetcher.fetch_yfinance("BTC-USD", period="1y")
    df_fng = FearAndGreedFetcher.fetch_historical()

    df_btc['date_only'] = df_btc['timestamp'].dt.date
    df_fng['date_only'] = df_fng['timestamp'].dt.date
    df_fng_clean = df_fng[['date_only', 'fng_value']].drop_duplicates(subset=['date_only'])
    df_merged = pd.merge(df_btc, df_fng_clean, on='date_only', how='left').drop(columns=['date_only'])

    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df_merged)
    df_features = engineer.add_lag_and_rolling_features(df_features)
    df_features = engineer.add_date_features(df_features)

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df_features)

    feature_cols = ['volume', 'rsi', 'macd', 'bb_width', 'fng_value', 'atr']
    df_clean = preprocessor.handle_outliers(df_clean, feature_cols)
    df_scaled = preprocessor.fit_transform(df_clean, feature_cols)

    print("=== 2. Generating PyTorch Forecasting Datasets ===")
    train_ds, val_ds = SequenceGenerator.create_datasets(
        df_scaled, max_encoder_length=30, max_prediction_length=7
    )

    train_dataloader = train_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

    print("=== 3. Building TFT Model ===")
    # Using tiny hyper-parameters just for testing
    tft_model = TFTBuilder.build_model(
        training_dataset=train_ds,
        learning_rate=0.03,
        hidden_size=8,
        attention_head_size=1,
        dropout=0.1
    )

    print("=== 4. Starting Training Loop (3 Epochs) ===")
    trainer = TFTTrainer.get_trainer(max_epochs=3, use_gpu=True)
    trainer.fit(
        tft_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    print("\n✅ TFT Model compiled and trained successfully!")

if __name__ == "__main__":
    main()