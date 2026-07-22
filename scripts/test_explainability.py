"""
Integration test for TFT Explainability and Visualization.
Generates PNG charts for Predictions, Feature Importance, and Attention.
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
from evaluation.explainer import ModelExplainer


def main():
    print("=== 1. Loading & Preparing Dataset ===")
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

    train_ds, val_ds = SequenceGenerator.create_datasets(
        df_scaled, max_encoder_length=30, max_prediction_length=7
    )

    train_loader = train_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

    print("=== 2. Quick Training TFT Model ===")
    model = TFTBuilder.build_model(train_ds, learning_rate=0.03, hidden_size=8, attention_head_size=1)
    
    # Train for just 1 epoch to get weights populated for evaluation
    trainer = TFTTrainer.get_trainer(max_epochs=1, use_gpu=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n=== 3. Generating Explainability Visualizations ===")
    explainer = ModelExplainer(model, val_ds)
    
    explainer.plot_prediction_intervals(val_loader, num_plots=2)
    explainer.plot_feature_importance()
    explainer.plot_attention_heatmap()
    
    print("\n✅ Explainability charts generated successfully! Check the 'visualizations/' folder.")

if __name__ == "__main__":
    main()