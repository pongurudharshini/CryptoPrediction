"""
Baseline Model Benchmarking Suite.
Runs data preprocessing, fits classical ML and DL baselines, and generates comparison metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from data_pipeline.ingest.historical import HistoricalDataFetcher
from data_pipeline.ingest.alternative import FearAndGreedFetcher
from data_pipeline.features.technical_indicators import FeatureEngineer
from data_pipeline.process.preprocessor import DataPreprocessor
from models.baselines.ml_baselines import TabularMLBaselines
from models.baselines.dl_baselines import PyTorchRecurrentBaseline, train_recurrent_baseline


def prepare_sequence_data(df: pd.DataFrame, feature_cols: list, target_col: str = 'close', seq_len: int = 30):
    """Converts 2D tabular features into 3D sequence matrices for PyTorch models."""
    X_seq, y_seq = [], []
    data_feat = df[feature_cols].values
    data_target = df[target_col].values

    for i in range(len(df) - seq_len):
        X_seq.append(data_feat[i:i + seq_len])
        y_seq.append(data_target[i + seq_len])

    return np.array(X_seq), np.array(y_seq)


def main():
    print("=== 1. Ingesting & Engineering Features for Benchmark ===")
    df_btc = HistoricalDataFetcher.fetch_yfinance("BTC-USD", period="1y")
    df_fng = FearAndGreedFetcher.fetch_historical()

    df_btc['date_only'] = df_btc['timestamp'].dt.date
    df_fng['date_only'] = df_fng['timestamp'].dt.date
    
    # Drop timestamp from df_fng before merge to prevent timestamp_x / timestamp_y collisions
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

    # Split train/test (80% train, 20% test chronologically)
    split_idx = int(len(df_scaled) * 0.8)

    print("\n=== 2. Evaluating Tabular ML Baselines (Random Forest & Gradient Boosting) ===")
    X_tab = df_scaled[feature_cols]
    y_tab = df_scaled['close']

    X_train_tab, X_test_tab = X_tab.iloc[:split_idx], X_tab.iloc[split_idx:]
    y_train_tab, y_test_tab = y_tab.iloc[:split_idx], y_tab.iloc[split_idx:]

    ml_benchmarks = TabularMLBaselines()
    rf_res, gb_res = ml_benchmarks.train_and_evaluate(X_train_tab, y_train_tab, X_test_tab, y_test_tab)

    print("\n=== 3. Evaluating Deep Learning Recurrent Baselines (LSTM & GRU) ===")
    seq_len = 15
    X_seq, y_seq = prepare_sequence_data(df_scaled, feature_cols, target_col='close', seq_len=seq_len)

    seq_split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:seq_split_idx], X_seq[seq_split_idx:]
    y_train_seq, y_test_seq = y_seq[:seq_split_idx], y_seq[seq_split_idx:]

    # LSTM Baseline
    lstm_model = PyTorchRecurrentBaseline(input_size=len(feature_cols), cell_type="LSTM")
    lstm_res = train_recurrent_baseline(
        lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, epochs=15, model_name="LSTM Baseline"
    )

    # GRU Baseline
    gru_model = PyTorchRecurrentBaseline(input_size=len(feature_cols), cell_type="GRU")
    gru_res = train_recurrent_baseline(
        gru_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, epochs=15, model_name="GRU Baseline"
    )

    print("\n=========================================================")
    print("           BASELINE MODEL EVALUATION SUMMARY            ")
    print("=========================================================")
    results_df = pd.DataFrame([rf_res, gb_res, lstm_res, gru_res])
    print(results_df.to_string(index=False))
    print("=========================================================")


if __name__ == "__main__":
    main()