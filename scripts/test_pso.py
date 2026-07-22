"""
Integration test for PSO Hyperparameter Optimization over Temporal Fusion Transformer.
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
from models.pso.pso_optimizer import PSOOptimizer


def main():
    print("=== 1. Loading & Preparing Dataset for PSO ===")
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

    # 2. Define Objective Function for PSO
    def pso_objective_func(params: dict) -> float:
        """
        Objective function evaluated by each particle.
        Builds a candidate TFT model, trains for 2 epochs, and returns validation loss.
        """
        try:
            model = TFTBuilder.build_model(
                training_dataset=train_ds,
                learning_rate=params['learning_rate'],
                hidden_size=params['hidden_size'],
                attention_head_size=params['attention_head_size'],
                dropout=params['dropout']
            )

            trainer = TFTTrainer.get_trainer(max_epochs=2, use_gpu=True)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                return float(val_loss)
            return 999.0 # Fallback high penalty loss if metric missing
        except Exception as e:
            print(f"Error during particle evaluation: {e}")
            return 999.0

    print("\n=== 2. Defining PSO Search Space ===")
    param_bounds = {
        'learning_rate': (0.001, 0.05, 'float'),
        'hidden_size': (8, 32, 'int'),
        'attention_head_size': (1, 4, 'int'),
        'dropout': (0.05, 0.3, 'float')
    }

    print("\n=== 3. Executing Particle Swarm Optimization ===")
    pso = PSOOptimizer(
        objective_func=pso_objective_func,
        param_bounds=param_bounds,
        num_particles=3,    # Small swarm size for quick execution
        max_iterations=2     # Small iteration limit for verification
    )

    best_params, best_loss, fitness_history = pso.optimize()

    print("\n=========================================================")
    print("         PSO OPTIMIZATION COMPLETED SUCCESSFULLY          ")
    print("=========================================================")
    print(f"Optimal Hyperparameters Found:")
    for param, val in best_params.items():
        print(f"  └─ {param}: {val}")
    print(f"Best Validation Loss: {best_loss:.5f}")
    print(f"Fitness History Across Iterations: {fitness_history}")
    print("=========================================================")


if __name__ == "__main__":
    main()