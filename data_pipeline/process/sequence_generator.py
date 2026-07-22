"""
Sequence Generation for PyTorch Forecasting.
Structures tabular data into sliding windows for the Temporal Fusion Transformer.
"""

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from typing import Tuple

class SequenceGenerator:
    
    @staticmethod
    def create_datasets(
        df: pd.DataFrame, 
        max_encoder_length: int = 30, 
        max_prediction_length: int = 7
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Creates PyTorch Forecasting datasets.
        
        Args:
            df: Cleaned and scaled DataFrame.
            max_encoder_length: Number of days to look back.
            max_prediction_length: Number of days to predict into the future.
            
        Returns:
            Tuple of (training_dataset, validation_dataset)
        """
        # Ensure correct types
        df['symbol'] = df['symbol'].astype(str)
        df['month'] = df['month'].astype(str)
        
        # Define the training cutoff (leave last 10% for validation)
        training_cutoff = df['time_idx'].max() - int(len(df) * 0.1)

        # Create the Training Dataset
        training_dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="close",
            group_ids=["symbol"],
            min_encoder_length=max_encoder_length // 2, 
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["symbol"], # Things that never change
            time_varying_known_categoricals=["month"], # Things we know in the future
            time_varying_known_reals=["time_idx", "day_of_week_sin", "day_of_week_cos"], 
            time_varying_unknown_reals=[
                "close", "volume", "rsi", "macd", "bb_width", 
                "fng_value", "ma_crossover", "atr"
            ], # Things we don't know in the future (the model must figure this out)
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Create Validation Dataset utilizing the same parameters
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, 
            df, 
            predict=True, 
            stop_randomization=True
        )

        return training_dataset, validation_dataset