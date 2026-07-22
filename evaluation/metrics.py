"""
Evaluation Metrics Calculator Engine.
Computes MAE, MSE, RMSE, MAPE, SMAPE, R2 Score, and Directional Accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Calculates quantitative performance metrics for time-series forecasting."""

    @staticmethod
    def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        # Prevent division by zero
        zero_mask = denominator == 0
        denominator[zero_mask] = 1e-8
        
        diff = np.abs(y_pred - y_true) / denominator
        diff[zero_mask] = 0.0
        return float(np.mean(diff) * 100)

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates Mean Absolute Percentage Error (MAPE)."""
        eps = 1e-8
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    @classmethod
    def calculate_directional_accuracy(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates Directional Accuracy (Hit Ratio %).
        Measures percentage of correctly predicted price movement directions (up/down).
        """
        if len(y_true) < 2:
            return 0.0

        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)

        # True if both direction signs match (positive product)
        correct_directions = (true_diff * pred_diff) > 0
        return float(np.mean(correct_directions) * 100)

    @classmethod
    def evaluate_all(cls, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluates predictions across all standard time-series metrics.
        
        Args:
            y_true: 1D array of ground truth target values.
            y_pred: 1D array of predicted target values.
            model_name: Name label for reporting.
            
        Returns:
            Dict containing computed metrics.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = cls.calculate_mape(y_true, y_pred)
        smape = cls.calculate_smape(y_true, y_pred)
        da = cls.calculate_directional_accuracy(y_true, y_pred)

        return {
            "Model": model_name,
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2_Score": round(float(r2), 4),
            "MAPE (%)": round(float(mape), 2),
            "SMAPE (%)": round(float(smape), 2),
            "Hit_Ratio (%)": round(float(da), 2)
        }