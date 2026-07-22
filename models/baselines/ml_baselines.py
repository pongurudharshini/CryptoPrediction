"""
Tabular ML Baseline Models (XGBoost & Random Forest).
Uses feature matrices with lagged attributes for tabular time-series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from evaluation.metrics import ModelEvaluator


class TabularMLBaselines:
    """Wrapper for traditional Machine Learning baseline algorithms."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)

    def train_and_evaluate(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fits Random Forest and Gradient Boosting models and evaluates performance.
        
        Returns:
            Tuple of metric dictionaries for Random Forest and Gradient Boosting.
        """
        # Clean any remaining NaNs in features
        X_train_clean = X_train.ffill().bfill().fillna(0)
        X_test_clean = X_test.ffill().bfill().fillna(0)

        # 1. Random Forest
        self.rf_model.fit(X_train_clean, y_train)
        rf_preds = self.rf_model.predict(X_test_clean)
        rf_metrics = ModelEvaluator.evaluate_all(y_test.values, rf_preds, model_name="Random Forest")

        # 2. Gradient Boosting (XGBoost Equivalent)
        self.gb_model.fit(X_train_clean, y_train)
        gb_preds = self.gb_model.predict(X_test_clean)
        gb_metrics = ModelEvaluator.evaluate_all(y_test.values, gb_preds, model_name="Gradient Boosting")

        return rf_metrics, gb_metrics