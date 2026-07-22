"""
Unit Tests for Feature Engineering Engine.
Tests technical indicator generation on synthetic OHLCV data.
"""

import pytest
import pandas as pd
import numpy as np
from data_pipeline.features.technical_indicators import FeatureEngineer


@pytest.fixture
def dummy_ohlcv_df():
    """Generates synthetic OHLCV data for testing."""
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    data = {
        "timestamp": dates,
        "open": np.random.uniform(50000, 60000, size=100),
        "high": np.random.uniform(60000, 65000, size=100),
        "low": np.random.uniform(45000, 50000, size=100),
        "close": np.random.uniform(50000, 60000, size=100),
        "volume": np.random.uniform(100000, 500000, size=100),
    }
    return pd.DataFrame(data)


def test_add_technical_indicators(dummy_ohlcv_df):
    """Tests that all technical indicators are calculated and appended."""
    engineer = FeatureEngineer()
    df_result = engineer.add_technical_indicators(dummy_ohlcv_df)

    # Check key columns
    expected_cols = ["rsi", "macd", "macd_signal", "bb_high", "bb_low", "atr", "obv"]
    for col in expected_cols:
        assert col in df_result.columns, f"Missing indicator column: {col}"

    # Verify column length matches input length
    assert len(df_result) == len(dummy_ohlcv_df)


def test_add_date_features(dummy_ohlcv_df):
    """Tests cyclical date feature generation."""
    engineer = FeatureEngineer()
    df_result = engineer.add_date_features(dummy_ohlcv_df)

    assert "day_of_week_sin" in df_result.columns
    assert "day_of_week_cos" in df_result.columns
    assert "month" in df_result.columns