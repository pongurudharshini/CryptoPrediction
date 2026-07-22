"""
Pydantic Schemas for API Input/Output Validation.
Ensures type safety between the frontend dashboard and the backend AI models.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    """Schema for requesting a price prediction."""
    symbol: str = Field(..., description="Cryptocurrency ticker symbol (e.g., BTC-USD)", examples=["BTC-USD"])
    horizon_days: int = Field(default=7, ge=1, le=30, description="Number of days to predict into the future")

class PredictionResponse(BaseModel):
    """Schema for the prediction output."""
    symbol: str
    current_price: float
    predictions: List[float]
    lower_bound_10: List[float] # 10th percentile (Pessimistic)
    upper_bound_90: List[float] # 90th percentile (Optimistic)
    feature_importance: Optional[Dict[str, float]] = None
    insights: Optional[str] = Field(None, description="LLM generated plain-English explanation")

class TrainingRequest(BaseModel):
    """Schema to trigger a retraining pipeline."""
    symbol: str = Field(default="BTC-USD")
    use_pso: bool = Field(default=True, description="Whether to run PSO hyperparameter tuning before training")
    epochs: int = Field(default=10, ge=1, le=100)

class HealthResponse(BaseModel):
    """Schema for API health check."""
    status: str
    model_loaded: bool
    version: str