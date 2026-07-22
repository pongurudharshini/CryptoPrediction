"""
Prediction API Endpoint.
Handles inference requests, data fetching, and model serving.
"""

from fastapi import APIRouter, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
import logging

# We will mock the actual deep learning inference for the API structure setup.
# In the final production run, this connects to the saved PyTorch .ckpt file.
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """
    Generates a multi-horizon price prediction using the trained PSO-TFT model.
    """
    logger.info(f"Received prediction request for {request.symbol} over {request.horizon_days} days.")
    
    try:
        # TODO: Load actual model checkpoint and run inference.
        # For now, we simulate the output to verify the API wiring.
        current_price = 65000.0 if "BTC" in request.symbol else 3000.0
        
        # Simulate a slight upward trend with noise
        base_trend = [current_price * (1 + (i * 0.005)) for i in range(1, request.horizon_days + 1)]
        noise = np.random.normal(0, current_price * 0.02, request.horizon_days)
        
        predictions = np.array(base_trend) + noise
        lower_bound = predictions * 0.95
        upper_bound = predictions * 1.05

        # Simulated feature importance from VSN
        mock_importance = {
            "macd": 35.2,
            "rsi": 28.1,
            "fng_value": 20.5,
            "volume": 16.2
        }
        
        mock_insight = f"The model is highly confident in an upward trend for {request.symbol}, primarily driven by strong MACD momentum and positive Fear & Greed sentiment."

        return PredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            predictions=predictions.tolist(),
            lower_bound_10=lower_bound.tolist(),
            upper_bound_90=upper_bound.tolist(),
            feature_importance=mock_importance,
            insights=mock_insight
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")