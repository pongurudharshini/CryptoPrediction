"""
Integration Tests for FastAPI Endpoints.
Verifies HTTP API endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Tests the /health endpoint returns 200 OK and active status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["model_loaded"] is True


def test_predict_endpoint_valid_request():
    """Tests /api/v1/predict/ with valid payload."""
    payload = {
        "symbol": "BTC-USD",
        "horizon_days": 7
    }
    response = client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert len(data["predictions"]) == 7
    assert len(data["lower_bound_10"]) == 7
    assert len(data["upper_bound_90"]) == 7
    assert "insights" in data


def test_predict_endpoint_invalid_horizon():
    """Tests input validation when requesting horizon_days outside allowed bounds (1-30)."""
    payload = {
        "symbol": "BTC-USD",
        "horizon_days": 100  # Exceeds max limit of 30
    }
    response = client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)