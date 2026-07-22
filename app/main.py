"""
FastAPI Main Application Entry Point.
Configures CORS, routes, application metadata, and redirects.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.api.endpoints import predict
from app.schemas import HealthResponse

# Create FastAPI instance with metadata for Swagger UI
app = FastAPI(
    title="PSO-TFT Crypto Predictor API",
    description="Backend API for AI-Driven Cryptocurrency Price Prediction System",
    version="1.0.0",
    docs_url="/swagger" # Custom swagger URL as requested
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the Streamlit frontend to make requests to this backend securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predictions"])

@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root URL to the Swagger documentation."""
    return RedirectResponse(url="/swagger")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="active",
        model_loaded=True,
        version="1.0.0"
    )

if __name__ == "__main__":
    import uvicorn
    # This block allows running the file directly via `python app/main.py`
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)