"""
FastAPI Application

Main API for Tadawul news-based stock movement prediction.
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
    TICKERS,
)
from .predictor import Predictor


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Predictor | None = None


class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""

    ticker: str
    texts: List[str]


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""

    ticker: str
    labels: List[str]
    predictions: List[int]
    probabilities_up: List[float]


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on application startup."""
    global predictor
    predictor = Predictor()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/tickers")
async def list_tickers():
    """Return available Tadawul tickers."""
    return TICKERS


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict Up/Down movement from news texts for a given ticker."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Invalid ticker")
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    result = predictor.predict_from_texts(req.texts)
    return PredictResponse(
        ticker=req.ticker,
        labels=result["labels"],
        predictions=result["predictions"],
        probabilities_up=result["probabilities_up"],
    )
