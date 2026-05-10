"""
FastAPI ML inference server.
Install: pip install fastapi uvicorn scikit-learn numpy pydantic
Run:     uvicorn fastapi_app:app --reload
Docs:    http://localhost:8000/docs
"""
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────
MODEL = None
SCALER = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, SCALER
    logger.info("Loading model...")
    try:
        MODEL = joblib.load("models/model.pkl")
        SCALER = joblib.load("models/scaler.pkl")
        logger.info(f"Model loaded: {type(MODEL).__name__}")
    except FileNotFoundError:
        # For demo: create a simple fallback model
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        ds = load_breast_cancer()
        X_train, _, y_train, _ = train_test_split(ds.data, ds.target, random_state=42)
        SCALER = StandardScaler().fit(X_train)
        MODEL = RandomForestClassifier(n_estimators=50, random_state=42)
        MODEL.fit(SCALER.transform(X_train), y_train)
        logger.info("Demo model trained (no saved model found)")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ML Inference API",
    description="REST API for ML model inference with health checks and metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Schemas ──────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector (30 values for breast cancer)")
    return_probabilities: bool = Field(False, description="Return class probabilities")

    @validator("features")
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        if any(np.isnan(f) or np.isinf(f) for f in v):
            raise ValueError("Features contain NaN or Inf values")
        return v


class BatchPredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="List of feature vectors")
    return_probabilities: bool = False

    @validator("instances")
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError("Empty batch")
        if len(v) > 1000:
            raise ValueError("Batch size exceeds limit of 1000")
        return v


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: Optional[List[float]] = None
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: List[int]
    confidences: List[float]
    count: int
    latency_ms: float


# ── Middleware for request logging ────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({latency:.1f}ms)")
    return response


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/info", tags=["ops"])
async def model_info():
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_type": type(MODEL).__name__,
        "n_features": MODEL.n_features_in_,
        "classes": MODEL.classes_.tolist(),
        "n_estimators": getattr(MODEL, "n_estimators", None),
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    X = np.array(request.features, dtype=np.float32).reshape(1, -1)

    if X.shape[1] != MODEL.n_features_in_:
        raise HTTPException(
            422, f"Expected {MODEL.n_features_in_} features, got {X.shape[1]}"
        )

    X_scaled = SCALER.transform(X)
    pred = int(MODEL.predict(X_scaled)[0])
    proba = MODEL.predict_proba(X_scaled)[0].tolist()
    confidence = float(max(proba))
    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        prediction=pred,
        confidence=confidence,
        probabilities=proba if request.return_probabilities else None,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(request: BatchPredictRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    X = np.array(request.instances, dtype=np.float32)

    if X.shape[1] != MODEL.n_features_in_:
        raise HTTPException(422, f"Expected {MODEL.n_features_in_} features, got {X.shape[1]}")

    X_scaled = SCALER.transform(X)
    preds = MODEL.predict(X_scaled).tolist()
    probas = MODEL.predict_proba(X_scaled)
    confidences = probas.max(axis=1).tolist()
    latency = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        predictions=preds,
        confidences=[round(c, 4) for c in confidences],
        count=len(preds),
        latency_ms=round(latency, 2),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
