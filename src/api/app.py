from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import time

from prometheus_client import start_http_server

from src.api.model_loader import (
    load_model_from_registry,
    load_registry
)

from src.api.feature_builder import (
    build_feature_vector,
    build_features_from_datetime,
)

from src.monitoring.metrics import (
    prediction_count,
    prediction_latency
)

app = FastAPI(title="Energy Forecast API", version="0.6.0")


# ---------------------------------------------------------
# Start Prometheus metrics server
# ---------------------------------------------------------

start_http_server(8001)


class PredictRequest(BaseModel):
    state: str
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class PredictFromDatetimeRequest(BaseModel):
    state: str
    datetime: datetime


@app.get("/health")
def health():
    return {"status": "ok"}


def _available_states(full_cfg: Dict[str, Any]):

    # production.json: states veya state_models destekle
    block = full_cfg.get("states") or full_cfg.get("state_models") or {}

    if not isinstance(block, dict):
        return []

    return list(block.keys())


@app.get("/model-info/{state}")
def model_info(state: str):

    state = state.strip().upper()

    try:
        _, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    full_cfg = load_registry()

    return {
        "state": state,
        "loader": state_cfg.get("loader") or full_cfg.get("loader"),
        "feature_names": state_cfg.get("feature_names"),
        "available_states": _available_states(full_cfg),
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(req: PredictRequest):

    start = time.time()

    state = req.state.strip().upper()

    try:
        model, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_names = state_cfg.get("feature_names", [])

    missing = [f for f in feature_names if f not in req.features]

    if missing:
        raise HTTPException(status_code=400, detail=f"Eksik feature'lar: {missing}")

    x = build_feature_vector(req.features, feature_names)

    try:
        y = model.predict(x)
        pred = float(y[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict hatası: {str(e)}")

    # -----------------------------------------------------
    # Prometheus metrics
    # -----------------------------------------------------

    prediction_count.inc()
    prediction_latency.observe(time.time() - start)

    return {
        "state": state,
        "prediction": pred
    }


# ---------------------------------------------------------
# Datetime based prediction endpoint
# ---------------------------------------------------------

@app.post("/predict-from-datetime")
def predict_from_datetime(req: PredictFromDatetimeRequest):

    start = time.time()

    state = req.state.strip().upper()

    try:
        model, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        features = build_features_from_datetime(state, req.datetime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_names = state_cfg.get("feature_names", [])

    x = build_feature_vector(features, feature_names)

    try:
        y = model.predict(x)
        pred = float(y[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict hatası: {str(e)}")

    # -----------------------------------------------------
    # Prometheus metrics
    # -----------------------------------------------------

    prediction_count.inc()
    prediction_latency.observe(time.time() - start)

    return {
        "state": state,
        "datetime": req.datetime.isoformat(),
        "prediction": pred
    }