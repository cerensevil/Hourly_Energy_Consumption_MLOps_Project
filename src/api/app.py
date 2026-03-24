from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import time
import logging

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.model_loader import load_model_from_registry
from src.api.feature_builder import (
    build_feature_vector,
    build_features_from_datetime,
)

from src.monitoring.metrics import (
    prediction_count,
    prediction_latency,
    prediction_errors,
    prediction_value,
    absolute_error,
    squared_error,
    baseline_squared_error,  # 🔥 NEW
    underprediction_count,
    overprediction_count,
    cost_weighted_error_metric
)

from src.metrics.cost_weighted_error import cost_weighted_error


# ================================
# Logging
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ================================
# App Init
# ================================
app = FastAPI(title="Energy Forecast API", version="1.0.0")


# ================================
# Schemas
# ================================
class PredictRequest(BaseModel):
    state: str
    features: Dict[str, float]
    actual: Optional[float] = None


class PredictFromDatetimeRequest(BaseModel):
    state: str
    datetime: datetime
    actual: Optional[float] = None


# ================================
# Root
# ================================
@app.get("/")
def root():
    return {"message": "Energy Forecast API is running"}


# ================================
# Health
# ================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ================================
# Metrics
# ================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ================================
# CORE METRIC LOGIC (REUSABLE)
# ================================
def log_metrics(state: str, pred: float, latency: float, actual: Optional[float], features: Dict[str, float]):

    # ================================
    # BASIC METRICS
    # ================================
    prediction_count.labels(state=state).inc()
    prediction_latency.labels(state=state).observe(latency)
    prediction_value.labels(state=state).observe(pred)

    # ================================
    # ERROR + BUSINESS METRICS
    # ================================
    if actual is None:
        return

    error = actual - pred
    abs_err = abs(error)

    absolute_error.labels(state=state).observe(abs_err)
    squared_error.labels(state=state).observe(error ** 2)

    # ================================
    # BASELINE ERROR (🔥 CRITICAL)
    # ================================
    if "target_lag_24" in features:
        y_baseline = features["target_lag_24"]
        baseline_err = (actual - y_baseline) ** 2
        baseline_squared_error.labels(state=state).observe(baseline_err)

    # ================================
    # UNDER / OVER
    # ================================
    if error > 0:
        underprediction_count.labels(state=state).inc()
    else:
        overprediction_count.labels(state=state).inc()

    # ================================
    # BUSINESS METRIC (CWE)
    # ================================
    cwe = cost_weighted_error([actual], [pred])
    cost_weighted_error_metric.labels(state=state).observe(cwe)


# ================================
# Prediction Endpoint
# ================================
@app.post("/predict")
def predict(req: PredictRequest):

    start = time.time()
    state = req.state.strip().upper()

    try:
        model, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=str(e))

    feature_names = state_cfg.get("feature_names", [])
    missing = [f for f in feature_names if f not in req.features]

    if missing:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    try:
        x = build_feature_vector(req.features, feature_names)
        y = model.predict(x)
        pred = float(y[0])

        logger.info(f"[PRED DEBUG] state={state} prediction={pred}")

    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    # 🔥 LOG EVERYTHING
    log_metrics(state, pred, latency, req.actual, req.features)

    return {
        "state": state,
        "prediction": pred,
        "latency": latency
    }


# ================================
# Datetime Prediction Endpoint
# ================================
@app.post("/predict-from-datetime")
def predict_from_datetime(req: PredictFromDatetimeRequest):

    start = time.time()
    state = req.state.strip().upper()

    try:
        model, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=str(e))

    try:
        features = build_features_from_datetime(state, req.datetime)
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=str(e))

    try:
        x = build_feature_vector(features, state_cfg.get("feature_names", []))
        y = model.predict(x)
        pred = float(y[0])

        logger.info(f"[PRED DEBUG] state={state} prediction={pred}")

    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    # 🔥 LOG EVERYTHING
    log_metrics(state, pred, latency, req.actual, features)

    return {
        "state": state,
        "datetime": req.datetime.isoformat(),
        "prediction": pred,
        "latency": latency
    }