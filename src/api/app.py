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
    baseline_squared_error,
    underprediction_count,
    overprediction_count,
    cost_weighted_error_metric,
    active_model,
)

from src.metrics.cost_weighted_error import cost_weighted_error
from src.training.evaluate_and_promote_flow import evaluate_and_promote

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
app = FastAPI(title="Energy Forecast API", version="2.0.0")

# ================================
# STATE (NEW)
# ================================
LAST_RETRAIN_TIME = None
LAST_SIGNAL_TIME = None

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
# CORE METRIC LOGIC
# ================================
def log_metrics(
    state: str,
    pred: float,
    latency: float,
    actual: Optional[float],
    features: Dict[str, float],
    model_name: str,
    model_version: str
):

    active_model.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).set(1)

    prediction_count.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).inc()

    prediction_latency.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).observe(latency)

    prediction_value.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).observe(pred)

    if actual is None:
        return

    error = actual - pred
    abs_err = abs(error)

    absolute_error.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).observe(abs_err)

    squared_error.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).observe(error ** 2)

    if "target_lag_24" in features:
        y_baseline = features["target_lag_24"]
        baseline_err = (actual - y_baseline) ** 2
        baseline_squared_error.labels(state=state).observe(baseline_err)

    if error > 0:
        underprediction_count.labels(
            state=state,
            model_type=model_name,
            model_version=model_version
        ).inc()
    else:
        overprediction_count.labels(
            state=state,
            model_type=model_name,
            model_version=model_version
        ).inc()

    cwe = cost_weighted_error([actual], [pred])
    cost_weighted_error_metric.labels(
        state=state,
        model_type=model_name,
        model_version=model_version
    ).observe(cwe)


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

    model_name = state_cfg.get("model_type", "unknown")
    model_version = state_cfg.get("version", "v0")

    feature_names = state_cfg.get("feature_names", [])
    missing = [f for f in feature_names if f not in req.features]

    if missing:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    try:
        x = build_feature_vector(req.features, feature_names)
        y = model.predict(x)
        pred = float(y[0])

        logger.info(f"[PRED] state={state} pred={pred}")

    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    log_metrics(
        state,
        pred,
        latency,
        req.actual,
        req.features,
        model_name,
        model_version
    )

    return {
        "state": state,
        "model": model_name,
        "version": model_version,
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

    model_name = state_cfg.get("model_type", "unknown")
    model_version = state_cfg.get("version", "v0")

    try:
        features = build_features_from_datetime(state, req.datetime)
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=str(e))

    try:
        x = build_feature_vector(features, state_cfg.get("feature_names", []))
        y = model.predict(x)
        pred = float(y[0])

        logger.info(f"[PRED] state={state} pred={pred}")

    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    log_metrics(
        state,
        pred,
        latency,
        req.actual,
        features,
        model_name,
        model_version
    )

    return {
        "state": state,
        "datetime": req.datetime.isoformat(),
        "model": model_name,
        "version": model_version,
        "prediction": pred,
        "latency": latency
    }


# ================================
# RETRAIN CONTROL (NEW 🚀)
# ================================
@app.post("/approve_retrain")
def approve_retrain():

    global LAST_RETRAIN_TIME

    logger.warning("🚀 Manual retraining triggered")

    try:
        evaluate_and_promote()

        LAST_RETRAIN_TIME = datetime.utcnow()

        return {
            "status": "success",
            "message": "Retraining started",
            "time": LAST_RETRAIN_TIME.isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Retraining failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# RETRAIN PREVIEW
# ================================
@app.get("/retrain-preview")
def retrain_preview():

    return {
        "message": "Retraining is NOT automatic",
        "action_required": "Call /approve_retrain to execute",
        "note": "Triggered based on monitoring signals"
    }


# ================================
# RETRAIN STATUS
# ================================
@app.get("/retrain-status")
def retrain_status():

    return {
        "last_retrain_time": LAST_RETRAIN_TIME.isoformat() if LAST_RETRAIN_TIME else None,
        "last_signal_time": LAST_SIGNAL_TIME.isoformat() if LAST_SIGNAL_TIME else None,
        "mode": "manual"
    }