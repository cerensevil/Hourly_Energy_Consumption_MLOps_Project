from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Optional
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
# App
# ================================
app = FastAPI(title="Energy Forecast API", version="2.3.0")

LAST_RETRAIN_TIME = None

# ================================
# 🔥 FIX: State listesi
# inference_flow.py /available-states endpoint'ini çağırıyor.
# Bu endpoint olmadığı için flow sadece ["DAYTON"] ile çalışıyordu.
# ================================
STATES = ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "PJME", "PJMW"]


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
# Basic
# ================================
@app.get("/")
def root():
    return {"message": "API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ================================
# 🔥 FIX: /available-states endpoint EKLENDİ
# ================================
@app.get("/available-states")
def available_states():
    return {"states": STATES}


# ================================
# 🔥 FINAL METRIC LOGGER
# ================================
def log_metrics_safe(state, pred, latency, actual, features, model_name, model_version):
    try:
        # ALWAYS
        active_model.labels(state=state, model_type=model_name, model_version=model_version).set(1)

        prediction_count.labels(state=state, model_type=model_name, model_version=model_version).inc()
        prediction_latency.labels(state=state, model_type=model_name, model_version=model_version).observe(latency)
        prediction_value.labels(state=state, model_type=model_name, model_version=model_version).observe(pred)

        # ONLY IF ACTUAL EXISTS
        if actual is not None:

            error = actual - pred

            absolute_error.labels(state=state, model_type=model_name, model_version=model_version).observe(abs(error))
            squared_error.labels(state=state, model_type=model_name, model_version=model_version).observe(error ** 2)

            if "target_lag_24" in features:
                baseline_squared_error.labels(state=state).observe(
                    (actual - features["target_lag_24"]) ** 2
                )

            if error > 0:
                underprediction_count.labels(state=state, model_type=model_name, model_version=model_version).inc()
            else:
                overprediction_count.labels(state=state, model_type=model_name, model_version=model_version).inc()

            cwe = cost_weighted_error([actual], [pred])

            cost_weighted_error_metric.labels(state=state, model_type=model_name, model_version=model_version).observe(cwe)

    except Exception as e:
        logger.warning(f"⚠️ Metric logging failed: {e}")


# ================================
# PREDICT (FEATURE)
# ================================
@app.post("/predict")
def predict(req: PredictRequest):

    start = time.time()
    state = req.state.strip().upper()

    try:
        model, cfg = load_model_from_registry(state)
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=str(e))

    feature_names = cfg.get("feature_names") or []
    if not feature_names:
        raise HTTPException(status_code=500, detail="Feature names missing")

    try:
        x = build_feature_vector(req.features, feature_names)
        pred = float(model.predict(x)[0])
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    log_metrics_safe(
        state,
        pred,
        latency,
        req.actual,
        req.features,
        cfg.get("model_type", "unknown"),
        cfg.get("version", "v1")
    )

    return {"prediction": pred, "latency": latency}


# ================================
# 🔥 MAIN ENDPOINT (AUTO ACTUAL)
# ================================
@app.post("/predict-from-datetime")
def predict_from_datetime(req: PredictFromDatetimeRequest):

    start = time.time()
    state = req.state.strip().upper()

    try:
        model, cfg = load_model_from_registry(state)
    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail=str(e))

    feature_names = cfg.get("feature_names") or []
    if not feature_names:
        raise HTTPException(status_code=500, detail="Feature names missing")

    try:
        # 🔥 DATA
        features = build_features_from_datetime(state, req.datetime)

        # 🔥 AUTO ACTUAL
        actual = features.get("target")

        if actual is None:
            actual = features.get(f"{state}_MW")

        x = build_feature_vector(features, feature_names)
        pred = float(model.predict(x)[0])

    except Exception as e:
        prediction_errors.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    log_metrics_safe(
        state,
        pred,
        latency,
        actual,
        features,
        cfg.get("model_type", "unknown"),
        cfg.get("version", "v1")
    )

    return {
        "state": state,
        "prediction": pred,
        "actual": actual,
        "latency": latency
    }


# ================================
# RETRAIN
# ================================
@app.post("/approve_retrain")
def approve_retrain():
    global LAST_RETRAIN_TIME

    evaluate_and_promote()
    LAST_RETRAIN_TIME = datetime.utcnow()

    return {"status": "ok", "time": LAST_RETRAIN_TIME.isoformat()}