from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from src.api.model_loader import load_model_from_registry
from src.api.feature_builder import (
    build_feature_vector,
    build_features_from_datetime,
)

app = FastAPI(title="Energy Forecast API", version="0.3.0")


# ==========================================================
# REQUEST MODELLERİ
# ==========================================================

class PredictRequest(BaseModel):
    state: str
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class PredictFromDatetimeRequest(BaseModel):
    state: str
    datetime: datetime


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health():
    try:
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


# ==========================================================
# MODEL INFO (STATE-BASED)
# ==========================================================

@app.get("/model-info/{state}")
def model_info(state: str):
    try:
        model, cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "state": state,
        "loader": cfg.get("loader"),
        "feature_names": cfg.get("feature_names"),
        "available_states": list(cfg.get("state_models", {}).keys()),
    }


# ==========================================================
# MANUAL FEATURE PREDICT
# ==========================================================

@app.post("/predict")
def predict(req: PredictRequest):

    try:
        model, cfg = load_model_from_registry(state=req.state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model is None or cfg is None:
        raise HTTPException(status_code=500, detail="Model yüklenmedi.")

    feature_names = cfg.get("feature_names", [])

    missing = [f for f in feature_names if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Eksik feature'lar: {missing}"
        )

    x = build_feature_vector(req.features, feature_names)

    try:
        y = model.predict(x)
        pred = float(y[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Predict hatası: {str(e)}"
        )

    return {
        "state": req.state,
        "prediction": pred
    }


# ==========================================================
# DATETIME-BASED SMART PREDICT
# ==========================================================

@app.post("/predict-from-datetime")
def predict_from_datetime(req: PredictFromDatetimeRequest):

    try:
        model, cfg = load_model_from_registry(state=req.state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        features = build_features_from_datetime(req.state, req.datetime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_names = cfg.get("feature_names", [])
    x = build_feature_vector(features, feature_names)

    try:
        y = model.predict(x)
        pred = float(y[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Predict hatası: {str(e)}"
        )

    return {
        "state": req.state,
        "datetime": req.datetime,
        "prediction": pred
    }
