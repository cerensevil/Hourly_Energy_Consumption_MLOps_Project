from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from src.api.model_loader import (
    load_model_from_registry,
    load_registry
)
from src.api.feature_builder import (
    build_feature_vector,
    build_features_from_datetime,
)

app = FastAPI(title="Energy Forecast API", version="0.5.1")


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


@app.post("/predict")
def predict(req: PredictRequest):

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

    return {"state": state, "prediction": pred}


@app.post("/predict-from-datetime")
def predict_from_datetime(req: PredictFromDatetimeRequest):

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

    return {"state": state, "datetime": req.datetime.isoformat(), "prediction": pred}
