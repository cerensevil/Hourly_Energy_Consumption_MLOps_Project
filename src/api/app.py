from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime as py_datetime

import polars as pl

from src.api.model_loader import load_model_from_registry, load_registry
from src.api.feature_builder import (
    _get_state_df,
    build_feature_vector,
    build_features_from_datetime,
    build_features_future_recursive,
)
import os
print("### LOADED API FILE:", os.path.abspath(__file__))

app = FastAPI(title="Energy Forecast API", version="0.5.2")


# =========================
# Schemas
# =========================
class PredictRequest(BaseModel):
    state: str
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class PredictFromDatetimeRequest(BaseModel):
    state: str
    datetime: py_datetime


# =========================
# Helpers
# =========================
def _to_py_datetime(x) -> py_datetime:
    """
    Polars scalar datetime / Python datetime / string -> Python datetime
    """
    if isinstance(x, py_datetime):
        return x
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime()
    if hasattr(x, "to_python"):
        return x.to_python()
    # fallback
    return py_datetime.fromisoformat(str(x).replace("Z", ""))


def _available_states(full_cfg: Dict[str, Any]):

    block = full_cfg.get("states") or full_cfg.get("state_models") or {}
    if not isinstance(block, dict):
        return []
    return list(block.keys())


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/available-range")
def available_range_all_states():
    """
    UI için genel aralık: tüm state'lerdeki min/max.
    (İsterseniz sadece /available-range/{state} kullanın.)
    """
    full_cfg = load_registry()
    states = _available_states(full_cfg)
    if not states:
        raise HTTPException(status_code=500, detail="Registry'de state bulunamadı.")

    mins = []
    maxs = []

    for s in states:
        try:
            df = _get_state_df(s).sort("Datetime")
            mn = _to_py_datetime(df.select(pl.col("Datetime").min()).item())
            mx = _to_py_datetime(df.select(pl.col("Datetime").max()).item())
            mins.append(mn)
            maxs.append(mx)
        except Exception:
            # bazı state dosyası yoksa vs. atla
            continue

    if not mins or not maxs:
        raise HTTPException(status_code=500, detail="Hiçbir state için range hesaplanamadı.")

    return {
        "min_datetime": min(mins).isoformat(),
        "max_datetime": max(maxs).isoformat(),
        "states_count": len(states),
    }


@app.get("/available-range/{state}")
def available_range(state: str):
    """
    State bazlı min/max datetime (UI için en doğrusu).
    """
    state = state.strip().upper()
    try:
        df = _get_state_df(state).sort("Datetime")
        mn = _to_py_datetime(df.select(pl.col("Datetime").min()).item())
        mx = _to_py_datetime(df.select(pl.col("Datetime").max()).item())
        return {"state": state, "min_datetime": mn.isoformat(), "max_datetime": mx.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Range alınamadı: {e}")


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
    if not feature_names:
        raise HTTPException(status_code=500, detail="Registry'de feature_names boş. (production.json düzeltin)")

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

    feature_names = state_cfg.get("feature_names", [])
    if not feature_names:
        raise HTTPException(status_code=500, detail="Registry'de feature_names boş. (production.json düzeltin)")

    # dataset max datetime
    try:
        df = _get_state_df(state).sort("Datetime")
        max_dt_raw = df.select(pl.col("Datetime").max()).item()
        max_dt = _to_py_datetime(max_dt_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State verisi okunamadı: {e}")

    # features: geçmişte lookup, gelecekte recursive forecast
    try:
        if req.datetime <= max_dt:
            features = build_features_from_datetime(state, req.datetime)  # lookup
        else:
            features = build_features_future_recursive(
                state=state,
                dt=req.datetime,
                model=model,
                feature_names=feature_names,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    missing = [f for f in feature_names if f not in features]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature üretimi eksik: {missing}")

    x = build_feature_vector(features, feature_names)

    try:
        y = model.predict(x)
        pred = float(y[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict hatası: {str(e)}")

    return {"state": state, "datetime": req.datetime.isoformat(), "prediction": pred}
