from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from functools import lru_cache
import os
from src.api.model_loader import load_model_from_registry
from src.api.feature_builder import build_features_from_datetime, build_feature_vector

app = FastAPI(title="Energy Forecast API")

@lru_cache(maxsize=10)
def get_model(state: str):
    return load_model_from_registry(state=state)

class PredictRequest(BaseModel):
    state: str
    datetime: datetime

@app.get("/health")
def health():
    return {"status": "ok", "processed_files": os.listdir("data/processed") if os.path.exists("data/processed") else []}

@app.post("/predict-from-datetime")
def predict(req: PredictRequest):
    try:
        model, state_cfg = get_model(req.state.upper())
        features = build_features_from_datetime(req.state.upper(), req.datetime)
        x = build_feature_vector(features, state_cfg["feature_names"])
        y = model.predict(x)
        return {"prediction": float(y[0]), "state": req.state, "datetime": req.datetime}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))