from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from src.api.model_loader import load_model_from_registry
from src.api.feature_builder import build_feature_vector

app = FastAPI(title="Energy Forecast API", version="1.0.0")

MODEL = None
CFG = None


class PredictRequest(BaseModel):
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def startup():
    global MODEL, CFG
    MODEL, CFG = load_model_from_registry()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None
    }


@app.get("/model-info")
def model_info():
    if CFG is None:
        raise HTTPException(status_code=500, detail="Config yüklenmedi.")

    return {
        "model_type": CFG.get("model_type"),
        "loader": CFG.get("loader"),
        "model_path": CFG.get("model_path"),
        "feature_names": CFG.get("feature_names"),
        "mlflow": CFG.get("mlflow")
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None or CFG is None:
        raise HTTPException(status_code=500, detail="Model yüklenmedi.")

    feature_names = CFG.get("feature_names", [])

    missing = [f for f in feature_names if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Eksik feature'lar: {missing}")

    x = build_feature_vector(req.features, feature_names)

    try:
        y = MODEL.predict(x)
        pred = float(y[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict hatası: {str(e)}")

    return {"prediction": pred}
