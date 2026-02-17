import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
REGISTRY_FILE = BASE_DIR / "registry" / "production.json"


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry bulunamadı: {REGISTRY_FILE}")
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def load_model_from_registry() -> Tuple[Any, Dict[str, Any]]:
    cfg = load_registry()
    loader = cfg.get("loader", "local").lower()

    if loader == "local":
        model_path = BASE_DIR / cfg["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası yok: {model_path}")

        model = joblib.load(model_path)
        return model, cfg

    if loader == "mlflow":
        import mlflow

        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri")
        model_uri = mlflow_cfg.get("model_uri")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if not model_uri:
            raise ValueError("MLflow loader seçili ama model_uri boş.")

        model = mlflow.pyfunc.load_model(model_uri)
        return model, cfg

    raise ValueError(f"Bilinmeyen loader: {loader}")
