import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import joblib


REGISTRY_FILE = Path("src/registry/production.json")


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry bulunamadı: {REGISTRY_FILE}")
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def load_model_from_registry() -> Tuple[Any, Dict[str, Any]]:
    cfg = load_registry()
    loader = cfg.get("loader", "local").lower()

    if loader == "local":
        model_path = Path(cfg["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası yok: {model_path}")
        model = joblib.load(model_path)
        return model, cfg

    if loader == "mlflow":
        # MLflow hazır olacak ama şimdilik kullanmayacağız.
        # Yarın MLflow kurunca burayı aktive edeceğiz.
        import mlflow

        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri")
        model_uri = mlflow_cfg.get("model_uri")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if not model_uri:
            raise ValueError("MLflow loader seçili ama model_uri boş. production.json içinde doldur.")

        model = mlflow.pyfunc.load_model(model_uri)
        return model, cfg

    raise ValueError(f"Bilinmeyen loader: {loader}")


def build_feature_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)
