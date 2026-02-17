import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

REGISTRY_FILE = Path("src/registry/production.json")

_cached_signature = None
_cached_model = None
_cached_cfg = None


def _signature(cfg: Dict[str, Any]) -> str:
    # model_path + loader + mlflow.model_uri gibi kritik alanları imzala
    return json.dumps(
        {
            "loader": cfg.get("loader"),
            "model_path": cfg.get("model_path"),
            "mlflow": cfg.get("mlflow", {}),
        },
        sort_keys=True,
    )


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry bulunamadı: {REGISTRY_FILE}")
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def load_model_from_registry(force: bool = False) -> Tuple[Any, Dict[str, Any]]:
    global _cached_signature, _cached_model, _cached_cfg

    cfg = load_registry()
    sig = _signature(cfg)

    if (not force) and _cached_signature == sig and _cached_model is not None:
        return _cached_model, _cached_cfg

    loader = (cfg.get("loader", "local") or "local").lower()

    if loader == "local":
        model_path = Path(cfg["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası yok: {model_path}")
        model = joblib.load(model_path)

    elif loader == "mlflow":
        import mlflow

        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri")
        model_uri = mlflow_cfg.get("model_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if not model_uri:
            raise ValueError("MLflow loader seçili ama model_uri boş.")
        model = mlflow.pyfunc.load_model(model_uri)

    else:
        raise ValueError(f"Bilinmeyen loader: {loader}")

    _cached_signature = sig
    _cached_model = model
    _cached_cfg = cfg
    return model, cfg


def build_feature_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)
