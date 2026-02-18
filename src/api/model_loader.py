import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np

REGISTRY_FILE = Path("src/registry/production.json")

# 🔥 State bazlı cache
_cached_models: Dict[str, Dict[str, Any]] = {}


# ==========================================================
# SIGNATURE
# ==========================================================
def _signature(cfg: Dict[str, Any], state: str) -> str:
    state_cfg = cfg.get("state_models", {}).get(state)

    if not state_cfg:
        raise ValueError(f"State '{state}' production.json içinde bulunamadı.")

    return json.dumps(
        {
            "loader": cfg.get("loader"),
            "model_path": state_cfg.get("model_path"),
            "mlflow": cfg.get("mlflow", {}),
        },
        sort_keys=True,
    )


# ==========================================================
# REGISTRY LOAD
# ==========================================================
def load_registry() -> Dict[str, Any]:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry bulunamadı: {REGISTRY_FILE}")

    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


# ==========================================================
# STATE AWARE MODEL LOAD
# ==========================================================
def load_model_from_registry(
    state: Optional[str],
    force: bool = False
) -> Tuple[Any, Dict[str, Any]]:

    if state is None:
        raise ValueError("State belirtilmelidir.")

    cfg = load_registry()

    state_cfg = cfg.get("state_models", {}).get(state)

    if not state_cfg:
        raise ValueError(f"State '{state}' için model bulunamadı.")

    sig = _signature(cfg, state)

    # 🔥 Cache kontrol
    if (
        not force
        and state in _cached_models
        and _cached_models[state]["signature"] == sig
    ):
        return _cached_models[state]["model"], cfg

    loader = (cfg.get("loader", "local") or "local").lower()

    # ======================================================
    # LOCAL LOADER
    # ======================================================
    if loader == "local":

        model_path = Path(state_cfg["model_path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası yok: {model_path}")

        model = joblib.load(model_path)

    # ======================================================
    # MLFLOW LOADER
    # ======================================================
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

    # 🔥 Cache güncelle
    _cached_models[state] = {
        "signature": sig,
        "model": model,
    }

    return model, cfg


# ==========================================================
# FEATURE VECTOR BUILDER
# ==========================================================
def build_feature_vector(
    features: Dict[str, float],
    feature_names: List[str]
) -> np.ndarray:

    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)
