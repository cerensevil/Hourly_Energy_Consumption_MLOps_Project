import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np

REGISTRY_FILE = Path("src/registry/production.json")

# 🔥 State bazlı RAM cache
_cached_models: Dict[str, Any] = {}


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry bulunamadı: {REGISTRY_FILE}")
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _get_states_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    states_block = cfg.get("states")
    if isinstance(states_block, dict) and states_block:
        return states_block

    states_block = cfg.get("state_models")
    if isinstance(states_block, dict) and states_block:
        return states_block

    raise ValueError("production.json içinde 'states' veya 'state_models' bulunamadı.")


def load_model_from_registry(
    state: Optional[str],
    force: bool = False
) -> Tuple[Any, Dict[str, Any]]:

    if not state:
        raise ValueError("State belirtilmelidir.")

    state = state.strip().upper()

    cfg = load_registry()
    states_block = _get_states_block(cfg)

    if state not in states_block:
        raise ValueError(f"State '{state}' için model bulunamadı.")

    state_cfg = states_block[state]

    # 🔥 CACHE
    if (not force) and (state in _cached_models):
        return _cached_models[state], state_cfg

    loader = (state_cfg.get("loader") or cfg.get("loader") or "local").lower()

    # =========================
    # LOCAL LOADER
    # =========================
    if loader == "local":
        model_path = Path(state_cfg.get("model_path", ""))

        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası yok: {model_path}")

        model = joblib.load(model_path)

    # =========================
    # MLFLOW LOADER (🔥 FIXED)
    # =========================
    elif loader == "mlflow":
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        model_name = state_cfg.get("model_name")

        if not model_name:
            raise ValueError(f"{state} için model_name eksik.")

        model_uri = f"models:/{model_name}/Production"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            raise RuntimeError(f"MLflow model yüklenemedi: {model_uri} | {str(e)}")

        # 🔥 CRITICAL FIX: FEATURE NAMES OKU
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])

        if versions:
            version = versions[0]
            tags = version.tags

            feature_names_str = tags.get("feature_names")

            if feature_names_str:
                try:
                    feature_names = json.loads(feature_names_str)
                except:
                    feature_names = []
            else:
                feature_names = []
        else:
            feature_names = []

        # 🔥 state_cfg içine inject et
        state_cfg["feature_names"] = feature_names

    else:
        raise ValueError(f"Bilinmeyen loader: {loader}")

    _cached_models[state] = model

    return model, state_cfg


def build_feature_vector(
    features: Dict[str, float],
    feature_names: List[str]
) -> np.ndarray:
    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)