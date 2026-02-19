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
    """
    production.json hem eski şemayı hem yeni şemayı desteklesin:
    - states
    - state_models
    """
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

    # 🔥 Cache kontrol
    if (not force) and (state in _cached_models):
        return _cached_models[state], state_cfg

    # loader önceliği: state_cfg > cfg > local
    loader = (state_cfg.get("loader") or cfg.get("loader") or "local").lower()

    if loader == "local":
        model_path = Path(state_cfg["model_path"])
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
    
    # --- EKLE (model ile feature_names uyum kontrolü) ---
    fns = state_cfg.get("feature_names") or []
    n_expected = getattr(model, "n_features_in_", None)  # sklearn için
    if n_expected is not None and fns and len(fns) != int(n_expected):
        raise ValueError(
            f"Feature mismatch: registry feature_names={len(fns)} ama model n_features_in_={n_expected}. "
            f"production.json -> {state} -> feature_names düzelt."
        )

    _cached_models[state] = model
    return model, state_cfg
    
def build_feature_vector(
    features: Dict[str, float],
    feature_names: List[str]
) -> np.ndarray:
    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)
