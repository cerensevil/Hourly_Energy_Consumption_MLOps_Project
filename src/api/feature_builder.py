from typing import Dict, List
import numpy as np


def build_feature_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)
