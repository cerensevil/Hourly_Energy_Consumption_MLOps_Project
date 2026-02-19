from typing import Dict, List, Union
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime

# 🔥 RAM CACHE
_STATE_CACHE: Dict[str, pl.DataFrame] = {}

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------
# 1️⃣ Feature Vector Builder
# ---------------------------------------------------
def build_feature_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    x = [features[name] for name in feature_names]
    return np.array([x], dtype=float)


# ---------------------------------------------------
# 2️⃣ Internal: State Data Loader (RAM Cache)
# ---------------------------------------------------
def _get_state_df(state: str, force_reload: bool = False) -> pl.DataFrame:
    state = state.strip().upper()

    if (not force_reload) and state in _STATE_CACHE:
        return _STATE_CACHE[state]

    parquet_path = PROCESSED_DIR / f"{state}_hourly_processed.parquet"

    if not parquet_path.exists():
        raise ValueError(f"{state} için processed veri bulunamadı: {parquet_path}")

    df = pl.read_parquet(parquet_path)

    _STATE_CACHE[state] = df
    return df


# ---------------------------------------------------
# 3️⃣ Datetime Normalize
# ---------------------------------------------------
def _ensure_datetime(dt: Union[str, datetime]) -> datetime:
    if isinstance(dt, datetime):
        return dt

    try:
        return datetime.fromisoformat(str(dt).replace("Z", ""))
    except Exception:
        raise ValueError(f"Datetime parse edilemedi: {dt}")


# ---------------------------------------------------
# 4️⃣ Datetime'den Feature Üretici (CACHE'Lİ)
# ---------------------------------------------------
def build_features_from_datetime(state: str, dt: Union[str, datetime]) -> Dict[str, float]:

    dt = _ensure_datetime(dt)
    state = state.strip().upper()
    df = _get_state_df(state)

    dt_pl = pl.datetime(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second
    )

    row = df.filter(pl.col("Datetime") == dt_pl)

    if row.height == 0:
        raise ValueError(
            f"{state} için bu datetime bulunamadı: {dt}"
        )

    feature_dict = {}

    # 🔥 MODELİN BEKLEDİĞİ ORİJİNAL TARGET ADI EKLENİYOR
    target_col_name = f"{state}_MW"
    target_value = row.select("target").item()
    feature_dict[target_col_name] = float(target_value)

    # Diğer feature'lar
    feature_cols = [
        "hour", "dayofweek", "month", "year", "is_weekend",
        "sin_hour", "cos_hour",
        "target_lag_1", "target_lag_24",
        "target_roll_mean_24", "target_roll_std_24"
    ]

    for col in feature_cols:
        value = row.select(col).item()
        feature_dict[col] = float(value)

    return feature_dict


# ---------------------------------------------------
# 5️⃣ Cache Info
# ---------------------------------------------------
def cache_info() -> Dict[str, List[str]]:
    return {
        "cached_states": list(_STATE_CACHE.keys()),
        "count": len(_STATE_CACHE)
    }


def clear_cache(state: str = None):
    if state is None:
        _STATE_CACHE.clear()
    else:
        _STATE_CACHE.pop(state.strip().upper(), None)
