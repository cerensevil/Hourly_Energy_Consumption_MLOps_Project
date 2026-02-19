from typing import Dict, List, Union
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from collections import deque
import math

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
    df = _get_state_df(state)

    # Polars datetime objesi üret
    dt_pl = pl.datetime(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second
    )

    row = df.filter(pl.col("Datetime") == dt_pl)

    if row.height == 0:
        raise ValueError(
            f"{state} için bu datetime bulunamadı: {dt}"
        )

    # hedef kolon processed dosyada "target"
    feature_cols = [c for c in df.columns if c not in ["Datetime", "state", "target", "AEP_MW"]]

    feature_dict = {}

    for col in feature_cols:
        value = row.select(col).item()
        feature_dict[col] = float(value)

    return feature_dict


# ---------------------------------------------------
# 5️⃣ Cache Info (opsiyonel debug için)
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

# feature_builder.py içine EKLE

def _pick_target_col(df: pl.DataFrame, state: str) -> str:
    cols = set(df.columns)
    if "target" in cols:
        return "target"
    if state in cols:
        return state
    # bazı projelerde "Load" vb olabilir; gerekirse buraya ekleyebiliriz
    raise ValueError(f"Target kolonu bulunamadı. Kolonlar: {df.columns}")

def _time_features(dt: datetime) -> Dict[str, float]:
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    year = dt.year
    is_weekend = 1.0 if dayofweek >= 5 else 0.0
    sin_hour = math.sin(2 * math.pi * hour / 24.0)
    cos_hour = math.cos(2 * math.pi * hour / 24.0)

    return {
        "hour": float(hour),
        "dayofweek": float(dayofweek),
        "month": float(month),
        "year": float(year),
        "is_weekend": float(is_weekend),
        "sin_hour": float(sin_hour),
        "cos_hour": float(cos_hour),
    }

def _rolling_stats(last24: deque) -> Dict[str, float]:
    arr = np.array(last24, dtype=float)
    return {
        "target_roll_mean_24": float(arr.mean()),
        "target_roll_std_24": float(arr.std(ddof=0)),
    }

def build_features_future_recursive(
    state: str,
    dt: Union[str, datetime],
    model,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    dt dataset'in max'ından büyükse:
    - son 24 gerçek değerden deque başlat
    - max_dt+1'den dt'ye kadar saat saat ilerle
    - her adımda model tahminini "yeni target" gibi deque'ye ekle
    - en sonda dt için feature dict döndür
    """
    state = state.strip().upper()
    dt = _ensure_datetime(dt)
    df = _get_state_df(state)

    # Datetime kolonu polars datetime ise:
    if "Datetime" not in df.columns:
        raise ValueError("Processed parquet içinde 'Datetime' kolonu yok.")

    df = df.sort("Datetime")
    max_dt_pl = df.select(pl.col("Datetime").max()).item()
    max_dt = max_dt_pl.to_pydatetime()

    # geleceğe gitmiyorsa burada kullanılmaz
    if dt <= max_dt:
        raise ValueError("build_features_future_recursive sadece gelecekteki dt için kullanılmalı.")

    target_col = _pick_target_col(df, state)

    # Son 24 gerçek değer
    tail = df.select(["Datetime", target_col]).tail(24)
    if tail.height < 24:
        raise ValueError("Recursive forecast için en az 24 saatlik geçmiş gerekir.")

    last24 = deque(tail[target_col].to_list(), maxlen=24)

    # kaç saat ileri?
    steps = int((dt - max_dt).total_seconds() // 3600)
    if steps <= 0:
        steps = 1

    current_dt = max_dt

    # saat saat ileri sar
    for _ in range(steps):
        current_dt = current_dt.replace(minute=0, second=0, microsecond=0) + __import__("datetime").timedelta(hours=1)

        feats = _time_features(current_dt)
        feats["target_lag_1"] = float(last24[-1])
        feats["target_lag_24"] = float(last24[0])
        feats.update(_rolling_stats(last24))

        # model input kolon sırası
        x = build_feature_vector(feats, feature_names)
        y = model.predict(x)
        pred = float(y[0])

        # tahmini yeni gerçek gibi ekle
        last24.append(pred)

    # dt için feature set (son step sonunda current_dt == dt olmalı)
    feats = _time_features(current_dt)
    feats["target_lag_1"] = float(last24[-1])
    feats["target_lag_24"] = float(last24[0])
    feats.update(_rolling_stats(last24))
    return feats
