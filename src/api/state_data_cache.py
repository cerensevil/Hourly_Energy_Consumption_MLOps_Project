from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import polars as pl

# Basit RAM cache: state -> DataFrame
_STATE_DF_CACHE: Dict[str, pl.DataFrame] = {}


def _default_processed_path(state: str) -> Path:
    # Örn: data/processed/DAYTON_hourly_processed.parquet
    return Path("data/processed") / f"{state}_hourly_processed.parquet"


def get_state_df(state: str, processed_path: Optional[str] = None, force: bool = False) -> pl.DataFrame:
    """
    State parquet'i RAM'e bir kere yükler, sonraki çağrılarda RAM'den döner.
    force=True -> cache'i bypass edip yeniden yükler.
    """
    state = (state or "").strip().upper()
    if not state:
        raise ValueError("state boş olamaz.")

    if (not force) and state in _STATE_DF_CACHE:
        return _STATE_DF_CACHE[state]

    path = Path(processed_path) if processed_path else _default_processed_path(state)
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet bulunamadı: {path}")

    df = pl.read_parquet(path)
    _STATE_DF_CACHE[state] = df
    return df


def clear_state_cache(state: Optional[str] = None) -> None:
    """
    state=None -> tüm cache temizlenir
    state='AEP' -> sadece o state temizlenir
    """
    if state is None:
        _STATE_DF_CACHE.clear()
        return

    state = state.strip().upper()
    _STATE_DF_CACHE.pop(state, None)


def cache_stats() -> dict:
    return {
        "cached_states": sorted(_STATE_DF_CACHE.keys()),
        "count": len(_STATE_DF_CACHE),
    }
