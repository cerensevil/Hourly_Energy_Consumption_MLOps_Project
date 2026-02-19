# tests/conftest.py
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# -------------------------------------------------
# Yardımcı fonksiyon
# -------------------------------------------------

def _list_parquets() -> List[Path]:
    files = sorted(PROCESSED_DIR.glob("*_processed.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No processed parquet files found in: {PROCESSED_DIR}\n"
            f"Expected pattern: *_processed.parquet"
        )

    return files


# -------------------------------------------------
# 1) Parametrize: tüm dosyalar için test çalıştır
# -------------------------------------------------

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "parquet_path" in metafunc.fixturenames:
        files = _list_parquets()
        metafunc.parametrize(
            "parquet_path",
            files,
            ids=[p.name for p in files],
        )


# -------------------------------------------------
# Fixture'lar
# -------------------------------------------------

@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return PROCESSED_DIR


@pytest.fixture(scope="session")
def parquet_files(processed_dir: Path) -> List[Path]:
    # processed_dir bağımlılığı fixture grafiği için kalsın
    return _list_parquets()


@pytest.fixture
def df(parquet_path: Path) -> pl.DataFrame:
    return pl.read_parquet(parquet_path)


@pytest.fixture(scope="session")
def baseline_df(processed_dir: Path) -> pl.DataFrame:
    aep = processed_dir / "AEP_hourly_processed.parquet"

    if aep.exists():
        path = aep
    else:
        path = _list_parquets()[0]

    return pl.read_parquet(path)
