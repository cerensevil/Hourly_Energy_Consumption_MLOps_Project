import polars as pl
import numpy as np
import os
import json
from pathlib import Path
from prefect import task, flow

@task(retries=3, retry_delay_seconds=5)
def process_file_to_parquet(csv_path: Path, output_dir: Path) -> str:
    file_name = csv_path.stem
    state_name = file_name.split("_")[0].upper()
    output_path = output_dir / f"{file_name}_processed.parquet"
    metadata_path = output_path.with_suffix(".json")

    # 1) Target column tespiti
    schema = pl.scan_csv(csv_path).collect_schema()
    all_cols = schema.names()
    target_col = [c for c in all_cols if c != "Datetime"][0]

    # 2) Lazy pipeline başlangıcı
    lf = pl.scan_csv(csv_path)

    # 3) Tip Dönüşümleri ve İsimlendirme
    lf = lf.with_columns([
        pl.col("Datetime").str.to_datetime(strict=False),
        pl.col(target_col).cast(pl.Float64, strict=False).alias("target"),
        pl.lit(state_name).alias("state")
    ])

    # 4) Temizlik (Zaman özellikleri öncesi Datetime temiz olmalı)
    lf = lf.drop_nulls(subset=["Datetime"]).sort("Datetime").unique(subset=["Datetime"])

    # 5) Zaman Özellikleri (Ayrı bir with_columns bloğu daha güvenlidir)
    lf = lf.with_columns([
        pl.col("Datetime").dt.hour().alias("hour"),
        pl.col("Datetime").dt.weekday().alias("dayofweek"),
        pl.col("Datetime").dt.month().alias("month"),
        pl.col("Datetime").dt.year().alias("year"),
        (pl.col("Datetime").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend")
    ])

    # 6) Cyclical Encoding & Lags
    lf = lf.with_columns([
        (pl.col("hour") * (2 * np.pi / 24)).sin().alias("sin_hour"),
        (pl.col("hour") * (2 * np.pi / 24)).cos().alias("cos_hour"),
        pl.col("target").shift(1).alias("target_lag_1"),
        pl.col("target").shift(24).alias("target_lag_24"),
    ])

    # 7) Rolling
    lf = lf.with_columns([
        pl.col("target").shift(1).rolling_mean(window_size=24).alias("target_roll_mean_24"),
        pl.col("target").shift(1).rolling_std(window_size=24).alias("target_roll_std_24"),
    ])

    # 8) Final Collect
    df_final = lf.drop_nulls().collect()

    if df_final.height > 0:
        df_final.write_parquet(output_path)
        metadata = {
            "state": state_name,
            "row_count": int(df_final.height),
            "feature_columns": df_final.columns
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"✅ Başarıyla işlendi: {state_name}")
        return str(output_path)
    
    return ""

@flow(name="Energy Data Multi-File Pipeline")
def energy_pipeline(raw_data_dir: str = "data/raw_data"):
    raw_path, processed_path = Path(raw_data_dir), Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    for f in processed_path.glob("*.parquet"): f.unlink()
    
    raw_files = list(raw_path.glob("*.csv"))
    for file in raw_files:
        process_file_to_parquet.fn(file, processed_path)

if __name__ == "__main__":
    energy_pipeline()