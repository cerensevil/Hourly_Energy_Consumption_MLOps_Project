import polars as pl
import numpy as np
import os
import json
from pathlib import Path
from prefect import task, flow


# =====================================================
# TASK
# =====================================================

@task(retries=3, retry_delay_seconds=5)
def process_file_to_parquet(csv_path: Path, output_dir: Path) -> str:
    """
    CSV'yi okur,
    - target kolonunu standartlaştırır
    - state kolonunu dosya isminden üretir
    - feature engineering yapar
    - parquet + metadata json olarak kaydeder
    """

    file_name = csv_path.stem                 # örn: AEP_hourly
    state_name = file_name.split("_")[0].upper()

    output_path = output_dir / f"{file_name}_processed.parquet"
    metadata_path = output_path.with_suffix(".json")

    # -------------------------
    # 1) Target column tespiti
    # -------------------------
    schema = pl.scan_csv(csv_path).schema
    all_cols = list(schema.keys())
    target_col = [c for c in all_cols if c != "Datetime"][0]

    print(f"---> İşleniyor: {file_name}")
    print(f"     State: {state_name}")
    print(f"     Orijinal Target: {target_col}")

    # -------------------------
    # 2) Lazy pipeline
    # -------------------------
    lf = pl.scan_csv(csv_path)

    lf = lf.with_columns([
        pl.col("Datetime").str.to_datetime(strict=False),
        pl.col(target_col).cast(pl.Float64, strict=False),
    ])

    # Target standartlaştır
    lf = lf.with_columns([
        pl.col(target_col).alias("target")
    ])
    
    # Orijinal target kolonunu drop et (AEP_MW vb. feature'a karışmasın)
    if target_col != "target":
        lf = lf.drop([target_col])

    # State kolonunu ekle
    lf = lf.with_columns([
        pl.lit(state_name).alias("state")
    ])

    # Temizlik
    lf = lf.drop_nulls(subset=["Datetime"])
    lf = lf.sort("Datetime")
    lf = lf.unique(subset=["Datetime"], keep="first")

    # -------------------------
    # 3) Zaman Feature'ları
    # -------------------------
    lf = lf.with_columns([
        pl.col("Datetime").dt.hour().alias("hour"),
        pl.col("Datetime").dt.weekday().alias("dayofweek"),
        pl.col("Datetime").dt.month().alias("month"),
        pl.col("Datetime").dt.year().alias("year"),
        (pl.col("Datetime").dt.weekday() >= 6)
            .cast(pl.Int8)
            .alias("is_weekend")
    ])

    # -------------------------
    # 4) Cyclical Encoding
    # -------------------------
    lf = lf.with_columns([
        (pl.col("hour") * (2 * np.pi / 24)).sin().alias("sin_hour"),
        (pl.col("hour") * (2 * np.pi / 24)).cos().alias("cos_hour"),
    ])

    # -------------------------
    # 5) Lag & Rolling
    # -------------------------
    lf = lf.with_columns([
        pl.col("target").shift(1).alias("target_lag_1"),
        pl.col("target").shift(24).alias("target_lag_24"),
        pl.col("target").shift(1).rolling_mean(24).alias("target_roll_mean_24"),
        pl.col("target").shift(1).rolling_std(24).alias("target_roll_std_24"),
    ])

    # -------------------------
    # 6) Null temizliği
    # -------------------------
    df_final = lf.drop_nulls().collect()

    if df_final.height == 0:
        print(f"UYARI: {file_name} için veri boş. Kaydedilmedi.")
        return ""

    # -------------------------
    # 7) Parquet kaydet
    # -------------------------
    df_final.write_parquet(output_path)

    # -------------------------
    # 8) Metadata kaydet
    # -------------------------
    metadata = {
        "state": state_name,
        "original_target_column": target_col,
        "standardized_target": "target",
        "row_count": int(df_final.height),
        "feature_columns": df_final.columns
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print(f"Kaydedildi: {output_path}")
    print(f"Metadata yazıldı: {metadata_path}")

    return str(output_path)


# =====================================================
# FLOW
# =====================================================

@flow(name="Energy Data Multi-File Pipeline")
def energy_pipeline(raw_data_dir: str = "data/raw_data"):

    raw_path = Path(raw_data_dir)
    processed_path = Path("data/processed")

    processed_path.mkdir(parents=True, exist_ok=True)

    # Eski processed dosyaları temizle
    for f in processed_path.glob("*_processed.parquet"):
        f.unlink()

    for f in processed_path.glob("*_processed.json"):
        f.unlink()

    raw_files = list(raw_path.glob("*.csv"))

    if not raw_files:
        print(f"UYARI: '{raw_path}' dizininde CSV bulunamadı!")
        return

    print(f"Sistem hazır. Toplam {len(raw_files)} dosya işleme alınıyor...\n")

    processed_results = []

    for file in raw_files:
        res = process_file_to_parquet(file, processed_path)
        if res:
            processed_results.append(res)

    print("\n" + "=" * 50)
    print(f"BAŞARILI: {len(processed_results)} dosya işlendi.")
    print(f"Çıktı klasörü: {processed_path}")
    print("=" * 50)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    os.environ["PREFECT_SERVER_STARTUP_TIMEOUT_SECONDS"] = "60"
    energy_pipeline()
