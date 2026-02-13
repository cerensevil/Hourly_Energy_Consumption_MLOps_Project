import polars as pl
import numpy as np
import os
from pathlib import Path
from prefect import task, flow


# =====================================================
# TASK
# =====================================================

@task(retries=3, retry_delay_seconds=5)
def process_file_to_parquet(csv_path: Path, output_dir: Path) -> str:
    """
    CSV'yi okur, hedef sütunu otomatik bulur,
    feature engineering yapar ve processed klasörüne parquet olarak kaydeder.
    """

    file_name = csv_path.stem
    output_path = output_dir / f"{file_name}_processed.parquet"

    # -------------------------
    # 1) Target column tespiti
    # -------------------------
    schema = pl.scan_csv(csv_path).schema
    all_cols = list(schema.keys())
    target_col = [c for c in all_cols if c != "Datetime"][0]

    print(f"---> İşleniyor: {file_name} | Hedef Sütun: {target_col}")

    # -------------------------
    # 2) Lazy pipeline
    # -------------------------
    lf = pl.scan_csv(csv_path)

    # Datetime parse + numeric cast
    lf = lf.with_columns([
        pl.col("Datetime").str.to_datetime(strict=False),
        pl.col(target_col).cast(pl.Float64, strict=False)
    ])

    # Datetime null olanları at
    lf = lf.drop_nulls(subset=["Datetime"])

    # Sort
    lf = lf.sort("Datetime")

    # 🔥 DST duplicate fix (CRITICAL)
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
    # 5) Lag & Rolling (Leakage-safe)
    # -------------------------
    lf = lf.with_columns([
        pl.col(target_col).shift(1).alias("target_lag_1"),
        pl.col(target_col).shift(24).alias("target_lag_24"),
        pl.col(target_col)
            .shift(1)
            .rolling_mean(window_size=24)
            .alias("target_roll_mean_24"),
        pl.col(target_col)
            .shift(1)
            .rolling_std(window_size=24)
            .alias("target_roll_std_24")
    ])

    # -------------------------
    # 6) Null temizliği
    # -------------------------
    df_final = lf.drop_nulls().collect()

    # Eğer boş dataframe oluştuysa kaydetme
    if df_final.height == 0:
        print(f"UYARI: {file_name} için veri boş. Kaydedilmedi.")
        return ""

    # -------------------------
    # 7) Parquet kaydet
    # -------------------------
    df_final.write_parquet(output_path)

    return str(output_path)


# =====================================================
# FLOW
# =====================================================

@flow(name="Energy Data Multi-File Pipeline")
def energy_pipeline(raw_data_dir: str = "data/raw_data"):

    raw_path = Path(raw_data_dir)
    processed_path = Path("data/processed")

    processed_path.mkdir(parents=True, exist_ok=True)

    # Eski processed dosyaları temizle (production best practice)
    for f in processed_path.glob("*_processed.parquet"):
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

    print("\n" + "=" * 40)
    print(f"BAŞARILI: {len(processed_results)} dosya işlendi.")
    print(f"Çıktı klasörü: {processed_path}")
    print("=" * 40)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    os.environ["PREFECT_SERVER_STARTUP_TIMEOUT_SECONDS"] = "60"
    energy_pipeline()
