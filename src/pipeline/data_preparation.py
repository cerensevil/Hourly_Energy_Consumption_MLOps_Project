import polars as pl
import numpy as np
import os
from pathlib import Path
from prefect import task, flow

# ---------------------------
# Tasks
# ---------------------------

@task(retries=3, retry_delay_seconds=5)
def process_file_to_parquet(csv_path: Path, output_dir: Path) -> str:
    """
    CSV'yi okur, hedef sütunu otomatik bulur, feature engineering yapar 
    ve 'processed' klasörüne parquet olarak kaydeder.
    """
    file_name = csv_path.stem
    output_path = output_dir / f"{file_name}_processed.parquet"

    # 1. Sütun isimlerini hızlıca tara ve Datetime dışındaki ana sütunu bul
    schema = pl.scan_csv(csv_path).schema
    all_cols = list(schema.keys())
    target_col = [c for c in all_cols if c != "Datetime"][0]
    
    print(f"---> İşleniyor: {file_name} | Hedef Sütun: {target_col}")

    # 2. LazyFrame ile veri işleme hattını tanımla
    lf = pl.scan_csv(csv_path)

    # Zaman dönüşümü ve sıralama
    lf = lf.with_columns(
        pl.col("Datetime").str.to_datetime()
    ).sort("Datetime")

    # Temel Zaman Özellikleri
    lf = lf.with_columns([
        pl.col("Datetime").dt.hour().alias("hour"),
        pl.col("Datetime").dt.weekday().alias("dayofweek"),
        pl.col("Datetime").dt.month().alias("month"),
        pl.col("Datetime").dt.year().alias("year"),
        (pl.col("Datetime").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend")
    ])

    # Döngüsel Zaman Özellikleri (Sin/Cos)
    lf = lf.with_columns([
        (np.sin(2 * np.pi * pl.col("hour") / 24)).alias("sin_hour"),
        (np.cos(2 * np.pi * pl.col("hour") / 24)).alias("cos_hour"),
    ])

    # Lag ve Rolling Özellikleri (Shift(1) ile sızıntı/leakage önlenir)
    # Çıktı sütun isimlerini 'target_' ön ekiyle sabitledim ki model eğitirken kolaylık olsun
    lf = lf.with_columns([
        pl.col(target_col).shift(1).alias("target_lag_1"),
        pl.col(target_col).shift(24).alias("target_lag_24"),
        pl.col(target_col).shift(1).rolling_mean(window_size=24).alias("target_roll_mean_24"),
        pl.col(target_col).shift(1).rolling_std(window_size=24).alias("target_roll_std_24")
    ])

    # Hesaplamayı başlat ve oluşan Null satırları (lag/roll sebebiyle) temizle
    df_final = lf.drop_nulls().collect()

    # Parquet formatında diske yaz
    df_final.write_parquet(output_path)
    
    return str(output_path)

# Flow


@flow(name="Energy Data Multi-File Pipeline")
def energy_pipeline(raw_data_dir: str = "data/raw_data"):
    # Klasör yollarını Path objesine çevir
    raw_path = Path(raw_data_dir)
    processed_path = Path("data/processed")

    # 1. Çıktı klasörü yoksa oluştur
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Klasördeki tüm CSV dosyalarını listele
    raw_files = list(raw_path.glob("*.csv"))
    
    if not raw_files:
        print(f"UYARI: '{raw_path}' dizininde işlenecek CSV dosyası bulunamadı!")
        return

    print(f"Sistem hazır. Toplam {len(raw_files)} dosya işleme alınıyor...")

    # 3. Her dosya için işleme taskını çalıştır
    processed_results = []
    for file in raw_files:
        res = process_file_to_parquet(file, processed_path)
        processed_results.append(res)
    
    print("\n" + "="*30)
    print(f"BAŞARILI: {len(processed_results)} dosya işlendi ve '{processed_path}' klasörüne kaydedildi.")
    print("="*30)

if __name__ == "__main__":
    # Prefect zaman aşımı sorunlarını önlemek için opsiyonel çevre değişkeni
    os.environ["PREFECT_SERVER_STARTUP_TIMEOUT_SECONDS"] = "60"
    
    energy_pipeline()