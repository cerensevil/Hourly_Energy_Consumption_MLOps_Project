import pandas as pd
import os
from datetime import datetime

def build_features_from_datetime(state: str, dt: datetime):
    file_path = f"data/processed/{state}_hourly_processed.parquet"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")

    # Polars çıktısını Pandas indeksiyle senkronize et (Hız için kritik)
    df = pd.read_parquet(file_path)
    if "Datetime" in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index("Datetime", inplace=True)
        df.sort_index(inplace=True)

    # Seçilen tarihin mevcudiyetini kontrol et (Timeout önleyici)
    if dt not in df.index:
        raise ValueError(f"Tarih {dt} veri aralığı dışında! Arallık: {df.index.min()} - {df.index.max()}")

    # Sadece ilgili satırı sözlük olarak döndür
    return df.loc[[dt]].to_dict(orient='records')[0]

def build_feature_vector(features, feature_names):
    # Modelin beklediği sütun sırasına göre vektör oluşturur
    return pd.DataFrame([features])[feature_names]