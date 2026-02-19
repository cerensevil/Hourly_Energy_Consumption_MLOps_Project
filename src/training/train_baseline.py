import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.models.baseline_model import BaselineModel


def train_baseline(data_path: str, target_col: str):

    # 1️⃣ Load data
    df = pd.read_parquet(data_path)

    # -------------------------
    # Feature temizliği
    # -------------------------

    # Datetime model feature değil
    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    # State metadata'dır
    if "state" in df.columns:
        df = df.drop(columns=["state"])

    # -------------------------
    # 2️⃣ Train / Test split
    # Son 24 saat test
    # -------------------------
    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]

    # -------------------------
    # 3️⃣ Model
    # -------------------------
    model = BaselineModel(target_col=target_col)

    # Baseline model fit gerektirmez ama mimari uyumu için çağırıyoruz
    model.fit(train_df)

    # -------------------------
    # 4️⃣ Forecast
    # -------------------------
    predictions = model.forecast_next_24(train_df)

    # -------------------------
    # 5️⃣ Evaluate
    # -------------------------
    mae = mean_absolute_error(test_df[target_col], predictions)

    return {
        "model": model,
        "mae": float(mae)
    }
