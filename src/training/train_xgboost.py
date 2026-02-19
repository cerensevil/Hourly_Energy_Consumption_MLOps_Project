import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.models.xgboost_model import XGBoostModel


def train_xgboost(data_path: str, target_col: str):

    # 1️⃣ Load data
    df = pd.read_parquet(data_path)

    # -------------------------
    # Feature temizliği
    # -------------------------

    # Datetime model feature olamaz
    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    # State metadata'dır, model feature değildir
    if "state" in df.columns:
        df = df.drop(columns=["state"])

    # Kritik: raw ölçüm kolonu feature olmasın (gelecekte yok)
    if "AEP_MW" in df.columns:
        df = df.drop(columns=["AEP_MW"])    

    # -------------------------
    # 2️⃣ Train / Test split
    # Son 24 saat test
    # -------------------------
    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # -------------------------
    # 3️⃣ Model
    # -------------------------
    model = XGBoostModel()
    model.train(X_train, y_train)

    # -------------------------
    # 4️⃣ Predict
    # -------------------------
    predictions = model.predict(X_test)

    # -------------------------
    # 5️⃣ Evaluate
    # -------------------------
    mae = mean_absolute_error(y_test, predictions)

    return {
        "model": model,
        "mae": float(mae)
    }
