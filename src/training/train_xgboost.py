import pandas as pd

from src.models.xgboost_model import XGBoostModel
from src.training.evaluate import evaluate_regression


FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "year",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "target_lag_1",
    "target_lag_24",
    "target_roll_mean_24",
    "target_roll_std_24"
]


def train_xgboost(data_path: str, target_col: str):

    # 1️⃣ Load data
    df = pd.read_parquet(data_path)

    # -------------------------
    # Remove metadata columns
    # -------------------------

    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    if "state" in df.columns:
        df = df.drop(columns=["state"])

    # -------------------------
    # 🔹 TRAIN WINDOW
    # Only historical data
    # -------------------------

    train_df = df[df["year"] < 2018]

    # -------------------------
    # Feature selection
    # -------------------------

    X = train_df[FEATURE_COLUMNS]
    y = train_df[target_col]

    # -------------------------
    # Train / Validation split
    # last 24h validation
    # -------------------------

    X_train = X.iloc[:-24]
    X_test = X.iloc[-24:]

    y_train = y.iloc[:-24]
    y_test = y.iloc[-24:]

    # -------------------------
    # Model
    # -------------------------

    model = XGBoostModel()
    model.train(X_train, y_train)

    # -------------------------
    # Predict
    # -------------------------

    predictions = model.predict(X_test)

    # -------------------------
    # Evaluate
    # -------------------------

    metrics = evaluate_regression(y_test, predictions)

    return {
        "model": model,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "cwe": metrics["cwe"]
    }