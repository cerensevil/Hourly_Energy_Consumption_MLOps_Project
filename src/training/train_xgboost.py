import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.models.xgboost_model import XGBoostModel

FEATURE_COLS = [
    "hour","dayofweek","month","year","is_weekend",
    "sin_hour","cos_hour",
    "target_lag_1","target_lag_24",
    "target_roll_mean_24","target_roll_std_24"
]

def train_xgboost(data_path: str, target_col: str = "target"):

    df = pd.read_parquet(data_path)

    # sadece gerekli kolonlar (target + featurelar) ile çalış
    keep = [target_col] + FEATURE_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    df = df[keep].copy()

    # Train/Test split (son 24 saat test)
    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[target_col]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[target_col]

    model = XGBoostModel()
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return {"model": model, "mae": float(mae), "feature_names": FEATURE_COLS}
