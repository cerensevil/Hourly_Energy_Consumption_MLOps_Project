import pandas as pd

from src.models.baseline_model import BaselineModel
from src.training.evaluate import evaluate_regression


def train_baseline(data_path: str, target_col: str):

    df = pd.read_parquet(data_path)

    # Metadata kolonları model feature değildir
    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    if "state" in df.columns:
        df = df.drop(columns=["state"])

    # ------------------------------------------------
    # TRAIN WINDOW
    # sadece historical data
    # ------------------------------------------------

    df = df[df["year"] < 2018]

    # ------------------------------------------------
    # Train / Test split
    # ------------------------------------------------

    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model = BaselineModel(target_col)

    model.fit(train_df)

    preds = model.forecast_next_24(train_df)

    y_test = test_df[target_col].values

    # ------------------------------------------------
    # Evaluate
    # ------------------------------------------------

    metrics = evaluate_regression(y_test, preds)

    return {
        "model": model,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "cwe": metrics["cwe"]
    }