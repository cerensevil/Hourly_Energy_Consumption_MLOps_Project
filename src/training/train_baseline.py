# src/training/train_baseline.py

import pandas as pd
import mlflow
import mlflow.pyfunc
from prefect import task
from src.models.baseline_model import Baseline24hModel


@task
def train_baseline(data_path: str, target_col: str):

    df = pd.read_parquet(data_path)

    model = Baseline24hModel(target_col)
    model.fit(df)

    predictions = model.predict()

    # Basit MAE hesapla (son 24 vs bir önceki 24)
    actual = df[target_col].tail(24).values
    mae = abs(actual - predictions).mean()

    mlflow.set_experiment("energy-baseline")

    with mlflow.start_run():

        mlflow.log_param("model_type", "baseline_24h")
        mlflow.log_metric("mae", mae)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model
        )

    print("Baseline training completed.")
