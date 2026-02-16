from prefect import flow
import pandas as pd
import mlflow

from src.models.baseline_model import Baseline24hModel
from src.pipeline.data.split import time_series_split
from src.training.evaluate import evaluate_regression


@flow(name="baseline-training-flow")
def baseline_training_flow():

    data_path = "data/processed/AEP_hourly_processed.parquet"
    target_col = "AEP"

    df = pd.read_parquet(data_path)

    train, test = time_series_split(df, target_col)

    model = Baseline24hModel(target_col)
    model.fit(train)

    preds = model.predict()

    metrics = evaluate_regression(
        test[target_col].values,
        preds
    )

    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({"model_type": "baseline"})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    return metrics
