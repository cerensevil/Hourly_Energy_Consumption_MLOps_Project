from prefect import flow
import pandas as pd
import mlflow

from src.models.xgboost_model import XGBoostModel
from src.pipeline.data.split import time_series_split
from src.training.evaluate import evaluate_regression


@flow(name="xgboost-training-flow")
def xgboost_training_flow():

    data_path = "data/processed/AEP_hourly_processed.parquet"
    target_col = "AEP"

    df = pd.read_parquet(data_path)

    train, test = time_series_split(df, target_col)

    model = XGBoostModel(target_col)
    model.fit(train)

    X_test = test.drop(columns=[target_col])
    preds = model.predict(X_test)

    metrics = evaluate_regression(
        test[target_col].values,
        preds
    )

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({"model_type": "xgboost"})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    return metrics
