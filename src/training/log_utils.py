import mlflow


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_model(model, artifact_path: str):
    mlflow.sklearn.log_model(model, artifact_path)
