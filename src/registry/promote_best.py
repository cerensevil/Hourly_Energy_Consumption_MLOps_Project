import mlflow
from mlflow.tracking import MlflowClient


def promote_best_model(model_name: str):

    client = MlflowClient()

    runs = mlflow.search_runs(
        order_by=["metrics.rmse ASC"]
    )

    best_run_id = runs.iloc[0].run_id

    model_uri = f"runs:/{best_run_id}/model"

    client.create_registered_model(model_name)

    client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run_id
    )
