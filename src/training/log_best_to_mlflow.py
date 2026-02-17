import json
from pathlib import Path
import mlflow
import joblib

PROD = Path("src/registry/production.json")

def main():
    cfg = json.loads(PROD.read_text(encoding="utf-8"))
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    exp_name = cfg["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    model_path = Path(cfg["model_path"])
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="promoted-best") as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="energy_forecasting_model",
        )

        run_id = run.info.run_id
        print("Logged run_id:", run_id)

        # production.json içine model_uri yaz (en pratik yaklaşım)
        cfg["loader"] = "mlflow"
        cfg["mlflow"]["run_id"] = run_id
        cfg["mlflow"]["model_uri"] = f"runs:/{run_id}/model"
        PROD.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("Updated production.json to MLflow loader.")

if __name__ == "__main__":
    main()
