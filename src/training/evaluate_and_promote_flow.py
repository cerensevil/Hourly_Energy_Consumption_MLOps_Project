from prefect import flow

from src.config.dataset_config import DATASET_VERSION
from src.training.model_runner import run_models
from src.training.model_selection import select_best_model

import json
import joblib
from pathlib import Path
import polars as pl
from datetime import datetime, UTC

import mlflow
from mlflow.tracking import MlflowClient


FEATURE_NAMES = [
    "hour", "dayofweek", "month", "year", "is_weekend",
    "sin_hour", "cos_hour",
    "target_lag_1", "target_lag_24",
    "target_roll_mean_24", "target_roll_std_24"
]


@flow(name="evaluate-and-promote-multi-state")
def evaluate_and_promote():

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("energy_forecasting")

    client = MlflowClient()

    processed_dir = Path("data/processed")
    artifact_root = Path("src/registry/artifacts")
    production_file = Path("src/registry/production.json")

    parquet_files = list(processed_dir.glob("*_processed.parquet"))

    if not parquet_files:
        print("❌ Processed veri bulunamadı.")
        return

    print(f"\n🚀 Toplam {len(parquet_files)} state işlenecek.\n")

    production_config = {
        "mlflow": {
            "tracking_uri": "http://localhost:5000"
        },
        "states": {}
    }

    for file in parquet_files:

        print("=" * 60)
        print(f"İşleniyor: {file.name}")

        df = pl.read_parquet(file)
        state = df["state"][0]

        print(f"State: {state}")

        results = {}
        run_ids = {}

        raw_results = run_models(str(file), "target")

        print("\n📊 Model Metrics")

        # -------------------------------
        # STEP 1 — LOG ALL MODELS
        # -------------------------------
        for name, res in raw_results.items():

            with mlflow.start_run(run_name=f"{state}_{name}"):

                mlflow.log_param("model_type", name)
                mlflow.log_param("state", state)

                mlflow.log_metric("mae", res["mae"])
                mlflow.log_metric("rmse", res["rmse"])
                mlflow.log_metric("cwe", res["cwe"])

                mlflow.sklearn.log_model(
                    sk_model=res["model"],
                    name="model"
                )

                run_id = mlflow.active_run().info.run_id

            results[name] = res
            run_ids[name] = run_id

        # -------------------------------
        # STEP 2 — SELECT WINNER
        # -------------------------------
        winner_name, winner_result = select_best_model(results)

        baseline_rmse = results["baseline"]["rmse"]

        if winner_result["rmse"] > baseline_rmse * 1.2:
            print("⚠️ RMSE guardrail violated → fallback to baseline")
            winner_name = "baseline"
            winner_result = results["baseline"]

        winner_model = winner_result["model"]
        winner_run_id = run_ids[winner_name]

        print(f"\n🏆 Winner: {winner_name}")

        model_name = f"energy_model_{state}"

        # -------------------------------
        # STEP 3 — REGISTER WINNER
        # -------------------------------
        with mlflow.start_run(run_id=winner_run_id):

            mlflow.sklearn.log_model(
                sk_model=winner_model,
                name="model",
                registered_model_name=model_name
            )

        # -------------------------------
        # STEP 4 — PROMOTE
        # -------------------------------
        versions = client.search_model_versions(f"name='{model_name}'")

        if versions:
            latest_version = max(int(v.version) for v in versions)

            for v in versions:
                if v.current_stage == "Production":
                    client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived"
                    )

            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production"
            )

            print(f"🚀 Promoted to Production: {model_name} v{latest_version}")

            client.set_model_version_tag(
                name=model_name,
                version=latest_version,
                key="selection_metric",
                value="cwe"
            )

            client.set_model_version_tag(
                name=model_name,
                version=latest_version,
                key="feature_names",
                value=json.dumps(FEATURE_NAMES)
            )

        # -------------------------------
        # STEP 5 — LOCAL BACKUP
        # -------------------------------
        state_dir = artifact_root / state
        state_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(winner_model, state_dir / "best_model.pkl")

        metadata = {
            "state": state,
            "model_type": winner_name,
            "mae": float(winner_result["mae"]),
            "rmse": float(winner_result["rmse"]),
            "cwe": float(winner_result["cwe"]),
            "run_id": winner_run_id,
            "model_name": model_name,
            "dataset_version": DATASET_VERSION,
            "feature_names": FEATURE_NAMES,
            "timestamp": datetime.now(UTC).isoformat()
        }

        (state_dir / "best_model.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(f"📦 Model saved to {state_dir}")

        # -------------------------------
        # STEP 6 — PRODUCTION CONFIG
        # -------------------------------
        production_config["states"][state] = {
            "model_type": winner_name,
            "model_name": model_name,
            "loader": "mlflow",
            "model_uri": f"models:/{model_name}/Production",
            "dataset_version": DATASET_VERSION,
            "feature_names": FEATURE_NAMES
        }

    # -------------------------------
    # WRITE CONFIG
    # -------------------------------
    production_file.write_text(
        json.dumps(production_config, indent=2),
        encoding="utf-8"
    )

    print("\n📝 production.json güncellendi")
    print("✅ FULL production pipeline ready")

    return {
        "states_processed": list(production_config["states"].keys())
    }


if __name__ == "__main__":
    evaluate_and_promote()