from prefect import flow
from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost

import os
import json
import joblib
from pathlib import Path


@flow(name="evaluate-and-promote")
def evaluate_and_promote():

    # -------------------------
    # Config
    # -------------------------
    data_path = "data/processed/AEP_hourly_processed.parquet"
    target_col = "AEP_MW"

    print("🚀 Starting model training pipeline...")

    # -------------------------
    # 1️⃣ Train Baseline
    # -------------------------
    baseline_result = train_baseline(data_path, target_col)

    if "model" not in baseline_result or "mae" not in baseline_result:
        raise ValueError("train_baseline must return {'model': ..., 'mae': ...}")

    print("Baseline MAE:", baseline_result["mae"])

    # -------------------------
    # 2️⃣ Train XGBoost
    # -------------------------
    xgb_result = train_xgboost(data_path, target_col)

    if "model" not in xgb_result or "mae" not in xgb_result:
        raise ValueError("train_xgboost must return {'model': ..., 'mae': ...}")

    print("XGBoost MAE:", xgb_result["mae"])

    # -------------------------
    # 3️⃣ Compare & Select Winner
    # -------------------------
    if xgb_result["mae"] < baseline_result["mae"]:
        winner_name = "xgboost"
        winner_model = xgb_result["model"]
        winner_mae = xgb_result["mae"]
    else:
        winner_name = "baseline"
        winner_model = baseline_result["model"]
        winner_mae = baseline_result["mae"]

    print(f"🏆 Winner model: {winner_name}")

    # -------------------------
    # 4️⃣ Save Artifacts (Registry uyumlu)
    # -------------------------
    registry_artifact_dir = Path("src/registry/artifacts")
    registry_artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = registry_artifact_dir / "best_model.pkl"
    metadata_path = registry_artifact_dir / "best_model.json"

    # Model kaydet
    joblib.dump(winner_model, model_path)

    # Metadata kaydet
    metadata = {
        "model_type": winner_name,
        "mae": float(winner_mae),
        "data_path": data_path
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("📦 Model saved to src/registry/artifacts/")

    # -------------------------
    # 5️⃣ Update production.json
    # -------------------------
    production_file = Path("src/registry/production.json")

    if production_file.exists():
        prod_cfg = json.loads(production_file.read_text(encoding="utf-8"))
    else:
        prod_cfg = {}

    prod_cfg.update({
        "model_type": winner_name,
        "loader": "local",
        "model_path": str(model_path),
        "feature_names": [
            "hour", "dayofweek", "month", "year", "is_weekend",
            "sin_hour", "cos_hour",
            "target_lag_1", "target_lag_24",
            "target_roll_mean_24", "target_roll_std_24"
        ],
        "mlflow": prod_cfg.get("mlflow", {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "energy_forecasting",
            "run_id": None,
            "model_uri": None
        })
    })

    production_file.write_text(json.dumps(prod_cfg, indent=2), encoding="utf-8")

    print("📝 production.json updated")
    print("✅ Promotion completed successfully")

    return {
        "winner": winner_name,
        "mae": winner_mae,
        "model_path": str(model_path)
    }


if __name__ == "__main__":
    evaluate_and_promote()
