from prefect import flow
from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost

import os
import json
import joblib


@flow(name="evaluate-and-promote")
def evaluate_and_promote():

    # -------------------------
    # Config
    # -------------------------
    data_path = "data/processed/AEP_hourly_processed.parquet"
    target_col = "AEP_MW"

    print("Starting model training pipeline...")

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

    print(f"Winner model: {winner_name}")

    # -------------------------
    # 4️⃣ Save Artifacts
    # -------------------------
    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    # Model kaydet
    model_path = os.path.join(artifact_dir, "best_model.pkl")
    joblib.dump(winner_model, model_path)

    # Metadata kaydet
    metadata = {
        "model_type": winner_name,
        "mae": float(winner_mae),
        "data_path": data_path
    }

    metadata_path = os.path.join(artifact_dir, "best_model.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Model saved to artifacts/")
    print("Promotion completed successfully ✅")

    return {
        "winner": winner_name,
        "mae": winner_mae,
        "model_path": model_path
    }


if __name__ == "__main__":
    evaluate_and_promote()
