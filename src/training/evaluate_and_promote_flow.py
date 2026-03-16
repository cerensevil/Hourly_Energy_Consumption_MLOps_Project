from prefect import flow

from src.config.dataset_config import DATASET_VERSION
from src.training.model_runner import run_models
from src.training.model_selection import select_best_model

import json
import joblib
from pathlib import Path
import polars as pl
from datetime import datetime


@flow(name="evaluate-and-promote-multi-state")
def evaluate_and_promote():

    processed_dir = Path("data/processed")
    artifact_root = Path("src/registry/artifacts")
    production_file = Path("src/registry/production.json")
    leaderboard_dir = Path("src/registry/leaderboards")

    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(processed_dir.glob("*_processed.parquet"))

    if not parquet_files:
        print("❌ Processed veri bulunamadı.")
        return

    print(f"\n🚀 Toplam {len(parquet_files)} state işlenecek.\n")

    # ==========================================================
    # Yeni production config
    # ==========================================================

    production_config = {
        "mlflow": {
            "tracking_uri": "http://mlflow:5000",
            "experiment_name": "energy_forecasting",
            "run_id": None,
            "model_uri": None
        },
        "states": {}
    }

    # ==========================================================
    # LOOP – HER STATE
    # ==========================================================

    for file in parquet_files:

        print("=" * 60)
        print(f"İşleniyor: {file.name}")

        df = pl.read_parquet(file)
        state = df["state"][0]

        print(f"State: {state}")

        # ------------------------------------------------------
        # Train Models
        # ------------------------------------------------------

        results = run_models(str(file), "target")

        print("\n📊 Model Metrics")

        for name, res in results.items():

            print(
                f"{name} | "
                f"MAE={res['mae']:.2f} "
                f"RMSE={res['rmse']:.2f} "
                f"CWE={res['cwe']:.2f}"
            )

        # ------------------------------------------------------
        # Select Winner
        # ------------------------------------------------------

        winner_name, winner_result = select_best_model(results)

        winner_model = winner_result["model"]

        print(f"\n🏆 Winner: {winner_name}")

        # ------------------------------------------------------
        # Save Artifacts
        # ------------------------------------------------------

        state_dir = artifact_root / state
        state_dir.mkdir(parents=True, exist_ok=True)

        model_path = state_dir / "best_model.pkl"
        metadata_path = state_dir / "best_model.json"

        joblib.dump(winner_model, model_path)

        metadata = {
            "state": state,
            "model_type": winner_name,
            "mae": float(winner_result["mae"]),
            "rmse": float(winner_result["rmse"]),
            "cwe": float(winner_result["cwe"]),
            "data_file": str(file),
            "dataset_version": DATASET_VERSION,
            "timestamp": datetime.utcnow().isoformat()
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(f"📦 Model saved to {state_dir}")

        # ------------------------------------------------------
        # Save Leaderboard
        # ------------------------------------------------------

        leaderboard_path = leaderboard_dir / f"{state}.json"

        leaderboard_data = {
            "state": state,
            "dataset_version": DATASET_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "models": []
        }

        for name, res in results.items():

            leaderboard_data["models"].append({
                "model": name,
                "mae": float(res["mae"]),
                "rmse": float(res["rmse"]),
                "cwe": float(res["cwe"])
            })

        leaderboard_path.write_text(
            json.dumps(leaderboard_data, indent=2),
            encoding="utf-8"
        )

        print(f"📊 Leaderboard saved: {leaderboard_path}")

        # ------------------------------------------------------
        # Update production config
        # ------------------------------------------------------

        production_config["states"][state] = {
            "model_type": winner_name,
            "loader": "local",
            "model_path": str(model_path),
            "dataset_version": DATASET_VERSION,
            "feature_names": [
                "hour", "dayofweek", "month", "year", "is_weekend",
                "sin_hour", "cos_hour",
                "target_lag_1", "target_lag_24",
                "target_roll_mean_24", "target_roll_std_24"
            ]
        }

    # ==========================================================
    # Write production.json
    # ==========================================================

    production_file.write_text(
        json.dumps(production_config, indent=2),
        encoding="utf-8"
    )

    print("\n📝 production.json state-aware olarak güncellendi")
    print("✅ Tüm state'ler için promotion tamamlandı")

    return {
        "states_processed": list(production_config["states"].keys())
    }


if __name__ == "__main__":
    evaluate_and_promote()