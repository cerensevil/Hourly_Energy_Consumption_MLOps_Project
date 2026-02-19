from prefect import flow
from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost

import json
import joblib
from pathlib import Path
import polars as pl


@flow(name="evaluate-and-promote-multi-state")
def evaluate_and_promote():

    processed_dir = Path("data/processed")
    artifact_root = Path("src/registry/artifacts")
    production_file = Path("src/registry/production.json")

    parquet_files = list(processed_dir.glob("*_processed.parquet"))

    if not parquet_files:
        print("❌ Processed veri bulunamadı.")
        return

    print(f"\n🚀 Toplam {len(parquet_files)} state işlenecek.\n")

    # ==========================================================
    # Yeni production config oluştur (temiz başlangıç)
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
    # LOOP – HER STATE İÇİN MODEL EĞİT
    # ==========================================================
    for file in parquet_files:

        print("=" * 60)
        print(f"İşleniyor: {file.name}")

        df = pl.read_parquet(file)
        state = df["state"][0]

        print(f"State: {state}")

        # -------------------------
        # Train Models
        # -------------------------
        baseline_result = train_baseline(str(file), "target")
        xgb_result = train_xgboost(str(file), "target")

        print("Baseline MAE:", baseline_result["mae"])
        print("XGBoost MAE:", xgb_result["mae"])

        # -------------------------
        # Compare
        # -------------------------
        if xgb_result["mae"] < baseline_result["mae"]:
            winner_name = "xgboost"
            winner_model = xgb_result["model"]
            winner_mae = xgb_result["mae"]
        else:
            winner_name = "baseline"
            winner_model = baseline_result["model"]
            winner_mae = baseline_result["mae"]

        print(f"🏆 Winner: {winner_name}")

        # -------------------------
        # Save Artifacts (state bazlı)
        # -------------------------
        state_dir = artifact_root / state
        state_dir.mkdir(parents=True, exist_ok=True)

        model_path = state_dir / "best_model.pkl"
        metadata_path = state_dir / "best_model.json"

        joblib.dump(winner_model, model_path)

        metadata = {
            "state": state,
            "model_type": winner_name,
            "mae": float(winner_mae),
            "data_file": str(file)
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(f"📦 Model saved to {state_dir}")

        # -------------------------
        # production.json → states
        # -------------------------
        production_config["states"][state] = {
            "model_type": winner_name,
            "loader": "local",
            "model_path": str(model_path),
            "feature_names": [
                "hour", "dayofweek", "month", "year", "is_weekend",
                "sin_hour", "cos_hour",
                "target_lag_1", "target_lag_24",
                "target_roll_mean_24", "target_roll_std_24"
            ]
        }

    # ==========================================================
    # production.json overwrite
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
