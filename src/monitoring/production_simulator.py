import polars as pl
import joblib
from pathlib import Path
import json

from src.monitoring.drift_detection import run_drift_detection
from src.monitoring.model_drift import detect_model_drift, save_drift_report
from src.metrics.regression_metrics import evaluate_all_metrics
from src.training.evaluate_and_promote_flow import evaluate_and_promote


def trigger_retraining(reason: str):

    print("\n======================================")
    print("⚠️ RETRAINING TRIGGERED")
    print(f"Reason: {reason}")
    print("======================================\n")

    evaluate_and_promote()

    print("\n✅ Retraining pipeline finished\n")


def compute_baseline_metrics(df: pl.DataFrame):

    # Persistence baseline: y(t-24)
    baseline_df = df.with_columns(
        pl.col("target").shift(24).alias("baseline_pred")
    ).drop_nulls()

    y_true = baseline_df["target"].to_numpy()
    y_pred = baseline_df["baseline_pred"].to_numpy()

    baseline_metrics = evaluate_all_metrics(y_true, y_pred)

    return baseline_metrics


def simulate_state(state: str, model_path: Path, data_path: Path, feature_cols):

    print(f"\n🔎 Simulating production for {state}")

    model = joblib.load(model_path)

    df = pl.read_parquet(data_path)

    # ------------------------------------------------
    # Reference dataset (training era)
    # ------------------------------------------------

    reference_df = df.filter(pl.col("Datetime").dt.year() < 2018)

    # ------------------------------------------------
    # Production dataset
    # ------------------------------------------------

    production_df = df.filter(pl.col("Datetime").dt.year() == 2018)

    if production_df.height == 0:
        print(f"⚠️ No 2018 data for {state}, skipping")
        return None, None

    # ------------------------------------------------
    # Data Drift Detection
    # ------------------------------------------------

    dataset_drift = run_drift_detection(state, reference_df, production_df)

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------

    y_true = production_df["target"].to_numpy()

    X = production_df.select(feature_cols).to_numpy()

    preds = model.predict(X)

    # ------------------------------------------------
    # Model Metrics
    # ------------------------------------------------

    metrics = evaluate_all_metrics(y_true, preds)

    print(f"📊 Model Metrics for {state}: {metrics}")

    # ------------------------------------------------
    # Baseline Metrics
    # ------------------------------------------------

    baseline_metrics = compute_baseline_metrics(production_df)

    baseline_rmse = baseline_metrics["rmse"]

    print(f"📊 Baseline RMSE for {state}: {baseline_rmse:.2f}")

    return metrics, baseline_rmse, dataset_drift


def run_production_simulation():

    registry_path = Path("src/registry/production.json")

    registry = json.loads(registry_path.read_text())

    states = registry["states"]

    results = {}

    drift_results = []

    retrain_required = False

    for state, cfg in states.items():

        print("\n--------------------------------------")
        print(f"STATE: {state}")
        print("--------------------------------------")

        model_path = Path(cfg["model_path"])

        data_path = Path(f"data/processed/{state}_hourly_processed.parquet")

        if not data_path.exists():
            print(f"⚠️ Missing data for {state}")
            continue

        feature_cols = cfg["feature_names"]

        metrics, baseline_rmse, dataset_drift = simulate_state(
            state,
            model_path,
            data_path,
            feature_cols
        )

        if not metrics:
            continue

        results[state] = metrics

        # ------------------------------------------------
        # Model Drift Detection
        # ------------------------------------------------

        drift = detect_model_drift(
            state,
            metrics,
            baseline_rmse
        )

        drift_results.append(drift)

        # ------------------------------------------------
        # Drift Decision Logic
        # ------------------------------------------------

        if dataset_drift:

            print(f"\n⚠️ DATA DRIFT DETECTED for {state}")

            retrain_required = True

        if drift["drift_detected"]:

            print(f"\n⚠️ MODEL DRIFT DETECTED for {state}")
            print(f"Flags: {drift['flags']}")

            retrain_required = True

    # ------------------------------------------------
    # Save model drift report
    # ------------------------------------------------

    save_drift_report(drift_results)

    # ------------------------------------------------
    # Retraining trigger
    # ------------------------------------------------

    if retrain_required:

        trigger_retraining(
            reason="drift_detected_in_production"
        )

    else:

        print("\n✅ No retraining required")

    return results


if __name__ == "__main__":

    results = run_production_simulation()

    print("\n======================================")
    print("Production Simulation Results")
    print("======================================")
    print(results)