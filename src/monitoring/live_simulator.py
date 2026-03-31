import time
import random
import requests
import polars as pl
from pathlib import Path
import json

# NEW
from src.monitoring.drift_detection import run_drift_detection

API_URL = "http://localhost:8000/predict"


# =========================
# REGISTRY
# =========================
def load_registry():
    registry_path = Path("src/registry/production.json")
    return json.loads(registry_path.read_text())


def load_registry_states(registry):
    return list(registry["states"].keys())


# =========================
# DATA LOADING
# =========================
def load_state_data(state: str):
    path = Path(f"data/processed/{state}_hourly_processed.parquet")

    if not path.exists():
        print(f"⚠️ Missing data for {state}, skipping...")
        return None

    df = pl.read_parquet(path)
    return df


# =========================
# SIMULATION STEP
# =========================
def simulate_one_step(state: str, rows, feature_cols):

    row = random.choice(rows)

    features = {
        col: float(row[col])
        for col in feature_cols
    }

    actual = float(row["target"])

    payload = {
        "state": state,
        "features": features,
        "actual": actual
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            print(f"✅ {state}")
        else:
            print(f"❌ {state} error: {response.status_code}")

    except Exception as e:
        print(f"⚠️ {state} failed: {e}")


# =========================
# MAIN LOOP
# =========================
def run_live_simulation(delay: float = 0.5, drift_check_interval: int = 100):

    print("🚀 Starting multi-state live simulation")

    registry = load_registry()
    states = load_registry_states(registry)

    print(f"States: {states}")

    state_data = {}
    drift_counter = 0

    for state in states:
        df = load_state_data(state)

        if df is None:
            continue

        # 🔥 SPLIT: reference vs current
        reference_df = df.filter(pl.col("Datetime").dt.year() < 2018)
        current_df = df.filter(pl.col("Datetime").dt.year() == 2018)

        if current_df.height == 0 or reference_df.height == 0:
            print(f"⚠️ Not enough data for {state}")
            continue

        feature_cols = registry["states"][state]["feature_names"]

        state_data[state] = {
            "rows": current_df.to_dicts(),
            "features": feature_cols,
            "reference_df": reference_df,
            "current_df": current_df
        }

    print(f"✅ Loaded states: {list(state_data.keys())}")

    # =========================
    # LOOP
    # =========================
    while True:

        for state, data in state_data.items():

            simulate_one_step(
                state,
                data["rows"],
                data["features"]
            )

        drift_counter += 1

        # 🔥 DRIFT CHECK
        if drift_counter % drift_check_interval == 0:
            print("\n🔍 Running drift detection...\n")

            for state, data in state_data.items():
                run_drift_detection(
                    state,
                    data["reference_df"],
                    data["current_df"]
                )

        time.sleep(delay)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_live_simulation(delay=0.5, drift_check_interval=200)