import time
import random
import requests
import polars as pl
from pathlib import Path
import json

API_URL = "http://localhost:8000/predict"


def load_registry():
    registry_path = Path("src/registry/production.json")
    return json.loads(registry_path.read_text())


def load_registry_states(registry):
    return list(registry["states"].keys())


def load_state_data(state: str):
    path = Path(f"data/processed/{state}_hourly_processed.parquet")

    if not path.exists():
        print(f"⚠️ Missing data for {state}, skipping...")
        return None

    df = pl.read_parquet(path)
    return df


def simulate_one_step(state: str, rows, feature_cols):

    row = random.choice(rows)

    # ✅ SADECE MODELİN BEKLEDİĞİ FEATURE'LAR
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


def run_live_simulation(delay: float = 0.5):

    print("🚀 Starting multi-state live simulation")

    registry = load_registry()
    states = load_registry_states(registry)

    print(f"States: {states}")

    state_data = {}

    for state in states:
        df = load_state_data(state)

        if df is None:
            continue

        df = df.filter(pl.col("Datetime").dt.year() == 2018)

        if df.height == 0:
            print(f"⚠️ No 2018 data for {state}")
            continue

        feature_cols = registry["states"][state]["feature_names"]

        state_data[state] = {
            "rows": df.to_dicts(),
            "features": feature_cols
        }

    print(f"✅ Loaded states: {list(state_data.keys())}")

    while True:

        for state, data in state_data.items():
            simulate_one_step(
                state,
                data["rows"],
                data["features"]
            )

        time.sleep(delay)


if __name__ == "__main__":
    run_live_simulation(delay=0.5)