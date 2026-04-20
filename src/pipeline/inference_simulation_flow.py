from prefect import flow, task
import requests
import time
from datetime import timedelta
import polars as pl


API_URL = "http://localhost:8000/predict-from-datetime"


@task(retries=3, retry_delay_seconds=5)
def send_request(state, dt):

    payload = {
        "state": state,
        "datetime": dt
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code != 200:
            print(f"⚠️ {state} error: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")


@flow(name="inference-simulation-flow", log_prints=True)
def inference_simulation_flow():

    print("🚀 2018 simulation başladı...\n")

    files = list(pl.read_parquet("data/processed").glob("*_processed.parquet"))

    for file in files:

        df = pl.read_parquet(file)

        # 🔥 sadece 2018 al
        df_2018 = df.filter(pl.col("datetime").dt.year() == 2018)

        state = df["state"][0]

        print(f"\n📍 State: {state} → {len(df_2018)} kayıt")

        for row in df_2018.iter_rows(named=True):

            dt = row["datetime"].isoformat()

            send_request(state, dt)

            time.sleep(0.2)  # gerçek zaman hissi