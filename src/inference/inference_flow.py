from prefect import flow
import requests
from datetime import datetime, timedelta
import time

API_URL = "http://localhost:8000/predict-from-datetime"


def get_states():
    try:
        r = requests.get("http://localhost:8000/available-states", timeout=5)
        states = r.json()["states"]
        print(f"States from API: {states}")
        return states
    except Exception as e:
        print(f"State fetch error: {e}")
        return ["DAYTON"]


# ❌ artık task değil
def send_request(state, dt):
    payload = {
        "state": state,
        "datetime": dt.isoformat()
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        print(f"{state} → {r.status_code}")
    except Exception as e:
        print(f"❌ {state} error: {e}")


@flow(name="inference-simulation-flow", log_prints=True)
def inference_flow(sleep_seconds: float = 0.5):

    print("🚀 Continuous 2018 simulation başladı...\n")

    states = get_states()
    print(f"States: {states}")

    start = datetime(2018, 1, 1)
    end = datetime(2019, 1, 1)

    current = start

    while True:

        # 🔥 tüm state'ler için request
        for state in states:
            send_request(state, current)

        current += timedelta(hours=1)

        # yıl bitince başa sar
        if current >= end:
            print("🔁 2018 replay restart")
            current = start

        time.sleep(sleep_seconds)