import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

API_URL = "http://ml_ops_app:8000"  # Docker network içinde servis adı

st.set_page_config(page_title="Energy Forecast", layout="centered")

st.title("⚡ Energy Consumption Forecast")

# ---------------------------------------------------
# 1️⃣ State Seçimi
# ---------------------------------------------------

states = [
    "DAYTON","NI","EKPC","DOM","COMED","DEOK",
    "FE","AEP","DUQ","PJMW","PJM","PJME"
]

selected_state = st.selectbox("Select State", states)

# ---------------------------------------------------
# 2️⃣ Datetime Seçimi
# ---------------------------------------------------

selected_date = st.date_input("Select Date")
selected_time = st.time_input("Select Time")

# ---------------------------------------------------
# 3️⃣ Predict Butonu
# ---------------------------------------------------

if st.button("Predict"):

    selected_datetime = datetime.combine(selected_date, selected_time)

    payload = {
        "state": selected_state,
        "datetime": selected_datetime.isoformat()
    }

    try:
        response = requests.post(
            f"{API_URL}/predict-from-datetime",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {round(result['prediction'], 2)} MW")
        else:
            st.error(response.json())

    except Exception as e:
        st.error(f"API Error: {str(e)}")

# ---------------------------------------------------
# 4️⃣ Model Leaderboard
# ---------------------------------------------------

st.markdown("---")
st.subheader("📊 Model Leaderboard")

leaderboard_path = Path("src/registry/leaderboards") / f"{selected_state}.json"

if leaderboard_path.exists():

    leaderboard_data = json.loads(leaderboard_path.read_text())

    df = pd.DataFrame(leaderboard_data["models"])

    st.dataframe(df)

    # ---------------------------------------------------
    # MAE Chart
    # ---------------------------------------------------

    st.subheader("📉 MAE Comparison")

    st.bar_chart(
        df.set_index("model")["mae"]
    )

    # ---------------------------------------------------
    # CWE Chart
    # ---------------------------------------------------

    st.subheader("💰 Cost Weighted Error (CWE)")

    st.bar_chart(
        df.set_index("model")["cwe"]
    )

    # ---------------------------------------------------
    # Combined Chart
    # ---------------------------------------------------

    st.subheader("⚡ Model Performance (MAE vs CWE)")

    st.bar_chart(
        df.set_index("model")[["mae", "cwe"]]
    )

else:

    st.info("Leaderboard henüz oluşturulmadı. Pipeline çalıştırıldığında görünecek.")