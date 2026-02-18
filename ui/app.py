import streamlit as st
import requests
import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Energy Forecast", layout="centered")

st.title("⚡ Energy Consumption Forecast")

st.markdown("Model: Production XGBoost")

# ------------------------
# User Inputs
# ------------------------

st.subheader("Datetime Seç")

selected_date = st.date_input("Tarih", datetime.date.today())
selected_time = st.time_input("Saat", datetime.datetime.now().time())

datetime_str = f"{selected_date}T{selected_time}"

st.subheader("Lag Feature Girişleri")

lag_1 = st.number_input("target_lag_1", value=12000.0)
lag_24 = st.number_input("target_lag_24", value=11800.0)
roll_mean = st.number_input("target_roll_mean_24", value=11950.0)
roll_std = st.number_input("target_roll_std_24", value=250.0)

# ------------------------
# Predict Button
# ------------------------

if st.button("Tahmin Yap 🚀"):

    payload = {
        "features": {
            "hour": selected_time.hour,
            "dayofweek": selected_date.weekday(),
            "month": selected_date.month,
            "year": selected_date.year,
            "is_weekend": 1 if selected_date.weekday() >= 5 else 0,
            "sin_hour": 0,  # API üretmiyor çünkü A modundayız
            "cos_hour": 0,
            "target_lag_1": lag_1,
            "target_lag_24": lag_24,
            "target_roll_mean_24": roll_mean,
            "target_roll_std_24": roll_std
        }
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"📈 Tahmin: {round(result['prediction'], 2)} MW")
        else:
            st.error(f"Hata: {response.text}")

    except Exception as e:
        st.error(f"API bağlantı hatası: {str(e)}")
