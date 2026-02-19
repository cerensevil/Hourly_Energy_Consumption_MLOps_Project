import streamlit as st
import requests
from datetime import datetime

API_URL = "http://energy_mlops_container:8000"

st.set_page_config(page_title="Energy Forecast", page_icon="⚡")
st.title("⚡ Energy Consumption Forecast")

@st.cache_data(ttl=600)
def fetch_prediction(state, dt_iso):
    resp = requests.post(f"{API_URL}/predict-from-datetime", json={"state": state, "datetime": dt_iso}, timeout=60)
    if resp.status_code != 200: raise Exception(resp.json().get("detail", "Hata"))
    return resp.json()

with st.form("main_form"):
    col1, col2 = st.columns(2)
    state = col1.selectbox("Eyalet", ["DAYTON", "PJM", "AEP", "COMED"])
    dt = col2.date_input("Tarih", datetime(2018, 2, 6))
    tm = col2.time_input("Saat")
    if st.form_submit_button("Tahmin Et", use_container_width=True):
        try:
            res = fetch_prediction(state, datetime.combine(dt, tm).isoformat())
            st.success(f"Tahmin: {round(res['prediction'], 2)} MW")
        except Exception as e: st.error(str(e))