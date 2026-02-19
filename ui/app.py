import os
import streamlit as st
import requests
from datetime import datetime, date, time

# =========================
# Config
# =========================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.write("API_URL_DEBUG:", API_URL)

st.set_page_config(page_title="Energy Forecast", layout="centered")
st.title("⚡ Energy Consumption Forecast")

# =========================
# Helpers
# =========================
@st.cache_data(ttl=300)
def fetch_available_range(api_url: str):
    """
    Backend'den min/max datetime aralığını çeker.
    Endpoint yoksa veya hata varsa None döner.
    """
    try:
        r = requests.get(f"{api_url}/available-range", timeout=10)
        if not r.ok:
            return None
        data = r.json()
        min_dt = datetime.fromisoformat(data["min_datetime"])
        max_dt = datetime.fromisoformat(data["max_datetime"])
        return min_dt, max_dt
    except Exception:
        return None

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {"detail": resp.text}

# =========================
# 1) State selection
# =========================
states = [
    "DAYTON", "NI", "EKPC", "DOM", "COMED", "DEOK",
    "FE", "AEP", "DUQ", "PJMW", "PJM", "PJME"
]

selected_state = st.selectbox("Select State", states)

# =========================
# 2) Datetime selection with range guard
# =========================
range_info = fetch_available_range(API_URL)

if range_info is not None:
    min_dt, max_dt = range_info

    st.caption(
        f"Available data range: **{min_dt.strftime('%Y-%m-%d %H:%M')}** → "
        f"**{max_dt.strftime('%Y-%m-%d %H:%M')}**"
    )

    # default değerleri range içine koy
    default_dt = min(max_dt, max(min_dt, datetime.now()))

    selected_date = st.date_input(
        "Select Date",
        value=default_dt.date(),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )

    # Streamlit time_input min/max desteklemediği için UI tarafında sadece “uyarı” yapacağız.
    selected_time = st.time_input("Select Time", value=default_dt.time().replace(second=0, microsecond=0))
else:
    st.warning(
        "Backend'den tarih aralığı alınamadı. (GET /available-range yoksa ekleyin.) "
        "Bu durumda veri dışı tarih seçerseniz API 'datetime bulunamadı' döndürebilir."
    )
    selected_date = st.date_input("Select Date", value=date.today())
    selected_time = st.time_input("Select Time", value=time(12, 0))

# =========================
# 3) Predict
# =========================
if st.button("Predict"):
    selected_datetime = datetime.combine(selected_date, selected_time).replace(second=0, microsecond=0)

    # Range kontrolü (endpoint varsa)
    if range_info is not None:
        if selected_datetime < min_dt or selected_datetime > max_dt:
            st.error(
                f"Selected datetime is outside available range.\n\n"
                f"Min: {min_dt}\nMax: {max_dt}\nSelected: {selected_datetime}"
            )
            st.stop()

    payload = {
        "state": selected_state,
        "datetime": selected_datetime.isoformat()
    }

    try:
        r = requests.post(
            f"{API_URL}/predict-from-datetime",
            json=payload,
            timeout=15
        )

        if r.ok:
            result = r.json()
            pred = result.get("prediction", None)
            if pred is None:
                st.error(f"API response has no 'prediction' field: {result}")
            else:
                st.success(f"Prediction: {float(pred):.2f} MW")
                with st.expander("Request / Response details"):
                    st.write("Request payload:", payload)
                    st.write("Response:", result)
        else:
            err = safe_json(r)
            detail = err.get("detail", err)
            st.error(detail)
            with st.expander("Debug details"):
                st.write("Request payload:", payload)
                st.write(f"Status code: {r.status_code}")
                st.write("Raw response:", r.text)

    except requests.exceptions.ConnectTimeout:
        st.error("API timeout: Connection timed out.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}. Is the FastAPI service running?")
    except Exception as e:
        st.error(f"API Error: {str(e)}")
