import streamlit as st
import pandas as pd
import numpy as np

from src.api.state_data_cache import get_state_df
from src.api.model_loader import load_model_from_registry
from src.api.feature_builder import build_feature_vector

# ================================
# CONFIG
# ================================
st.markdown(
    """
    ### Production Simulation Dashboard (2018 Data)

    This dashboard simulates a real-world production environment using **2018 data**.

    - Models are trained on historical data (2014–2017)
    - 2018 is treated as **live production stream**
    - All metrics reflect **real-time operational impact**

    Focus: Business cost, risk, and model performance under production conditions
    """
)
st.set_page_config(layout="wide")
st.title("Energy Forecasting - Business Dashboard")

# ================================
# CACHE
# ================================
@st.cache_data
def load_data(state):
    return get_state_df(state).to_pandas()


@st.cache_resource
def load_model_cached(state):
    return load_model_from_registry(state=state)


@st.cache_data
def run_predictions_cached(df_pd, feature_names, target_col, state, under_penalty, over_penalty):

    model, _ = load_model_from_registry(state=state)

    valid_rows = []

    for _, row in df_pd.iterrows():
        features = {f: row[f] for f in feature_names if f in row}

        if len(features) == 0:
            continue

        try:
            x = build_feature_vector(features, feature_names)
            valid_rows.append((x, row))
        except:
            continue

    if len(valid_rows) == 0:
        return None

    X = np.vstack([item[0] for item in valid_rows])
    preds = model.predict(X)

    actuals = [row[target_col] for _, row in valid_rows]

    baselines = [
        row["target_lag_24"] if "target_lag_24" in row else np.nan
        for _, row in valid_rows
    ]

    df_result = pd.DataFrame({
        "actual": actuals,
        "prediction": preds,
        "baseline": baselines
    })

    df_result["error"] = df_result["actual"] - df_result["prediction"]

    df_result["cost"] = np.where(
        df_result["error"] > 0,
        df_result["error"] * under_penalty,
        np.abs(df_result["error"]) * over_penalty
    )

    df_result["baseline_error"] = df_result["actual"] - df_result["baseline"]

    df_result["baseline_cost"] = np.where(
        df_result["baseline_error"] > 0,
        df_result["baseline_error"] * under_penalty,
        np.abs(df_result["baseline_error"]) * over_penalty
    )

    return df_result


# ================================
# SIDEBAR
# ================================
st.sidebar.header("⚙️ Business Parameters")

under_penalty = st.sidebar.slider("Underprediction penalty", 0.0, 10.0, 3.0)
over_penalty = st.sidebar.slider("Overprediction penalty", 0.0, 10.0, 1.0)

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()

# ================================
# STATE
# ================================
state = st.selectbox(
    "Select State",
    ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI", "PJME", "PJMW"]
)

# ================================
# LOAD
# ================================
df_pd = load_data(state)
model, cfg = load_model_cached(state)

model_name = cfg.get("model_type", "unknown")
feature_names = cfg.get("feature_names", [])

# ================================
# PREPARE
# ================================
if "target" in df_pd.columns:
    target_col = "target"
else:
    target_col = df_pd.columns[-1]

# 2018 simulation
if "year" in df_pd.columns:
    df_pd = df_pd[df_pd["year"] == 2018]

# fallback features
if not feature_names:
    feature_names = [col for col in df_pd.columns if col != target_col]

df_pd = df_pd.tail(200)

# ================================
# RUN
# ================================
with st.spinner("Running predictions..."):
    df_result = run_predictions_cached(
        df_pd,
        feature_names,
        target_col,
        state,
        under_penalty,
        over_penalty
    )

if df_result is None or len(df_result) == 0:
    st.error("No predictions generated ❌")
    st.stop()

# ================================
# BUSINESS METRICS
# ================================
total_cost = df_result["cost"].sum()
baseline_cost = df_result["baseline_cost"].sum()

cost_diff = total_cost - baseline_cost
cost_diff_pct = (cost_diff / baseline_cost) * 100 if baseline_cost != 0 else 0

under_rate = (df_result["error"] > 0).mean()

# 🔥 PEAK ANALYSIS
if "hour" in df_pd.columns:
    hours = df_pd.tail(len(df_result))["hour"].values
else:
    hours = np.zeros(len(df_result))

df_result["hour"] = hours
df_result["is_peak"] = df_result["hour"].between(17, 21)

peak_cost = df_result[df_result["is_peak"]]["cost"].sum()
peak_under_rate = (df_result[df_result["is_peak"]]["error"] > 0).mean()

# 🔥 RISK METRICS
p95_cost = df_result["cost"].quantile(0.95)

under_cost = df_result[df_result["error"] > 0]["cost"].sum()
over_cost = df_result[df_result["error"] <= 0]["cost"].sum()

cost_per_unit = total_cost / df_result["actual"].sum()

df_result["rolling_cost"] = df_result["cost"].rolling(24).mean()

# ================================
# UI - TOP METRICS
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Cost", f"${total_cost:,.0f}")
col2.metric("📉 vs Baseline", f"{cost_diff_pct:.2f}%", delta=f"${cost_diff:,.0f}")
col3.metric("⚠️ Underprediction", f"{under_rate:.2%}")
col4.metric("🔥 Peak Cost", f"${peak_cost:,.0f}")

# ================================
# SECOND ROW
# ================================
col5, col6, col7, col8 = st.columns(4)

col5.metric("🔥 Peak Under", f"{peak_under_rate:.2%}")
col6.metric("⚠️ P95 Cost", f"${p95_cost:,.0f}")
col7.metric("🔴 Under Cost", f"${under_cost:,.0f}")
col8.metric("🟢 Over Cost", f"${over_cost:,.0f}")

# ================================
# MODEL INFO
# ================================
st.info(f"Active Model: {model_name}")

# ================================
# CHARTS
# ================================
st.subheader("💰 Cost Over Time")
st.line_chart(df_result["cost"])

st.subheader("📈 Rolling Cost (24h)")
st.line_chart(df_result["rolling_cost"])

st.subheader("📊 Actual vs Prediction")
st.line_chart(df_result[["actual", "prediction", "baseline"]])

st.subheader("📊 Cost Distribution")
st.bar_chart(df_result["cost"])

# ================================
# RAW
# ================================
with st.expander("Raw Data"):
    st.dataframe(df_result.tail(50))