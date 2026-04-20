import streamlit as st
import pandas as pd
import numpy as np

from src.api.state_data_cache import get_state_df
from src.api.model_loader import load_model_from_registry
from src.api.feature_builder import build_feature_vector

# ================================
# CONFIG
# ================================
st.set_page_config(layout="wide")
st.title("Energy Forecasting - Business Dashboard")

st.markdown("""
### Production Simulation Dashboard (2018 Data)

- Train: 2014–2017  
- Simulation: 2018  
- Focus: Cost & business impact
""")

# ================================
# CACHE
# ================================
@st.cache_data
def load_data(state):
    return get_state_df(state).to_pandas()


@st.cache_resource
def load_model_cached(state):
    return load_model_from_registry(state=state)


# ================================
# 🔥 FIXED FEATURE LIST
# ================================
DEFAULT_FEATURES = [
    "hour", "dayofweek", "month", "year", "is_weekend",
    "sin_hour", "cos_hour",
    "target_lag_1", "target_lag_24",
    "target_roll_mean_24", "target_roll_std_24"
]


@st.cache_data
def run_predictions_cached(df_pd, feature_names, target_col, state, under_penalty, over_penalty):

    model, _ = load_model_from_registry(state=state)

    valid_rows = []

    for _, row in df_pd.iterrows():
        features = {f: row[f] for f in feature_names if f in row}

        if len(features) != len(feature_names):
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
        row.get("target_lag_24", np.nan)
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

# 🔥 FIXED FEATURE NAMES
feature_names = cfg.get("feature_names")
if not feature_names or len(feature_names) == 0:
    feature_names = DEFAULT_FEATURES

# ================================
# PREPARE
# ================================
target_col = "target" if "target" in df_pd.columns else df_pd.columns[-1]

if "year" in df_pd.columns:
    df_pd = df_pd[df_pd["year"] == 2018]

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
# METRICS
# ================================
total_cost = df_result["cost"].sum()
baseline_cost = df_result["baseline_cost"].sum()

cost_diff = total_cost - baseline_cost
cost_diff_pct = (cost_diff / baseline_cost) * 100 if baseline_cost != 0 else 0

under_rate = (df_result["error"] > 0).mean()

# ================================
# UI
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Cost", f"${total_cost:,.0f}")
col2.metric("📉 vs Baseline", f"{cost_diff_pct:.2f}%", delta=f"${cost_diff:,.0f}")
col3.metric("⚠️ Underprediction", f"{under_rate:.2%}")
col4.metric("🔥 Peak Cost", f"${df_result['cost'].sum():,.0f}")

st.info(f"Active Model: {model_name}")

# ================================
# CHARTS
# ================================
st.subheader("📊 Actual vs Prediction")
st.line_chart(df_result[["actual", "prediction", "baseline"]])

st.subheader("💰 Cost Over Time")
st.line_chart(df_result["cost"])

# ================================
# RAW
# ================================
with st.expander("Raw Data"):
    st.dataframe(df_result.tail(50))