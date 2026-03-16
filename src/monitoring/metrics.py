from prometheus_client import Counter, Histogram, Gauge

# -----------------------------------
# Prediction metrics
# -----------------------------------

prediction_count = Counter(
    "prediction_count",
    "Total number of predictions served"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests"
)

# -----------------------------------
# Model metrics
# -----------------------------------

model_rmse = Gauge(
    "model_rmse",
    "Current model RMSE"
)

# -----------------------------------
# Drift metrics
# -----------------------------------

dataset_drift_events = Counter(
    "dataset_drift_events",
    "Number of detected dataset drift events"
)

model_drift_events = Counter(
    "model_drift_events",
    "Number of detected model drift events"
)

# -----------------------------------
# Retraining metrics
# -----------------------------------

retraining_events = Counter(
    "retraining_events",
    "Number of retraining events triggered"
)