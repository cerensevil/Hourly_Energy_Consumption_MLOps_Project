from prometheus_client import Counter, Histogram, Gauge


# -----------------------------------
# Prediction metrics
# -----------------------------------

prediction_count = Counter(
    "prediction_count_total",
    "Total number of predictions served",
    ["state"]
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests"
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

prediction_values = Histogram(
    "prediction_value_distribution",
    "Distribution of prediction values"
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
    "dataset_drift_events_total",
    "Number of detected dataset drift events"
)

model_drift_events = Counter(
    "model_drift_events_total",
    "Number of detected model drift events"
)


# -----------------------------------
# Retraining metrics
# -----------------------------------

retraining_events = Counter(
    "retraining_events_total",
    "Number of retraining events triggered"
)