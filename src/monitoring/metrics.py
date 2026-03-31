from prometheus_client import Counter, Histogram, Gauge

# ================================
# Prediction count
# ================================
prediction_count = Counter(
    "prediction_count_total",
    "Total number of predictions",
    ["state", "model_type", "model_version"]
)

# ================================
# Prediction latency
# ================================
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["state", "model_type", "model_version"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
)

# ================================
# Prediction errors (system errors)
# ================================
prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

# ================================
# Prediction value distribution
# ================================
prediction_value = Histogram(
    "prediction_value",
    "Distribution of prediction outputs",
    ["state", "model_type", "model_version"],
    buckets=[50000, 100000, 200000, 400000, 800000, 1200000]
)

# ================================
# Absolute error
# ================================
absolute_error = Histogram(
    "prediction_absolute_error",
    "Absolute error |y_true - y_pred|",
    ["state", "model_type", "model_version"],
    buckets=[1000, 5000, 10000, 20000, 50000, 100000]
)

# ================================
# Squared error (MODEL)
# ================================
squared_error = Histogram(
    "prediction_squared_error",
    "Squared error",
    ["state", "model_type", "model_version"],
    buckets=[1e6, 1e7, 5e7, 1e8, 5e8, 1e9]
)

# ================================
# Baseline squared error
# ================================
baseline_squared_error = Histogram(
    "baseline_squared_error",
    "Squared error of baseline model (t-24)",
    ["state"]
)

# ================================
# Under / Over prediction counters
# ================================
underprediction_count = Counter(
    "underprediction_total",
    "Number of underpredictions",
    ["state", "model_type", "model_version"]
)

overprediction_count = Counter(
    "overprediction_total",
    "Number of overpredictions",
    ["state", "model_type", "model_version"]
)

# ================================
# Cost Weighted Error (CWE)
# ================================
cost_weighted_error_metric = Histogram(
    "cost_weighted_error",
    "Cost weighted error",
    ["state", "model_type", "model_version"],
    buckets=[1000, 10000, 50000, 100000, 500000, 1000000]
)

# ================================
# Active model tracking (CRITICAL)
# ================================
active_model = Gauge(
    "active_model_info",
    "Active model per state",
    ["state", "model_type", "model_version"]
)

# ================================
# Retrain trigger counter
# ================================
retrain_trigger_count = Counter(
    "retrain_trigger_total",
    "Number of retraining triggers",
    ["state"]
)