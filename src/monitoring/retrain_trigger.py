import time
from datetime import datetime, timedelta

# Prometheus
from prometheus_client import Counter

# =========================
# METRICS
# =========================
RETRAIN_SIGNAL_COUNTER = Counter(
    "retrain_signal_total",
    "Number of retraining signals (NOT executed retraining)",
    ["reason"]
)

# =========================
# CONFIG
# =========================
COOLDOWN_MINUTES = 60

_last_signal_time = None


# =========================
# MAIN SIGNAL FUNCTION
# =========================
def trigger_retrain_signal(reason: str):

    global _last_signal_time

    now = datetime.utcnow()

    # -------------------------
    # Cooldown check
    # -------------------------
    if _last_signal_time is not None:
        diff = now - _last_signal_time

        if diff < timedelta(minutes=COOLDOWN_MINUTES):
            print("⛔ Retrain signal skipped (cooldown active)")
            return

    # -------------------------
    # SIGNAL (NO TRAINING!)
    # -------------------------
    print("\n🚨 RETRAIN SIGNAL")
    print(f"Reason: {reason}")
    print(f"Time: {now}")
    print("Action required: Review model before retraining.")

    # -------------------------
    # METRIC
    # -------------------------
    RETRAIN_SIGNAL_COUNTER.labels(reason=reason).inc()

    # -------------------------
    # STATE UPDATE
    # -------------------------
    _last_signal_time = now