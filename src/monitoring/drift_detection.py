import polars as pl
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ✅ UPDATED (manual signal, NOT retrain)
from src.monitoring.retrain_trigger import trigger_retrain_signal

# Prometheus
from prometheus_client import Gauge

# =========================
# METRICS
# =========================
DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift score per state",
    ["state"]
)

DRIFT_DETECTED = Gauge(
    "data_drift_detected",
    "Drift detected flag (0/1)",
    ["state"]
)

# =========================
# CONFIG
# =========================
DRIFT_THRESHOLD = 0.5


# =========================
# MAIN FUNCTION
# =========================
def run_drift_detection(state: str, reference_df, current_df):

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_df.to_pandas(),
        current_data=current_df.to_pandas()
    )

    # -------------------------
    # SAVE REPORT
    # -------------------------
    output_dir = Path("reports/drift")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"drift_{state}.html"
    report.save_html(str(report_path))

    print(f"📄 Drift report saved: {report_path}")

    # -------------------------
    # EXTRACT DRIFT SCORE
    # -------------------------
    result = report.as_dict()

    drift_score = result["metrics"][0]["result"]["dataset_drift"]

    print(f"📊 Drift score ({state}): {drift_score}")

    # -------------------------
    # PROMETHEUS METRICS
    # -------------------------
    DRIFT_SCORE.labels(state=state).set(float(drift_score))

    # -------------------------
    # DECISION LOGIC
    # -------------------------
    if drift_score > DRIFT_THRESHOLD:
        print(f"⚠️ Drift detected for {state}")

        DRIFT_DETECTED.labels(state=state).set(1)

        # 🚨 ONLY SIGNAL (NO AUTO RETRAIN)
        trigger_retrain_signal(reason=f"drift_detected_{state}")

    else:
        DRIFT_DETECTED.labels(state=state).set(0)

    return report_path