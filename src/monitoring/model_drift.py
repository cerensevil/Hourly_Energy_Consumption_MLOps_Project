import json
from pathlib import Path


def detect_model_drift(state, metrics, baseline_rmse):

    rmse = metrics["rmse"]
    mae = metrics["mae"]
    peak_rmse = metrics["peak_rmse"]

    drift_flags = []

    # Rule 1 — Model baseline'a yaklaşıyor
    if rmse >= 0.9 * baseline_rmse:
        drift_flags.append("baseline_convergence")

    # Rule 2 — Peak error çok yüksek
    if peak_rmse > 2 * rmse:
        drift_flags.append("high_peak_error")

    drift_detected = len(drift_flags) > 0

    return {
        "state": state,
        "drift_detected": drift_detected,
        "flags": drift_flags,
        "rmse": rmse,
        "baseline_rmse": baseline_rmse
    }


def save_drift_report(results):

    output_dir = Path("reports/model_drift")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "model_drift_report.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Model drift report saved: {output_file}")