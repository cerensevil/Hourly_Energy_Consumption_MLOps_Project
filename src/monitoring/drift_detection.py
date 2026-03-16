import polars as pl
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift_detection(state: str, reference_df, current_df):

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_df.to_pandas(),
        current_data=current_df.to_pandas()
    )

    output_dir = Path("reports/drift")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"drift_{state}.html"

    report.save_html(str(report_path))

    print(f"Drift report saved: {report_path}")

    return report_path