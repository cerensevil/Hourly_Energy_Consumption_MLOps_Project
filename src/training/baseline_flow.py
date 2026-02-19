from prefect import flow
from src.training.train_baseline import train_baseline


@flow(name="baseline-flow")
def baseline_flow():
    data_path = "data/processed/AEP_hourly_processed.parquet"
    target_col = "target"

    result = train_baseline(data_path, target_col)
    print(result)


if __name__ == "__main__":
    baseline_flow()
