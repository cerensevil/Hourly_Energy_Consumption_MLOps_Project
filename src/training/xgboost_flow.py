from prefect import flow
from src.training.train_xgboost import train_xgboost


@flow(name="xgboost-flow")
def xgboost_flow():
    data_path = "data/processed/AEP_hourly_processed.parquet"
    target_col = "AEP"

    result = train_xgboost(data_path, target_col)
    print(result)


if __name__ == "__main__":
    xgboost_flow()
