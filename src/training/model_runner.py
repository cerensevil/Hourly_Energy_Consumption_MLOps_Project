from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost
from src.training.train_lstm import train_lstm


def run_models(data_path, target_col):

    results = {}

    results["baseline"] = train_baseline(data_path, target_col)

    results["xgboost"] = train_xgboost(data_path, target_col)

    results["lstm"] = train_lstm(data_path, target_col)

    return results