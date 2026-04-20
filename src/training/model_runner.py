from src.training.train_baseline import train_baseline
from src.training.train_xgboost import train_xgboost

# 🔥 LSTM optional import
try:
    from src.training.train_lstm import train_lstm
    LSTM_AVAILABLE = True
except ImportError:
    print("⚠️ LSTM disabled (torch not installed)")
    LSTM_AVAILABLE = False


def run_models(data_path, target_col):

    results = {}

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------
    results["baseline"] = train_baseline(data_path, target_col)

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------
    results["xgboost"] = train_xgboost(data_path, target_col)

    # --------------------------------------------------
    # LSTM (optional)
    # --------------------------------------------------
    if LSTM_AVAILABLE:
        try:
            results["lstm"] = train_lstm(data_path, target_col)
        except Exception as e:
            print(f"⚠️ LSTM failed during training: {e}")

    return results