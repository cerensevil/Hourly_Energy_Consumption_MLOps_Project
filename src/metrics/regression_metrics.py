import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================================
# Core Metrics
# ================================
def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def peak_rmse(y_true, y_pred, percentile=90):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold

    if mask.sum() == 0:
        return None

    return float(rmse(y_true[mask], y_pred[mask]))


# ================================
# Error Decomposition (NEW 🔥)
# ================================
def error_stats(y_true, y_pred):
    """
    Returns detailed error statistics for monitoring & business metrics
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_true - y_pred

    return {
        "mean_error": float(np.mean(errors)),
        "mean_abs_error": float(np.mean(np.abs(errors))),
        "mean_squared_error": float(np.mean(errors ** 2)),
        "underprediction_rate": float(np.mean(errors > 0)),
        "overprediction_rate": float(np.mean(errors < 0)),
    }


# ================================
# Combined Evaluation
# ================================
def evaluate_all_metrics(y_true, y_pred):
    """
    Full evaluation bundle (training + monitoring compatible)
    """

    results = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "peak_rmse": peak_rmse(y_true, y_pred),
    }

    # add error stats
    results.update(error_stats(y_true, y_pred))

    return results