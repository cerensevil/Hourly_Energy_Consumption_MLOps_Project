import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def peak_rmse(y_true, y_pred, percentile=90):

    threshold = np.percentile(y_true, percentile)

    mask = y_true >= threshold

    if mask.sum() == 0:
        return None

    return rmse(y_true[mask], y_pred[mask])


def evaluate_all_metrics(y_true, y_pred):

    results = {
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
        "peak_rmse": peak_rmse(y_true, y_pred)
    }

    return results