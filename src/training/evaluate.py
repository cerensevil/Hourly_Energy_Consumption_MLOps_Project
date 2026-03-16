import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.metrics.cost_weighted_error import cost_weighted_error


def evaluate_regression(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    cwe = cost_weighted_error(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "cwe": float(cwe),
    }