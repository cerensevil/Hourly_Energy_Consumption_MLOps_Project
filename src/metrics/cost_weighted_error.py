import numpy as np


def cost_weighted_error(
    y_true,
    y_pred,
    under_penalty: float = 3.0,
    over_penalty: float = 1.0,
):
    """
    Energy forecasting için cost-aware error metric.

    Underprediction daha pahalıdır.

    error = y_true - y_pred

    error > 0  -> underprediction
    error < 0  -> overprediction
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_true - y_pred

    costs = np.where(
        errors > 0,
        errors * under_penalty,
        np.abs(errors) * over_penalty
    )

    return float(np.mean(costs))