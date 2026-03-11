from src.metrics.regression_metrics import evaluate_all_metrics


def evaluate_model(y_true, y_pred):

    metrics = evaluate_all_metrics(y_true, y_pred)

    return metrics