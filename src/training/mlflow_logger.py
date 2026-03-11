import mlflow


def start_run(model_type, state, dataset_version):

    mlflow.set_experiment("energy_forecasting")

    return mlflow.start_run(
        run_name=f"{state}_{model_type}",
        tags={
            "state": state,
            "model_type": model_type,
            "dataset_version": dataset_version,
        },
    )


def log_metrics(metrics):

    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_params(params):

    for k, v in params.items():
        mlflow.log_param(k, v)