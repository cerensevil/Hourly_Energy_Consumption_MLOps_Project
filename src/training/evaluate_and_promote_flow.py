from prefect import flow

from src.registry.promote_best import promote_best_model


@flow(name="evaluate-and-promote-flow")
def evaluate_and_promote_flow():

    promote_best_model("energy_forecast_model")
