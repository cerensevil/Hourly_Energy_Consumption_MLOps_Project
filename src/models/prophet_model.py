# src/models/prophet_model.py

import pandas as pd
from prophet import Prophet
from src.models.base_model import BaseModel


class ProphetModel(BaseModel):

    def __init__(self, target_col: str):
        super().__init__(target_col)
        self.model = Prophet()

    def fit(self, df: pd.DataFrame):

        df_prophet = df.rename(
            columns={
                "Datetime": "ds",
                self.target_col: "y"
            }
        )

        self.model.fit(df_prophet[["ds", "y"]])

    def predict(self, future_df: pd.DataFrame):

        forecast = self.model.predict(future_df)

        return forecast["yhat"].values
