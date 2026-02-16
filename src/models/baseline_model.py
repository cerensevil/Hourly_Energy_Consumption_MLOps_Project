# src/models/baseline_model.py

import pandas as pd
from src.models.base_model import BaseModel


class Baseline24hModel(BaseModel):

    def __init__(self, target_col: str):
        super().__init__(target_col)
        self.last_24 = None

    def fit(self, df: pd.DataFrame):

        if len(df) < 24:
            raise ValueError("Dataset must contain at least 24 rows.")

        self.last_24 = df[self.target_col].tail(24).values

    def predict(self):

        if self.last_24 is None:
            raise ValueError("Model not fitted yet.")

        return self.last_24.tolist()
