# src/models/xgboost_model.py

import xgboost as xgb
import pandas as pd
from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, target_col: str):
        super().__init__(target_col)
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def fit(self, df: pd.DataFrame):

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):

        return self.model.predict(X)
