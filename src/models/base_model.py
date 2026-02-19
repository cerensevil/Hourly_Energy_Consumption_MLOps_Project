# src/models/base_model.py

from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """
    Tüm modeller için ortak arayüz.
    """

    def __init__(self, target_col: str):
        self.target_col = target_col

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self):
        pass
