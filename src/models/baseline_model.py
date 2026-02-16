class BaselineModel:

    def __init__(self, target_col: str):
        self.target_col = target_col

    def fit(self, df):
        # Baseline için training yok
        return self

    def forecast_next_24(self, df):
        # Son 24 saati al
        last_24 = df[self.target_col].tail(24).values

        # Baseline: dünkü 24 saati aynen tekrar et
        return last_24
