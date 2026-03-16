import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.config.state_mapping import STATE_MAP
from src.models.lstm_model import LSTMForecast
from src.training.evaluate import evaluate_regression


FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "year",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "target_lag_1",
    "target_lag_24",
    "target_roll_mean_24",
    "target_roll_std_24"
]

WINDOW = 24
HORIZON = 24

MAX_SAMPLES = 30000   # ⚡ training hızlandırma


def create_sequences(X, y, states):

    xs = []
    ys = []
    ss = []

    for i in range(len(X) - WINDOW - HORIZON):

        xs.append(X[i:i+WINDOW])
        ys.append(y[i+WINDOW:i+WINDOW+HORIZON])
        ss.append(states[i+WINDOW])

    return (
        torch.tensor(np.array(xs)),
        torch.tensor(np.array(ys)),
        torch.tensor(np.array(ss))
    )


def train_lstm(data_path: str, target_col: str):

    processed_dir = Path("data/processed")

    dfs = []

    # ------------------------------------------------
    # GLOBAL DATASET
    # ------------------------------------------------
    for file in processed_dir.glob("*_processed.parquet"):

        df = pd.read_parquet(file)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if "Datetime" in df.columns:
        df = df.drop(columns=["Datetime"])

    df["state_id"] = df["state"].map(STATE_MAP)

    df = df[df["year"] < 2018]

    X = df[FEATURE_COLUMNS].values
    y = df[target_col].values
    states = df["state_id"].values

    # ------------------------------------------------
    # sequence oluştur
    # ------------------------------------------------

    X_seq, y_seq, state_seq = create_sequences(X, y, states)

    # ------------------------------------------------
    # ⚡ RANDOM SAMPLING (training hızlandırma)
    # ------------------------------------------------

    if len(X_seq) > MAX_SAMPLES:

        idx = np.random.choice(len(X_seq), MAX_SAMPLES, replace=False)

        X_seq = X_seq[idx]
        y_seq = y_seq[idx]
        state_seq = state_seq[idx]

    # ------------------------------------------------
    # train / test split
    # ------------------------------------------------

    train_X = X_seq[:-24]
    test_X = X_seq[-24:]

    train_y = y_seq[:-24]
    test_y = y_seq[-24:]

    train_states = state_seq[:-24]
    test_states = state_seq[-24:]

    dataset = TensorDataset(
        train_X.float(),
        train_states.long(),
        train_y.float()
    )

    # ⚡ batch büyüt
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = LSTMForecast(
        input_size=len(FEATURE_COLUMNS),
        horizon=HORIZON
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()

    # ⚡ epoch azalt
    for epoch in range(2):

        for xb, state_b, yb in loader:

            preds = model(xb, state_b)

            loss = loss_fn(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():

        preds = model(
            test_X.float(),
            test_states.long()
        ).numpy()

    preds_flat = preds.flatten()
    y_flat = test_y.numpy().flatten()

    metrics = evaluate_regression(y_flat, preds_flat)

    return {
        "model": model,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "cwe": metrics["cwe"]
    }