import torch
import torch.nn as nn


class LSTMForecast(nn.Module):

    def __init__(self, input_size, hidden_size=64, horizon=24):

        super().__init__()

        self.horizon = horizon

        # state embedding
        self.state_embedding = nn.Embedding(12, 4)

        self.lstm = nn.LSTM(
            input_size=input_size + 4,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x, state):

        state_emb = self.state_embedding(state)

        state_emb = state_emb.unsqueeze(1).repeat(1, x.size(1), 1)

        x = torch.cat([x, state_emb], dim=2)

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.fc(out)