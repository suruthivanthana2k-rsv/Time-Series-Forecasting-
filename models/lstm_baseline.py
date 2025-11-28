import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.1, output_len=10):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        # x: (B, T_in, n_features)
        out, _ = self.lstm(x)
        # take last step
        last = out[:, -1, :]
        out = self.proj(last)
        return out
