import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class SlidingWindowMultivariate(Dataset):
    def __init__(self, df: pd.DataFrame, input_len=60, output_len=10, target_col='y', scaler=None):
        self.df = df.copy()
        self.input_len = input_len
        self.output_len = output_len
        self.target_col = target_col
        self.cols = list(df.columns)
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df.values)
        else:
            self.scaler = scaler
        scaled = self.scaler.transform(self.df.values)
        self.scaled = pd.DataFrame(scaled, index=self.df.index, columns=self.cols)
        self.indices = []
        n = len(self.scaled)
        for i in range(self.input_len, n - self.output_len + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.scaled.iloc[i - self.input_len:i].values.astype('float32')  # (T_in, n_features)
        y = self.scaled.iloc[i:i + self.output_len][self.target_col].values.astype('float32')  # (T_out,)
        return x, y


def train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

