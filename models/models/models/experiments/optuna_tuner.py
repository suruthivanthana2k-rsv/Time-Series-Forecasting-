import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.lstm_with_attention import LSTMWithAttention
from utils.dataset import SlidingWindowMultivariate
from utils.metrics import mae
from sklearn.preprocessing import StandardScaler
import pandas as pd

def objective(trial, train_df, val_df, cfg):
    # Suggest hyperparams
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.05, 0.4)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    input_len = cfg['input_len']
    output_len = cfg['output_len']

    # Build data
    scaler = StandardScaler().fit(train_df.values)
    train_ds = SlidingWindowMultivariate(train_df, input_len=input_len, output_len=output_len, scaler=scaler)
    val_ds = SlidingWindowMultivariate(val_df, input_len=input_len, output_len=output_len, scaler=scaler)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMWithAttention(n_features=train_df.shape[1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_len=output_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()

    # short-run training loop (few epochs to evaluate)
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            preds, _ = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()

    # validation
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            p, _ = model(x)
            preds_list.append(p.cpu().numpy())
            trues_list.append(y.numpy())
    preds = np.vstack(preds_list)
    trues = np.vstack(trues_list)
    # aggregate MAE on validation set
    val_mae = mae(trues.flatten(), preds.flatten())
    return val_mae

def run_optuna(train_df, val_df, cfg, n_trials=30):
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, train_df, val_df, cfg)
    study.optimize(func, n_trials=n_trials)
    return study
