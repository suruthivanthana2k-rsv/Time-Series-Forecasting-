import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from models.lstm_with_attention import LSTMWithAttention
from models.lstm_baseline import LSTMForecast
from utils.dataset import SlidingWindowMultivariate, train_val_test_split
from baselines.sarimax_baseline import sarimax_train_forecast
from utils.metrics import rmse, mae, mape
import pandas as pd
import os


def train_and_save(cfg, df, outputs_dir='outputs'):
    train_df, val_df, test_df = train_val_test_split(df, train_ratio=cfg['train_ratio'], val_ratio=cfg['val_ratio'])
    scaler = StandardScaler().fit(train_df.values)
    train_ds = SlidingWindowMultivariate(train_df, input_len=cfg['input_len'], output_len=cfg['output_len'], scaler=scaler)
    val_ds = SlidingWindowMultivariate(val_df, input_len=cfg['input_len'], output_len=cfg['output_len'], scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attn_model = LSTMWithAttention(n_features=train_df.shape[1], hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], output_len=cfg['output_len']).to(device)
    lstm_model = LSTMForecast(n_features=train_df.shape[1], hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], output_len=cfg['output_len']).to(device)

    opt_attn = torch.optim.Adam(attn_model.parameters(), lr=cfg['lr'])
    opt_lstm = torch.optim.Adam(lstm_model.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.L1Loss()

    os.makedirs(outputs_dir, exist_ok=True)

    best_val = 1e9
    for epoch in range(cfg['epochs']):
        attn_model.train()
        lstm_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            # attention model
            opt_attn.zero_grad()
            p_attn, _ = attn_model(x, y_teacher=None, teacher_forcing_ratio=cfg.get('tf', 0.0))
            loss_attn = loss_fn(p_attn, y)
            loss_attn.backward()
            opt_attn.step()
            # baseline lstm
            opt_lstm.zero_grad()
            p_lstm = lstm_model(x)
            loss_lstm = loss_fn(p_lstm, y)
            loss_lstm.backward()
            opt_lstm.step()

        # validation
        attn_model.eval(); lstm_model.eval()
        vpreds = []
        vtrues = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                p_attn, _ = attn_model(x)
                vpreds.append(p_attn.cpu().numpy())
                vtrues.append(y.cpu().numpy())
        vpreds = np.vstack(vpreds); vtrues = np.vstack(vtrues)
        val_loss = mae(vtrues.flatten(), vpreds.flatten())
        print(f"Epoch {epoch+1} val MAE: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(attn_model.state_dict(), os.path.join(outputs_dir, 'attn_best.pt'))
            torch.save(lstm_model.state_dict(), os.path.join(outputs_dir, 'lstm_best.pt'))

    # Train SARIMAX on full train+val and forecast test as baseline separately in evaluation step
    return scaler



