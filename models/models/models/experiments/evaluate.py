import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import SlidingWindowMultivariate, train_val_test_split
from utils.metrics import rmse, mae, mape
from baselines.sarimax_baseline import sarimax_train_forecast
from models.lstm_with_attention import LSTMWithAttention
from models.lstm_baseline import LSTMForecast
from sklearn.preprocessing import StandardScaler
import pandas as pd


def evaluate_models(cfg, df, scaler, outputs_dir='outputs'):
    train_df, val_df, test_df = train_val_test_split(df, train_ratio=cfg['train_ratio'], val_ratio=cfg['val_ratio'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_ds = SlidingWindowMultivariate(test_df, input_len=cfg['input_len'], output_len=cfg['output_len'], scaler=scaler)
    loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

    # Load models
    attn = LSTMWithAttention(n_features=df.shape[1], hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], output_len=cfg['output_len']).to(device)
    lstm = LSTMForecast(n_features=df.shape[1], hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], output_len=cfg['output_len']).to(device)
    attn.load_state_dict(torch.load(f"{outputs_dir}/attn_best.pt", map_location=device))
    lstm.load_state_dict(torch.load(f"{outputs_dir}/lstm_best.pt", map_location=device))
    attn.eval(); lstm.eval()

    preds_attn = []
    preds_lstm = []
    trues = []
    attn_maps = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p_attn, attn_map = attn(x)
            p_lstm = lstm(x)
            preds_attn.append(p_attn.cpu().numpy())
            preds_lstm.append(p_lstm.cpu().numpy())
            trues.append(y.numpy())
            if attn_map is not None:
                attn_maps.append(attn_map)

    preds_attn = np.vstack(preds_attn); preds_lstm = np.vstack(preds_lstm); trues = np.vstack(trues)

    # Compute metrics per model (flatten predictions vs flattened true across all test windows)
    metrics = {
        'attn': {'RMSE': rmse(trues.flatten(), preds_attn.flatten()), 'MAE': mae(trues.flatten(), preds_attn.flatten()), 'MAPE': mape(trues.flatten(), preds_attn.flatten())},
        'lstm': {'RMSE': rmse(trues.flatten(), preds_lstm.flatten()), 'MAE': mae(trues.flatten(), preds_lstm.flatten()), 'MAPE': mape(trues.flatten(), preds_lstm.flatten())}
    }

    # SARIMAX baseline: fit on train+val, forecast length = len(test_df)
    sarimax_preds = sarimax_train_forecast(pd.concat([train_df, val_df]), test_df)
    # Create rolling windows aligned for comparison: take first output_len predictions from sarimax predictions
    # For fairness, compute metrics using the first output_len predictions per window (coarse approximation)
    # Here we compute simple metrics on test series directly
    metrics['sarimax'] = {'RMSE': rmse(test_df['y'].values, sarimax_preds.values), 'MAE': mae(test_df['y'].values, sarimax_preds.values), 'MAPE': mape(test_df['y'].values, sarimax_preds.values)}

    return metrics, attn_maps

