import os
import numpy as np
import pandas as pd
from numpy.random import default_rng

def generate_complex_multivariate(n_steps=12000, n_features=5, freq='H', seed=42, out_path='data/output.csv'):
    rng = default_rng(seed)
    idx = pd.date_range(start='2015-01-01', periods=n_steps, freq=freq)
    t = np.arange(n_steps)

    # Base target: trend + multiple seasonalities + regime shifts + filtered noise
    trend = 0.00025 * (t ** 1.15)
    daily = 3.0 * np.sin(2 * np.pi * t / 24 + 0.1)
    weekly = 1.2 * np.sin(2 * np.pi * t / (24 * 7) + 0.4)
    yearly = 0.6 * np.sin(2 * np.pi * t / 8766 + 0.7)
    amplitude = np.where(t < n_steps // 3, 1.0, np.where(t < 2 * n_steps // 3, 1.6, 0.7))

    wn = rng.standard_normal(n_steps)
    kernel = np.exp(-np.linspace(0, 6, 300))
    noise = np.convolve(wn, kernel, mode='same')

    y = amplitude * (trend + daily + weekly + yearly) + 0.9 * noise

    df = pd.DataFrame(index=idx)
    df['y'] = y

    # Create n_features - 1 exogenous features (total >=5)
    for i in range(n_features - 1):
        lagged = np.roll(y, i + 1) * (0.2 + 0.1 * rng.random())
        seasonal = 0.5 * np.sin(2 * np.pi * t / (24 * (i + 2)) + rng.random())
        ex = 0.4 * lagged + seasonal + 0.3 * rng.standard_normal(n_steps)
        df[f'x{i}'] = ex

    # Add a categorical-like indicator (one-hot-ish) for regime (converted to numeric)
    regimes = np.zeros(n_steps)
    regimes[n_steps // 3:2 * n_steps // 3] = 1
    df['regime'] = regimes

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path)
    print(f"Saved dataset to {out_path} with shape {df.shape}")
    return df

if __name__ == '__main__':
    generate_complex_multivariate()

