import argparse
import yaml
import pandas as pd
from data.generate import generate_complex_multivariate
from train import train_and_save
from evaluate import evaluate_models
from experiments.optuna_tuner import run_optuna


def load_config(path='configs.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['generate','tune','train','eval','all'], default='all')
    parser.add_argument('--config', default='configs.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.step in ('generate','all'):
        df = generate_complex_multivariate(n_steps=cfg['n_steps'], n_features=cfg['n_features'], out_path=cfg['data_path'])
    else:
        df = pd.read_csv(cfg['data_path'], index_col=0, parse_dates=True)

    if args.step in ('tune','all'):
        train_df, val_df, test_df = (df.iloc[:int(len(df)*cfg['train_ratio'])], df.iloc[int(len(df)*cfg['train_ratio']):int(len(df)*(cfg['train_ratio']+cfg['val_ratio']))], df.iloc[int(len(df)*(cfg['train_ratio']+cfg['val_ratio'])):])
        study = run_optuna(train_df, val_df, cfg, n_trials=cfg.get('n_trials',20))
        print('Best params:', study.best_params)

    if args.step in ('train','all'):
        scaler = train_and_save(cfg, df, outputs_dir=cfg['outputs_dir'])

    if args.step in ('eval','all'):
        metrics, attn_maps = evaluate_models(cfg, df, scaler, outputs_dir=cfg['outputs_dir'])
        print('Evaluation metrics:', metrics)

