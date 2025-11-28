# Advanced Time Series Forecasting (Transformer / LSTM + Attention)

This repository contains a full pipeline: synthetic multivariate data generation, LSTM baseline, LSTM+Luong attention model, SARIMAX baseline, Optuna hyperparameter search, walk-forward-ready data utilities, training and evaluation.

Quickstart:
1. Create virtual env and install: `pip install -r requirements.txt`
2. Generate data: `python data/generate.py`
3. Run tuning (optuna): `python main.py --step tune`
4. Train best models: `python main.py --step train`
5. Evaluate: `python main.py --step eval`

Deliverables you must include when submitting to CJR:
- code (this repo)
- `report_template.md` filled with your experimental log and analysis
- saved models in `outputs/` and the final metrics printed in `main.py` output



