# ml-stock-predictor
## Overview
This project builds a machine learning pipeline to predict short-term stock price direction using historical market data and technical indicators. The model explores how prediction horizon and asset type impact performance in financial time series.

## Features
- Data collection using yfinance
- Feature engineering (RSI, MACD, momentum, volatility, moving averages)
- Time-series validation (per-ticker) to prevent data leakage
- Ticker-aware modeling using one-hot encoding
- Models: Random Forest, Logistic Regression

## Results
- Used a 5-day prediction hoprizon to better capture trend-based behavior
- Achived up to **61% accuracy on certain equities (e.g., NVDA)**
- Observed significant variation in performance across assets:
     - Strong results on trend-driven stocks (NVDA, GOOG)
     - Weak performance on efficient assets (SPY)
- Findings highlight that predictive signals differ across equities and are not universally       transferable

## Setup
```bash
git clone https://github.com/michaeldao06/ml-stock-predictor.git
cd ml-stock-predictor

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
'''

## How to Run
1. Run data download:
   python data_download.py

2. Generate features:
   python feature_engineering.py

3. Train models:
   python train_model.py

## Future Improvements
- Per-ticker modeling training and comparison
- Backtesting trading strategy based on predictions
- Advanced feature engineering (lag features, news indicators)
- Experimentations with different prediction horizons
