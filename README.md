# ml-stock-predictor
## Overview
This project builds a machine learning pipeline to predict short-term stock price direction using historical market data and technical indicators.

## Features
- Data collection using yfinance
- Feature engineering (RSI, MACD, momentum, volatility, moving averages)
- Time-series validation to prevent data leakage
- Models: Random Forest, Logistic Regression

## Results
- Achieved ~51–52% accuracy depending on model used
- Performance varies across equities, highlighting market complexity

## How to Run
1. Run data download:
   python data_download.py

2. Generate features:
   python feature_engineering.py

3. Train models:
   python train_model.py

## Future Improvements
- Per-ticker modeling
- Backtesting trading strategy
- Feature selection optimization
