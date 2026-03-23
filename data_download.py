import yfinance as yf
import contextlib
import pandas as pd
import os
import io
import sys

# ticker = input("Enter a stock ticker:").upper() # Let user pick a stock ticker  #Maybe add back later

tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "SPY"]

def download_data(tickers):
    f = io.StringIO() #Creates a in-memory buffer that can capture printed text

    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):  #Redirect reroute stdout and stderr to f.
        data = yf.download(tickers, start="2015-01-01", end="2026-01-01") #download the selected ticker historical data
       
    if data.empty: #Checks if given ticker is valid
        print("Invalid Ticker", file=sys.stderr) #Since we capture yfinance's error messaged in f, now only our error message will print.
        exit()
    
    os.makedirs("data", exist_ok=True) #Checks if a data directory for the data exists, if not, create it.

    for ticker in tickers: #From the multi-stock dataframe, split it into individual datasets per ticker.
        df = data.xs(ticker, axis=1, level=1)
        df.to_csv(f"data/{ticker}.csv")
    return data

data = download_data(tickers)


print("Data downloaded successfully")
print(data.head())

