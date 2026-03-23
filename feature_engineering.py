import pandas as pd
import sys
import os

# ticker = input("Enter ticker to engineer features for: ").upper() #Maybe add back later
files = [
    f for f in os.listdir("data") 
    if f.endswith(".csv") 
    and not f.endswith("_features.csv") #Makes sure featured datasets aren't included
    and "[" not in f    #Makes sure the multi-index dataframe isn't included
]
if not files:
    print(f"No datasets found. Run data_download.py first.", file=sys.stderr)
    sys.exit(1)

for file in files:
    ticker = file.replace(".csv", "")
    file_path = f"data/{file}"
    try:
        data = pd.read_csv(file_path) #Attempt to lead csv file file into a pandas dataframe
    except FileNotFoundError:
        print(f"Dataset missing for {ticker}", file=sys.stderr)
        continue
    #Process data into numerics
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]   #List of columns that should contain numbers
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")   #Convert columns into numeric types

    #Daily Returns, (today_close/yesterday_close)-1
    data["Return"] = data["Close"].pct_change() #Calculates percentage change from previous day

    #Moving Averages for 5 and 10 days
    data["MA5"] = data["Close"].rolling(window=5).mean() #Short-term trends
    data["MA10"] = data["Close"].rolling(window=10).mean() #Longer trend signal

    #Volatility
    data["Volatility"] = data["Return"].rolling(window=5).std() #Standard deviation of returns over 5 days.

    #Momentum
    data["Momentum"] = data["Close"] - data["Close"].shift(5) # How much price has changed over last 5 days

    #RSI (Relative Strength Index), 0 -> very oversold, 100 -> very overbought
    delta = data["Close"].diff()    #Computes difference between today's close and yesterday's close

    gain = delta.clip(lower=0)  #Keep only positive changes and set negative changes=0, isolate upward price movement.
    loss = -delta.clip(upper=0) #Keep only negative changes and converts them to postive numbers

    avg_gain = gain.rolling(window=14).mean()   #ave gain over past 14 days
    avg_loss = loss.rolling(window=14).mean()   #average loss over past 14 days

    rs = avg_gain/avg_loss #Relative Strength
    data["RSI"] = 100 - (100 / (1 + rs)) 

    #MACD (Moving Average Convergence Divergence), measures trend strength using exponential moving averages
    ema12 = data["Close"].ewm(span=12, adjust=False).mean() #12-day EMA, short-term trend indicator
    ema26 = data["Close"].ewm(span=26, adjust=False).mean() #26-day moving average, long-term trend indicator
    data["MACD"] = ema12-ema26 #Positive -> Bullish Movement, Negative -> Bearish Movement

    #Price vs Moving Average
    data["Price_vs_MA10"] = data["Close"] / data["MA10"]

    #Momentum Return
    data["Momentum_Return"] = data["Return"].rolling(window=5).sum() #Returns over 5 days

    #Trend
    data["Trend"] = data["MA5"] / data["MA10"]

    #Distance from MA10
    data["Distance_MA10"] = data["Close"] / data['MA10'] - 1

    #Distance from MA20
    data["Distance_MA20"] = data["Close"] / data["Close"].rolling(20).mean() - 1

    #Target Variable: UP or DOWN tomorrow
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int) #tomorrow_close > today's close,
    # 1 = stock goes UP tomorrow
    # 0 = stock goes DOWN tomorrow
    #This is the goal of what the ML model will predict

    # Remove rows with NaN values created by rolling indicators
    data = data.dropna()

    #Save the engineered dataset
    data.to_csv(f"data/{ticker}_features.csv", index=False)
    print(f"Feature engineering for {ticker} complete.")

print("Feature engineering completed for all stocks.")
print(data.head())
