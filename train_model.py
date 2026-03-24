import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Load feature datasets
files = [
    f for f in os.listdir("data") 
    if f.endswith("_features.csv")
]

if not files:
    print("No feature datasets found. Run feature_engineering.py first", file=sys.stderr)
    sys.exit(1)

frames = []

for file in files:
    file_path = f"data/{file}"
    df = pd.read_csv(file_path)
    #Track which stock each row belongs to
    ticker = file.replace("_features.csv", "")
    df["Ticker"] = ticker
    frames.append(df)

#Combine datasets
#PUSH
data = pd.concat(frames, ignore_index=True)
print(f"Total rows: {len(data)}")

#Features used for prediction
features = [
    "Return",
    "MA5",
    "MA10",
    "Volatility",
    "Momentum",
    "RSI",
    "MACD",
    "Price_vs_MA10",
    "Momentum_Return",
    "Trend",
    "Distance_MA10",
    "Distance_MA20"
]

X = data[features]
X = X.shift(1) #Shift one day forward to prevent model from having future data
y = data["Target"]

# remove rows where shifting created NaNs
dataset = pd.concat([X,y], axis=1).dropna()
dataset["Ticker"] = data.loc[dataset.index, "Ticker"]

train_parts = []
test_parts = []

for ticker in dataset["Ticker"].unique():
    ticker_data = dataset[dataset["Ticker"] == ticker].copy()
    split_idx = int(len(ticker_data) * 0.80)

    train_parts.append(ticker_data.iloc[:split_idx])
    test_parts.append(ticker_data.iloc[split_idx:])

train_data = pd.concat(train_parts, ignore_index=True)
test_data = pd.concat(test_parts, ignore_index=True)

X_train = pd.get_dummies(train_data[features + ["Ticker"]], columns=["Ticker"])
y_train = train_data["Target"]

X_test = pd.get_dummies(test_data[features + ["Ticker"]], columns=["Ticker"])
y_test = test_data["Target"]

# Align columns (VERY important)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

#--Random Forest Model--
forest_model = RandomForestClassifier(
    n_estimators=500,
    max_depth= 6,
    min_samples_leaf = 20,
    random_state=42
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
forest_accuracy = accuracy_score(y_test, forest_predictions)
#Results for the Random Forest Model
print("Random Forest Model trained successfully.")
print(f"Random Forest Accuracy: {forest_accuracy:.2f}")
forest_importances = pd.Series(forest_model.feature_importances_, index=X_train.columns)
print(forest_importances.sort_values(ascending=False))


#--Scaling for Logistic Regression--
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#--Logistic Regression Model--
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

log_predictions = log_model.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, log_predictions)
#Results for Regression Model
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

results = test_data.copy()
results["Predicted"] = forest_predictions

ticker_acc = results.groupby("Ticker").apply(
    lambda x: (x["Target"] == x["Predicted"]).mean()
)

print("\nAccuracy per ticker:")
print(ticker_acc)


print(len(data))
print(data["Target"].value_counts(normalize=True))
