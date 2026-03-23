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
X = dataset[features]
y = dataset["Target"]

'''
data = data.dropna()
X = X.loc[data.index]
y = y.loc[data.index]
'''
# Split dataset into training and testing sets
split = int(len(X) * 0.80)
X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

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
forest_importances = pd.Series(forest_model.feature_importances_, index=features)
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


print(len(data))
print(data["Target"].value_counts(normalize=True))
