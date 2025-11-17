# ===============================
# Linear Regression Stock Predictor
# ===============================

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Load historical stock data ---
ticker = "AAPL"   # Change this to any stock symbol you want
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Use only the closing price
df = data[['Close']].copy()

# --- 2. Create a target column (next day's price) ---
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

# --- 3. Prepare data ---
X = df[['Close']].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- 4. Train Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. Evaluate ---
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"Model trained on {ticker}")
print("MSE:", mse)

# --- 6. Predict the next day's price ---
latest_price = df['Close'].iloc[-1]
predicted_next = model.predict([[latest_price]])[0]

print("Latest closing price:", latest_price)
print("Predicted next closing price:", predicted_next)
