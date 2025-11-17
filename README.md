# Stock Price Predictor (Linear Regression)

This project predicts the next day's stock closing price using a simple
**Linear Regression** model trained on historical stock data.

## 🚀 Features
- Fetches historical stock data using `yfinance`
- Trains a linear regression model
- Evaluates model accuracy with MSE
- Predicts the next day's closing price

## 📦 Requirements
Install the dependencies:

```bash
pip install yfinance pandas scikit-learn
```

## ▶️ Run the predictor

```bash
python predictor.py
```

## 🔧 Change the stock
Open `predictor.py` and modify this line:

```python
ticker = "AAPL"
```

Replace `"AAPL"` with any valid stock symbol.
