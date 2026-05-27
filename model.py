import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def fetch_data(ticker):
    df = yf.download(ticker, period="2y", auto_adjust=True)
    df = df[['Close']].copy()
    df.dropna(inplace=True)
    return df

def add_features(df):
    df['MA7']        = df['Close'].rolling(7).mean()
    df['MA21']       = df['Close'].rolling(21).mean()
    df['MA50']       = df['Close'].rolling(50).mean()
    df['Return']     = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(7).std()
    df['Target']     = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ['Close', 'MA7', 'MA21', 'MA50', 'Return', 'Volatility']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2  = r2_score(y_test, predictions)

    return model, X_test, y_test, predictions, mae, r2