import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def generate_trading_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0
    signals['SMA_20'] = data['Close'].rolling(window=20).mean()
    signals['SMA_50'] = data['Close'].rolling(window=50).mean()

    signals['Signal'][20:] = np.where(
        signals['SMA_20'][20:] > signals['SMA_50'][20:], 1, 0
    )
    signals['Position'] = signals['Signal'].diff()
    return signals

def portfolio_performance(data, signals, initial_capital=10000):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Stock'] = 100 * signals['Signal']

    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()

    portfolio['Holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    portfolio['Cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['Total'] = portfolio['Cash'] + portfolio['Holdings']
    portfolio['Returns'] = portfolio['Total'].pct_change()

    return portfolio
