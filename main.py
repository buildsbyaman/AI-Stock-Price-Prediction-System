import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker='AAPL', period='5y'):
        self.ticker = ticker
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None

    def fetch_data(self):
        self.data = yf.download(self.ticker, period=self.period)
        return self.data

    def calculate_indicators(self):
        df = self.data.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._rsi(df['Close'])
        df['MACD'] = self._macd(df['Close'])
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        return df.dropna()

    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _macd(self, prices):
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2

    def prepare_data(self, seq_len=60):
        enhanced_data = self.calculate_indicators()
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
        dataset = enhanced_data[features].values
        scaled_data = self.scaler.fit_transform(dataset)

        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i-seq_len:i])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train(self, epochs=50):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        return history, X_test, y_test

    def predict_future(self, days=30):
        enhanced_data = self.calculate_indicators()
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
        scaled_data = self.scaler.fit_transform(enhanced_data[features])

        last_sequence = scaled_data[-60:]
        predictions = []

        for _ in range(days):
            next_pred = self.model.predict(last_sequence.reshape(1, 60, -1), verbose=0)
            predictions.append(next_pred[0, 0])

            new_row = last_sequence[-1].copy()
            new_row[0] = next_pred[0, 0]
            last_sequence = np.vstack([last_sequence[1:], new_row])

        dummy = np.zeros((len(predictions), scaled_data.shape[1]))
        dummy[:, 0] = predictions
        actual_predictions = self.scaler.inverse_transform(dummy)[:, 0]
        return actual_predictions

if __name__ == "__main__":
    predictor = StockPredictor('AAPL', '5y')
    print("Fetching data...")
    data = predictor.fetch_data()
    print(f"Training model for {predictor.ticker}...")
    history, X_test, y_test = predictor.train(epochs=30)

    predictions = predictor.predict_future(30)
    print(f"Current price: ${data['Close'][-1]:.2f}")
    print(f"30-day prediction: ${predictions[-1]:.2f}")
