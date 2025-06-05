import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import timedelta

ALPHA_VANTAGE_API_KEY = 'LWTF8BX68NO5LQCK'  # Your working key

def fetch_alpha_vantage_daily(symbol):
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}"
        "&outputsize=full"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    r = requests.get(url)
    data = r.json()
    if "Time Series (Daily)" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff().dropna()
    if len(delta) < period:
        return 50
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    last_rs = rs.iloc[-1]
    if np.isnan(last_rs) or last_rs == 0:
        return 50
    return 100 - (100 / (1 + last_rs))

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("🏦 AI Stock Price Prediction System (Alpha Vantage Edition)")
st.sidebar.title("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").strip().upper()
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
prediction_days = st.sidebar.slider("Prediction Days", 1, 60, 30)

period_days = {
    "1mo": 22,
    "3mo": 66,
    "6mo": 132,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
    "max": 5000
}

if st.sidebar.button("Analyze Stock"):
    with st.spinner('Fetching data from Alpha Vantage...'):
        df = fetch_alpha_vantage_daily(ticker)
    if df.empty or len(df) < 30:
        st.error("⚠️ No data found for this ticker and period. Try another symbol or check your API key/usage limit.")
    else:
        # Filter by period
        df = df.tail(period_days[period])

        st.subheader(f"📊 {ticker} Stock Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${df['Close'][-1]:.2f}")
        with col2:
            if len(df) > 1:
                change = df['Close'][-1] - df['Close'][-2]
                pct = (change / df['Close'][-2]) * 100
                st.metric("Daily Change", f"${change:.2f}", f"{pct:.2f}%")
            else:
                st.metric("Daily Change", "N/A")
        with col3:
            st.metric("Volume", f"{df['Volume'][-1]:,.0f}")

        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='purple')))
        fig.update_layout(
            title="Price Chart with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prediction logic
        valid_data = df['Close'].dropna()
        if len(valid_data) >= 30:
            recent_trend = np.polyfit(range(30), valid_data.tail(30), 1)[0]
        else:
            recent_trend = 0

        future_dates = pd.date_range(
            start=valid_data.index[-1] + timedelta(days=1),
            periods=prediction_days
        )

        base_price = valid_data[-1]
        volatility = valid_data.pct_change().std()
        noise = np.random.normal(0, volatility * base_price, prediction_days)
        trend_predictions = base_price + (np.arange(prediction_days) * recent_trend) + noise

        # Prediction display
        st.subheader("🔮 Price Predictions")
        pred_df = pd.DataFrame({
            'Date': future_dates.date,
            'Predicted Price': trend_predictions
        }).set_index('Date')
        st.dataframe(pred_df.style.format({"Predicted Price": "${:.2f}"}))

        # Prediction chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=valid_data.index[-60:],
            y=valid_data[-60:],
            mode='lines',
            name='Historical Prices',
            line=dict(color='#1f77b4')
        ))
        fig2.add_trace(go.Scatter(
            x=future_dates,
            y=trend_predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(dash='dot', color='#ff7f0e')
        ))
        fig2.update_layout(
            title=f"{prediction_days}-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Trading signals
        ma_signal = "BUY" if df['SMA_20'][-1] > df['SMA_50'][-1] else "SELL"
        rsi_value = calculate_rsi(valid_data)
        rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        trend_signal = "Strong Buy" if trend_predictions[-1] > base_price * 1.05 else (
            "Sell" if trend_predictions[-1] < base_price * 0.95 else "Hold")

        st.subheader("🎯 Trading Signals")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MA Signal", ma_signal, delta=f"SMA20: {df['SMA_20'][-1]:.2f}")
        with col2:
            st.metric("RSI", f"{rsi_value:.1f} ({rsi_signal})", delta="14-day")
        with col3:
            st.metric("AI Trend", trend_signal, delta=f"{(trend_predictions[-1]/base_price-1)*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 AI Stock Predictor**")
st.sidebar.markdown("Powered by Alpha Vantage")
