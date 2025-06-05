import requests
import pandas as pd

API_KEY = "LWTF8BX68NO5LQCK"
symbol = "AAPL"

url = (
    "https://www.alphavantage.co/query"
    "?function=TIME_SERIES_DAILY"
    f"&symbol={symbol}"
    "&outputsize=compact"
    f"&apikey={API_KEY}"
)

response = requests.get(url)
data = response.json()

print("API Response:", data)  # Check raw response

if "Time Series (Daily)" in data:
    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    print("\nFirst 5 rows of data:")
    print(df.head())
else:
    print("Error:", data.get("Note", "Unknown error"))
