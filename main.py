from yfinance import Ticker
import pandas as pd
import numpy as np



ticker = Ticker("SPY")
chains = ticker.option_chain()
print(chains)

# Get daily data for a specific year
data_2023 = ticker.history(start="2023-01-01", end="2023-12-31")

# Get hourly data for the last 6 months
recent_data = ticker.history(period="6mo", interval="1h")

# Get data for multiple tickers
tickers = ["AAPL", "GOOGL", "MSFT"]
for symbol in tickers:
    data = Ticker(symbol).history(start="2023-01-01")
    print(f"{symbol}: {data.shape[0]} days of data")
    close = data['Close']
    returns = close.pct_change()
    std_dev = returns.std()
    print(f"{symbol} std dev: {std_dev}")
    ann_std_dev = std_dev * np.sqrt(252)
    print(f"{symbol} annualized std dev: {ann_std_dev}")

# You can also specify additional parameters
# historical_data = ticker.history(start="2023-01-01", end="2024-12-31", interval="1d")


#alpha_vantage requires api key
#finnhub-python requires api key
#pandas-datareader (NO API KEY NEEDED) returns global data