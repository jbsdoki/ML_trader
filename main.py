from dotenv import load_dotenv
load_dotenv()

from yfinance import Ticker  #Yahoo Finance API 
from finnhub import Finnhub  #Finnhub.ai
import praw                  #Reddit API
import pandas as pd
import numpy as np



from tools import annualized_std_dev



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
    ann_std_dev = annualized_std_dev(data, 'Close')
    print(f"{symbol} annualized std dev: {ann_std_dev}")

# You can also specify additional parameters
# historical_data = ticker.history(start="2023-01-01", end="2024-12-31", interval="1d")


#alpha_vantage requires api key
#finnhub-python requires api key
#pandas-datareader (NO API KEY NEEDED) returns global data