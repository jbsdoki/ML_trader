"""
Data ingestion: prices (yfinance, Alpaca), news (Finnhub, NewsAPI), Alpaca clock.
"""

from .alpaca_ingest import AlpacaIngestor, fetch_stock_bars, get_market_clock
from .finnhub_ingest import FinnhubIngestor, fetch_company_news
from .newsapi_ingest import NewsAPIIngestor, fetch_everything, fetch_for_symbol, fetch_top_headlines
from .yfinance_ingest import YFinanceIngestor, fetch_ohlcv

__all__ = [
    "AlpacaIngestor",
    "FinnhubIngestor",
    "NewsAPIIngestor",
    "YFinanceIngestor",
    "fetch_company_news",
    "fetch_everything",
    "fetch_for_symbol",
    "fetch_ohlcv",
    "fetch_stock_bars",
    "fetch_top_headlines",
    "get_market_clock",
]
