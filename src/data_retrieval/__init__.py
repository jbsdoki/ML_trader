"""
Data ingestion: prices (Alpaca), news (Finnhub, NewsAPI), Alpaca clock.
"""

from .alpaca_ingest import AlpacaIngestor, fetch_stock_bars, get_market_clock
from .finnhub_ingest import FinnhubIngestor, fetch_company_news
from .newsapi_ingest import NewsAPIIngestor, fetch_everything, fetch_for_symbol, fetch_top_headlines

__all__ = [
    "AlpacaIngestor",
    "FinnhubIngestor",
    "NewsAPIIngestor",
    "fetch_company_news",
    "fetch_everything",
    "fetch_for_symbol",
    "fetch_stock_bars",
    "fetch_top_headlines",
    "get_market_clock",
]
