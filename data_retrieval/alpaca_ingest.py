"""
Alpaca: historical stock bars (market data) and read-only trading helpers (clock).

In the .env file the alpaca api key and secret key are:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY

And hardcoded in .env to use only paper trading 
    ALPACA_PAPER=true


**Why this file is Alpaca-specific:** The SDK returns ``BarSet`` objects (often exposed as a
pandas DataFrame with Alpaca's column names and a multi-index). We map that into lowercase
OHLCV-style columns plus ``timestamp`` / ``symbol`` so it lines up with ``yfinance_ingest``.

**Official format / API references**

- Stock bars (REST shape the SDK wraps): https://docs.alpaca.markets/reference/stockbars
- Market data overview: https://docs.alpaca.markets/docs/about-market-data-api
- Python SDK: https://github.com/alpacahq/alpaca-py

"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
except ImportError as e:  # pragma: no cover
    StockHistoricalDataClient = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _require_alpaca() -> None:
    """Fail fast with a clear message if ``alpaca-py`` is not installed."""
    if StockHistoricalDataClient is None:
        raise ImportError("alpaca-py is not installed. pip install alpaca-py") from _IMPORT_ERROR


def _alpaca_keys(
    api_key: str | None = None,
    secret_key: str | None = None,
) -> tuple[str, str]:
    """Read API key + secret from args or env (supports Alpaca's env var names)."""
    key = (api_key or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
    secret = (
        secret_key
        or os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
        or ""
    ).strip()
    if not key or not secret:
        raise ValueError(
            "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "(or APCA_API_KEY_ID and APCA_API_SECRET_KEY)."
        )
    return key, secret


def _paper_flag() -> bool:
    """Return True if ``ALPACA_PAPER`` env indicates paper trading (default: true)."""
    v = os.getenv("ALPACA_PAPER", "true").strip().lower()
    return v in ("1", "true", "yes", "on")


def timeframe_from_string(
    interval: Literal["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mo"] | str,
) -> TimeFrame:
    """
    Map our string intervals (similar to yfinance) to Alpaca's ``TimeFrame`` objects.

    Alpaca defines bar length via ``TimeFrame(amount, unit)`` — see their timeframe docs.
    """
    _require_alpaca()
    s = interval.strip().lower()
    mapping: dict[str, TimeFrame] = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "60m": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
        "1wk": TimeFrame(1, TimeFrameUnit.Week),
        "1w": TimeFrame(1, TimeFrameUnit.Week),
        "1mo": TimeFrame(1, TimeFrameUnit.Month),
    }
    if s not in mapping:
        raise ValueError(
            f"Unsupported interval={interval!r}. "
            f"Use one of: {', '.join(sorted(set(mapping.keys())))}"
        )
    return mapping[s]


def _normalize_bars_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Turn Alpaca's bar DataFrame (post-``reset_index``) into a consistent OHLCV table.

    Lowercases columns, parses ``timestamp`` to UTC, ensures ``symbol`` is present.
    """
    if df.empty:
        return df

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    # BarSet.df often uses multi-index (symbol, timestamp) -> columns after reset_index
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    elif out.index.name == "timestamp" or isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "symbol" not in out.columns:
        out["symbol"] = symbol.upper()
    else:
        out["symbol"] = out["symbol"].astype(str).str.upper()
    # Keep timestamp as a normal column; setting index.name to "timestamp" while a
    # timestamp column exists can make pandas operations ambiguous.
    out.index.name = None
    return out.sort_values("timestamp").reset_index(drop=True)


def fetch_stock_bars(
    symbol: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    *,
    interval: str = "1d",
    feed: str | None = None,
    api_key: str | None = None,
    secret_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV bars via Alpaca Market Data and return a normalized DataFrame.

    Wraps ``StockHistoricalDataClient.get_stock_bars`` -> ``BarSet.df``, then flattens
    the index and normalizes columns for downstream feature code.

    Parameters
    ----------
    symbol
        Ticker, e.g. ``\"AAPL\"``.
    start, end
        Interval in UTC (naive datetimes are treated as UTC by pandas/Alpaca client).
    interval
        ``1m``, ``5m``, ``15m``, ``30m``, ``1h``, ``1d``, ``1w``, ``1mo``.
    feed
        Optional Alpaca data feed (e.g. ``\"sip\"``, ``\"iex\"``). If ``None``, Alpaca uses your default.
    """
    _require_alpaca()
    key, sec = _alpaca_keys(api_key, secret_key)
    sym = symbol.strip().upper()
    tf = timeframe_from_string(interval)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    client = StockHistoricalDataClient(key, sec)
    req_kwargs: dict[str, Any] = {
        "symbol_or_symbols": sym,
        "timeframe": tf,
        "start": start_ts.to_pydatetime(),
        "end": end_ts.to_pydatetime(),
    }
    if feed:
        req_kwargs["feed"] = feed

    request = StockBarsRequest(**req_kwargs)
    try:
        bars = client.get_stock_bars(request)
    except Exception:
        logger.exception("Alpaca get_stock_bars failed symbol=%s", sym)
        raise

    df = getattr(bars, "df", None)
    if df is None or df.empty:
        logger.warning("Alpaca returned no bars for %s", sym)
        return pd.DataFrame()

    # Multi-index (symbol, timestamp) -> flat table
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()

    return _normalize_bars_df(df, sym)


def get_market_clock(
    *,
    api_key: str | None = None,
    secret_key: str | None = None,
    paper: bool | None = None,
) -> dict[str, Any]:
    """
    Ask Alpaca's *trading* API for the current market clock (session open/close).

    Uses ``TradingClient.get_clock()`` — same keys as market data, ``paper`` controls endpoint.
    Returns a JSON-friendly dict for logging or schedulers.
    """
    _require_alpaca()
    key, sec = _alpaca_keys(api_key, secret_key)
    use_paper = _paper_flag() if paper is None else paper
    client = TradingClient(key, sec, paper=use_paper)
    clock = client.get_clock()
    return {
        "is_open": clock.is_open,
        "timestamp": clock.timestamp.isoformat() if clock.timestamp else None,
        "next_open": clock.next_open.isoformat() if clock.next_open else None,
        "next_close": clock.next_close.isoformat() if clock.next_close else None,
    }


class AlpacaIngestor:
    """
    Holds Alpaca credentials and defaults so callers don't repeat key arguments.

    Thin facade over :func:`fetch_stock_bars` and :func:`get_market_clock`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
        default_interval: str = "1d",
    ) -> None:
        """Store optional keys, paper override for clock, and default bar interval."""
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.default_interval = default_interval

    def stock_bars(
        self,
        symbol: str,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        *,
        interval: str | None = None,
        feed: str | None = None,
    ) -> pd.DataFrame:
        """Instance version of :func:`fetch_stock_bars` using stored credentials."""
        return fetch_stock_bars(
            symbol,
            start,
            end,
            interval=interval or self.default_interval,
            feed=feed,
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

    def market_clock(self) -> dict[str, Any]:
        """Instance version of :func:`get_market_clock` using stored credentials."""
        return get_market_clock(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )
