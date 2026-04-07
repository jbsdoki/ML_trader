"""
Pull OHLCV (and related) market data via yfinance — no API key required.

**Why this file is yfinance-specific:** ``yfinance`` wraps Yahoo Finance endpoints and
returns a pandas DataFrame from ``Ticker.history()`` with Yahoo's column names
(``Open``, ``High``, …) and a DatetimeIndex. We lowercase columns and name the index
``timestamp`` so price bars align with :mod:`alpaca_ingest` for feature pipelines.

**Note:** yfinance is unofficial / scraper-based; field availability can change.

**References**

- Library docs: https://ranaroussi.github.io/yfinance/
- ``history()`` parameters: https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import yfinance as yf

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Yahoo's ``history()`` DataFrame: lowercase OHLCV columns, datetime index.

    Keeps extra columns (e.g. dividends) if ``actions=True`` was used — also lowercased.
    """
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index.name = "timestamp"
    return out


def fetch_ohlcv(
    symbol: str,
    *,
    period: str | None = None,
    interval: str = "1d",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    auto_adjust: bool = False,
    prepost: bool = False,
    actions: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV history for one symbol using ``yf.Ticker(symbol).history(...)``.

    Output is Yahoo-shaped first, then passed through :func:`_normalize_ohlcv_df` and
    tagged with a ``symbol`` column.

    Parameters
    ----------
    symbol
        Equity ticker, e.g. ``\"AAPL\"``.
    period
        yfinance period string (e.g. ``\"1mo\"``, ``\"6mo\"``, ``\"1y\"``).
        Mutually exclusive with ``start``/``end`` in yfinance; prefer one style.
    interval
        Bar size: ``1m``, ``2m``, ``5m``, ``15m``, ``30m``, ``60m``, ``90m``,
        ``1h``, ``1d``, ``5d``, ``1wk``, ``1mo``, ``3mo``.
    start, end
        Inclusive start / exclusive end (yfinance semantics) as strings or timestamps.
    auto_adjust
        If True, adjust OHLC for splits and dividends (close-only metrics differ).
    prepost
        Include pre/post market data when available.
    actions
        Include dividends and stock splits columns when True.

    Returns
    -------
    DataFrame
        Index: ``timestamp`` (DatetimeIndex). Columns include at least
        ``open``, ``high``, ``low``, ``close``, ``volume`` (lowercase).
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(
        period=period,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        prepost=prepost,
        actions=actions,
    )
    if hist is None or hist.empty:
        logger.warning("No yfinance rows returned for symbol=%s", symbol)
        return pd.DataFrame()

    out = _normalize_ohlcv_df(hist)
    out["symbol"] = symbol.upper()
    return out


def fetch_ohlcv_many(
    symbols: Iterable[str],
    *,
    period: str | None = None,
    interval: str = "1d",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    **kwargs: bool | str | None,
) -> pd.DataFrame:
    """
    Loop symbols and concatenate :func:`fetch_ohlcv` results (isolated failures per ticker).

    For one HTTP round-trip for many tickers, consider ``yfinance.download`` later;
    that returns a different column layout (possible MultiIndex) and needs its own normalizer.
    """
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = fetch_ohlcv(
                sym.strip().upper(),
                period=period,
                interval=interval,
                start=start,
                end=end,
                **kwargs,
            )
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.exception("yfinance failed for symbol=%s", sym)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0).sort_index()


class YFinanceIngestor:
    """Stores default ``interval`` / adjustment flags for repeated :func:`fetch_ohlcv` calls."""

    def __init__(
        self,
        *,
        interval: str = "1d",
        auto_adjust: bool = False,
        prepost: bool = False,
    ) -> None:
        """Defaults mirror ``Ticker.history()`` kwargs on each :meth:`ohlcv` call."""
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.prepost = prepost

    def ohlcv(
        self,
        symbol: str,
        *,
        period: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Instance version of :func:`fetch_ohlcv` using stored interval and flags."""
        return fetch_ohlcv(
            symbol,
            period=period,
            interval=self.interval,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            prepost=self.prepost,
        )
