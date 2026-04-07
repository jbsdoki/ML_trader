"""
Pull company news (and optional quote) via Finnhub REST API (finnhub-python).

Requires env var ``FINNHUB_API_KEY`` or an explicit API key passed to the client.

**Why this file is Finnhub-specific:** Responses are Finnhub's JSON field names
(``headline``, ``summary``, Unix ``datetime`` on news items, etc.). We map those into a
stable pandas schema so sentiment and storage layers can treat Finnhub like other news
sources (compare ``newsapi_ingest`` — same *target* columns where possible).

**Official format / API references**

- Company news endpoint & fields: https://finnhub.io/docs/api/company-news
- Quote endpoint: https://finnhub.io/docs/api/quote
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

import pandas as pd

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)

try:
    import finnhub
except ImportError as e:  # pragma: no cover
    finnhub = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _require_finnhub() -> None:
    """Raise if ``finnhub-python`` is not installed."""
    if finnhub is None:
        raise ImportError(
            "finnhub-python is not installed. pip install finnhub-python"
        ) from _IMPORT_ERROR


def _client(api_key: str | None = None):
    """Build a Finnhub SDK client using ``FINNHUB_API_KEY`` or ``api_key``."""
    _require_finnhub()
    key = api_key or os.getenv("FINNHUB_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Finnhub API key missing. Set FINNHUB_API_KEY in the environment "
            "or pass api_key= to FinnhubIngestor."
        )
    return finnhub.Client(api_key=key)


def _news_records_to_df(records: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    """
    Convert Finnhub ``company_news`` list-of-dicts into a normalized DataFrame.

    Maps Finnhub keys (e.g. Unix ``datetime``, ``id``) to shared column names
    ``datetime``, ``article_id``, ``headline``, etc.
    """
    if not records:
        return pd.DataFrame(
            columns=[
                "symbol",
                "article_id",
                "datetime",
                "headline",
                "summary",
                "source",
                "url",
                "category",
                "image",
                "related",
            ]
        )

    rows: list[dict[str, Any]] = []
    for r in records:
        ts = r.get("datetime")
        if ts is not None:
            try:
                dt = datetime.utcfromtimestamp(int(ts))
            except (TypeError, ValueError, OSError):
                dt = pd.NaT
        else:
            dt = pd.NaT
        rows.append(
            {
                "symbol": symbol.upper(),
                "article_id": r.get("id"),
                "datetime": dt,
                "headline": r.get("headline"),
                "summary": r.get("summary"),
                "source": r.get("source"),
                "url": r.get("url"),
                "category": r.get("category"),
                "image": r.get("image"),
                "related": r.get("related"),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("datetime", na_position="last").reset_index(drop=True)
    return df


def fetch_company_news(
    symbol: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Call Finnhub's ``/company-news`` endpoint and return a normalized news DataFrame.

    Raw items are JSON objects with fields like ``headline``, ``summary``, ``id``, and
    Unix ``datetime``; see Finnhub docs. Parsed output uses :func:`_news_records_to_df`.

    Company news between ``start`` and ``end`` (inclusive date strings for API).

    Finnhub expects ``from`` / ``to`` as ``YYYY-MM-DD``.

    Parameters
    ----------
    symbol
        Ticker, e.g. ``\"AAPL\"``.
    start, end
        Start and end calendar dates for the news window.
    api_key
        Optional; defaults to ``FINNHUB_API_KEY``.

    Returns
    -------
    DataFrame
        One row per article with stable columns for deduplication (e.g. ``article_id``, ``url``).
    """
    client = _client(api_key)
    sym = symbol.strip().upper()

    def _fmt(d: str | date | datetime | pd.Timestamp) -> str:
        """Format a date-like value as ``YYYY-MM-DD`` for Finnhub's ``from``/``to`` params."""
        if isinstance(d, str):
            return d[:10]
        if hasattr(d, "strftime"):
            return d.strftime("%Y-%m-%d")  # type: ignore[union-attr]
        return pd.Timestamp(d).strftime("%Y-%m-%d")

    date_from = _fmt(start)
    date_to = _fmt(end)

    try:
        raw = client.company_news(sym, _from=date_from, to=date_to)
    except Exception:
        logger.exception(
            "Finnhub company_news failed symbol=%s from=%s to=%s",
            sym,
            date_from,
            date_to,
        )
        raise

    if not raw:
        logger.info("Finnhub returned no news for %s %s..%s", sym, date_from, date_to)
        return _news_records_to_df([], sym)

    return _news_records_to_df(list(raw), sym)


def fetch_quote(symbol: str, *, api_key: str | None = None) -> dict[str, Any]:
    """
    Latest quote for a symbol — returns Finnhub's raw dict (c, h, l, o, pc, t, …).

    See Finnhub quote docs for field meanings; we do not reshape this endpoint.
    """
    client = _client(api_key)
    sym = symbol.strip().upper()
    return client.quote(sym)


class FinnhubIngestor:
    """Optional default API key; delegates to module-level fetch functions."""

    def __init__(self, api_key: str | None = None) -> None:
        """``api_key`` overrides ``FINNHUB_API_KEY`` for all methods on this instance."""
        self.api_key = api_key

    def company_news(
        self,
        symbol: str,
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Call :func:`fetch_company_news` with this instance's API key."""
        return fetch_company_news(symbol, start, end, api_key=self.api_key)

    def quote(self, symbol: str) -> dict[str, Any]:
        """Call :func:`fetch_quote` with this instance's API key."""
        return fetch_quote(symbol, api_key=self.api_key)
