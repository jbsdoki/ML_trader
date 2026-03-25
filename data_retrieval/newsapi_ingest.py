"""
NewsAPI.org — headlines and article metadata via ``newsapi-python``.

Requires env var ``NEWSAPI_API_KEY`` (or pass ``api_key=`` explicitly).

**Plan limits:** free/developer tiers restrict how far back ``get_everything`` can
search and how many requests you can make per day. Check your dashboard.

**Why this file is NewsAPI-specific:** Responses wrap articles in ``{ "status", "articles" }``
with per-article keys like ``title``, ``description``, ``publishedAt``, nested ``source``.
We map those into the same broad column names as Finnhub where possible (``headline``,
``summary``, ``datetime``, …) so downstream code can merge sources.

**Official format / API references**

https://newsapi.org/docs/client-libraries/python

- Everything endpoint: https://newsapi.org/docs/endpoints/everything
- Top headlines: https://newsapi.org/docs/endpoints/top-headlines
- Article object in responses: https://newsapi.org/docs/get-started#article-objects-and-result-format
- Python package: https://github.com/mattlisiv/newsapi-python
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from newsapi import NewsApiClient
    from newsapi.newsapi_exception import NewsAPIException
except ImportError as e:  # pragma: no cover
    NewsApiClient = None  # type: ignore[misc, assignment]
    NewsAPIException = Exception  # type: ignore[misc, assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _require_newsapi() -> None:
    """Raise if ``newsapi-python`` is not installed."""
    if NewsApiClient is None:
        raise ImportError(
            "newsapi-python is not installed. pip install newsapi-python"
        ) from _IMPORT_ERROR


def _client(api_key: str | None = None) -> Any:
    """Instantiate ``NewsApiClient`` from ``NEWSAPI_API_KEY`` or ``api_key``."""
    _require_newsapi()
    key = (api_key or os.getenv("NEWSAPI_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "NewsAPI key missing. Set NEWSAPI_API_KEY or pass api_key= to NewsAPIIngestor."
        )
    return NewsApiClient(api_key=key)


def _fmt_date(d: str | date | datetime | pd.Timestamp) -> str:
    """Format a date for NewsAPI ``from`` / ``to`` query params (``YYYY-MM-DD``)."""
    if isinstance(d, str):
        return d[:10]
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _article_id(url: str | None, title: str | None) -> str:
    """
    Stable synthetic ID: NewsAPI articles may omit a numeric id; hash URL + title.

    Use for deduplication alongside ``url`` in your DB.
    """
    base = (url or "") + "\n" + (title or "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def _articles_to_df(articles: list[dict[str, Any]], context_symbol: str | None) -> pd.DataFrame:
    """
    Map NewsAPI ``articles`` list (see their article object spec) to a DataFrame.

    ``title`` -> ``headline``, ``description`` -> ``summary``, ``publishedAt`` -> ``datetime``.
    """
    if not articles:
        return pd.DataFrame(
            columns=[
                "symbol",
                "article_id",
                "datetime",
                "headline",
                "summary",
                "source",
                "url",
                "author",
                "content_snippet",
            ]
        )

    rows: list[dict[str, Any]] = []
    for a in articles:
        src = a.get("source") or {}
        src_name = src.get("name") if isinstance(src, dict) else None
        published = a.get("publishedAt")
        dt = pd.to_datetime(published, utc=True, errors="coerce") if published else pd.NaT
        title = a.get("title")
        url = a.get("url")
        rows.append(
            {
                "symbol": (context_symbol or "").upper() or None,
                "article_id": _article_id(url, title),
                "datetime": dt,
                "headline": title,
                "summary": a.get("description"),
                "source": src_name,
                "url": url,
                "author": a.get("author"),
                "content_snippet": (a.get("content") or "")[:2000],
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("datetime", na_position="last").reset_index(drop=True)
    return df


def fetch_everything(
    query: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 100,
    max_pages: int = 5,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Paginate through NewsAPI ``/v2/everything`` and return a normalized DataFrame.

    Parameters
    ----------
    query
        NewsAPI query string (see their query syntax), e.g. ``\"Apple\"`` or ``\"AAPL\"``.
    start, end
        Inclusive-ish window; API uses ``from`` and ``to`` as ``YYYY-MM-DD``.
    max_pages
        Safety cap: each page pulls up to ``page_size`` articles (max 100).
    """
    client = _client(api_key)
    date_from = _fmt_date(start)
    date_to = _fmt_date(end)

    all_rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        try:
            resp = client.get_everything(
                q=query,
                from_param=date_from,
                to=date_to,
                language=language,
                sort_by=sort_by,
                page=page,
                page_size=min(page_size, 100),
            )
        except NewsAPIException as e:
            # Improvement: free/developer plans cap "everything" to the first 100 results.
            # Treat this specific cap as partial success (stop paging) instead of failing
            # the whole symbol ingest after page 1 already returned usable rows.
            payload = e.args[0] if e.args else {}
            code = payload.get("code") if isinstance(payload, dict) else None
            if code == "maximumResultsReached":
                logger.warning(
                    "NewsAPI reached max results for query=%r (%s..%s); using collected rows only",
                    query,
                    date_from,
                    date_to,
                )
                break
            logger.exception(
                "NewsAPI get_everything failed query=%r from=%s to=%s page=%s",
                query,
                date_from,
                date_to,
                page,
            )
            raise
        except Exception:
            logger.exception(
                "NewsAPI get_everything failed query=%r from=%s to=%s page=%s",
                query,
                date_from,
                date_to,
                page,
            )
            raise

        status = resp.get("status")
        if status != "ok":
            logger.warning("NewsAPI non-ok status: %s %s", status, resp.get("message"))
            break

        batch = resp.get("articles") or []
        all_rows.extend(batch)
        total = int(resp.get("totalResults") or 0)
        if len(batch) < min(page_size, 100) or len(all_rows) >= total:
            break

    return _articles_to_df(all_rows, context_symbol=None)


def fetch_for_symbol(
    symbol: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    *,
    extra_query_terms: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Build a search query from a ticker (optional extra OR terms) then call :func:`fetch_everything`.

    Example query: ``\"AAPL OR Apple\"`` (tweak ``extra_query_terms`` for company name).
    """
    sym = symbol.strip().upper()
    q = sym
    if extra_query_terms:
        q = f"({sym}) OR ({extra_query_terms})"
    df = fetch_everything(q, start, end, api_key=api_key, **kwargs)
    if not df.empty:
        df["symbol"] = sym
    return df


def fetch_top_headlines(
    *,
    category: str | None = None,
    country: str = "us",
    q: str | None = None,
    page_size: int = 100,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch ``/v2/top-headlines`` and normalize to the same columns as :func:`fetch_everything`.

    Parameters follow NewsAPI's top-headlines endpoint (country, category, optional ``q``).
    """
    client = _client(api_key)
    try:
        resp = client.get_top_headlines(
            q=q,
            category=category,
            country=country,
            page_size=min(page_size, 100),
        )
    except Exception:
        logger.exception("NewsAPI get_top_headlines failed")
        raise

    if resp.get("status") != "ok":
        logger.warning("NewsAPI top_headlines: %s", resp.get("message"))
        return _articles_to_df([], None)

    articles = resp.get("articles") or []
    return _articles_to_df(articles, context_symbol=None)


class NewsAPIIngestor:
    """Holds optional default API key for NewsAPI calls."""

    def __init__(self, api_key: str | None = None) -> None:
        """``api_key`` overrides ``NEWSAPI_API_KEY`` for methods on this instance."""
        self.api_key = api_key

    def everything(
        self,
        query: str,
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Instance wrapper for :func:`fetch_everything`."""
        return fetch_everything(query, start, end, api_key=self.api_key, **kwargs)

    def for_symbol(
        self,
        symbol: str,
        start: str | date | datetime | pd.Timestamp,
        end: str | date | datetime | pd.Timestamp,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Instance wrapper for :func:`fetch_for_symbol`."""
        return fetch_for_symbol(symbol, start, end, api_key=self.api_key, **kwargs)

    def top_headlines(self, **kwargs: Any) -> pd.DataFrame:
        """Instance wrapper for :func:`fetch_top_headlines`."""
        return fetch_top_headlines(api_key=self.api_key, **kwargs)
