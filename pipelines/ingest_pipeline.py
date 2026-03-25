"""
Orchestrate **fetch → persist → dedupe** for news and OHLCV bars.

This module ties together:

- **Ingest** — :mod:`data` clients (Finnhub, NewsAPI, yfinance, Alpaca).
- **Storage** — :mod:`storage` SQLite upserts (see ``INSERT OR IGNORE`` / primary keys).

**Useful background**

- Twelve-factor config (env vars for secrets, paths):
  https://12factor.net/config
- SQLite ``INSERT OR IGNORE`` (dedupe on conflict):
  https://www.sqlite.org/lang_conflict.html
- Finnhub company news:
  https://finnhub.io/docs/api/company-news
- NewsAPI everything / limits:
  https://newsapi.org/docs/endpoints/everything
- yfinance ``history``:
  https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html
- Alpaca stock bars:
  https://docs.alpaca.markets/reference/stockbars

**Definitions**

- **Pipeline** — One coordinated run: open DB, optionally init schema, pull from APIs, upsert rows.
- **Dedupe** — Handled inside :mod:`storage` repos; re-running the same window does not duplicate PKs.
- **Bar interval** — Candle size (e.g. ``1d``, ``1h``). yfinance and Alpaca support overlapping but not identical sets; unsupported Alpaca intervals are skipped with a logged warning.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpaca bar intervals accepted by data.alpaca_ingest.timeframe_from_string
# (yfinance allows more, e.g. ``5d`` — Alpaca path is skipped for those unless mapped later).
# ---------------------------------------------------------------------------
_ALPACA_INTERVALS = frozenset(
    {"1m", "5m", "15m", "30m", "1h", "60m", "1d", "1w", "1wk", "1mo"}
)


@dataclass
class IngestConfig:
    """
    **IngestConfig** — All inputs needed for one ``run_ingest_pipeline`` execution.

    Attributes
    ----------
    symbols
        Equity tickers to process (uppercased when stored).
    start, end
        Calendar window ``YYYY-MM-DD`` for **news** (Finnhub/NewsAPI) and **bar** history.
    bar_interval
        OHLCV bar size passed to yfinance/Alpaca (e.g. ``1d``, ``1h``).
    finnhub, newsapi, yfinance, alpaca
        Enable flags per data source (missing API keys → warning + skip for that source).
    alpaca_feed
        Optional Alpaca data feed (e.g. ``sip``, ``iex``); ``None`` uses Alpaca default.
    newsapi_extra_query
        Optional OR-query fragment for :func:`data.newsapi_ingest.fetch_for_symbol`
        (e.g. company name) to improve recall beyond the raw ticker symbol.
    newsapi_page_size, newsapi_max_pages
        NewsAPI pagination controls. Defaults are dev-tier safe to avoid page-2
        failures on free plans (max 100 results total).
    """

    symbols: list[str]
    start: str
    end: str
    bar_interval: str = "1d"
    finnhub: bool = True
    newsapi: bool = True
    yfinance: bool = True
    alpaca: bool = True
    alpaca_feed: str | None = None
    newsapi_extra_query: str | None = None
    # Improvement: default to one page to avoid free-tier "maximumResultsReached".
    newsapi_page_size: int = 100
    newsapi_max_pages: int = 1


@dataclass
class IngestSummary:
    """
    **IngestSummary** — Counters and errors after a pipeline run.

    ``*_inserted`` counts come from SQLite ``total_changes`` (actual new rows), not attempts.
    """

    articles_finnhub_inserted: int = 0
    articles_newsapi_inserted: int = 0
    bars_yfinance_inserted: int = 0
    bars_alpaca_inserted: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_symbols(symbols: list[str]) -> list[str]:
    """
    Strip whitespace, drop empties, uppercase tickers.

    **Ticker** — Exchange symbol string (e.g. ``AAPL``); not validated against a master list here.
    """
    out: list[str] = []
    for s in symbols:
        t = (s or "").strip().upper()
        if t:
            out.append(t)
    return out


def parse_sources_csv(s: str) -> dict[str, bool]:
    """
    Parse a comma-separated source list into enable flags.

    Recognized tokens (case-insensitive): ``finnhub``, ``newsapi``, ``yfinance``, ``alpaca``.
    Unknown tokens are ignored with a warning.
    """
    raw = {x.strip().lower() for x in s.split(",") if x.strip()}
    if not raw:
        return {
            "finnhub": True,
            "newsapi": True,
            "yfinance": True,
            "alpaca": True,
        }
    known = ("finnhub", "newsapi", "yfinance", "alpaca")
    for token in raw:
        if token not in known:
            logger.warning("Unknown source in --sources: %r (ignored)", token)
    return {k: (k in raw) for k in known}


def load_ingest_config_yaml(path: Path) -> IngestConfig:
    """
    Load :class:`IngestConfig` from a YAML file (requires ``PyYAML``).

    Expected keys (minimal):

    .. code-block:: yaml

        symbols: [AAPL, MSFT]
        start: \"2025-01-01\"
        end: \"2025-03-18\"
        bar_interval: \"1d\"
        sources: [finnhub, newsapi, yfinance, alpaca]   # optional; default all
        newsapi_extra_query: null
        alpaca_feed: null
        newsapi_page_size: 100
        newsapi_max_pages: 1

    References: https://pyyaml.org/wiki/PyYAMLDocumentation
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Install PyYAML to use YAML config: pip install pyyaml") from e

    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text) or {}

    symbols = data.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [x.strip() for x in symbols.split(",") if x.strip()]
    start = str(data.get("start", "")).strip()
    end = str(data.get("end", "")).strip()
    if not symbols or not start or not end:
        raise ValueError(f"YAML {path} must define symbols, start, and end")

    bar_interval = str(data.get("bar_interval", "1d")).strip()
    sources = data.get("sources")
    if sources is None:
        finnhub = newsapi = yfinance = alpaca = True
    else:
        if isinstance(sources, str):
            src_set = {x.strip().lower() for x in sources.split(",") if x.strip()}
        else:
            src_set = {str(x).strip().lower() for x in sources}
        finnhub = "finnhub" in src_set
        newsapi = "newsapi" in src_set
        yfinance = "yfinance" in src_set
        alpaca = "alpaca" in src_set

    extra = data.get("newsapi_extra_query")
    feed = data.get("alpaca_feed")
    n_page_size = int(data.get("newsapi_page_size", 100))
    n_max_pages = int(data.get("newsapi_max_pages", 1))

    return IngestConfig(
        symbols=normalize_symbols(list(symbols)),
        start=start[:10],
        end=end[:10],
        bar_interval=bar_interval,
        finnhub=finnhub,
        newsapi=newsapi,
        yfinance=yfinance,
        alpaca=alpaca,
        alpaca_feed=None if feed in (None, "") else str(feed),
        newsapi_extra_query=None if extra in (None, "") else str(extra),
        newsapi_page_size=max(1, min(n_page_size, 100)),
        newsapi_max_pages=max(1, n_max_pages),
    )


def _ingest_finnhub(
    conn: sqlite3.Connection,
    symbols: list[str],
    start: str,
    end: str,
    summary: IngestSummary,
) -> None:
    """Fetch Finnhub company news per symbol and upsert into ``articles``."""
    from data_retrieval.finnhub_ingest import fetch_company_news
    from storage.articles_repo import upsert_articles

    for sym in symbols:
        try:
            df = fetch_company_news(sym, start, end)
            summary.articles_finnhub_inserted += upsert_articles(conn, df, "finnhub")
        except Exception as e:
            msg = f"Finnhub news failed for {sym}: {e}"
            logger.exception(msg)
            summary.errors.append(msg)


def _ingest_newsapi(
    conn: sqlite3.Connection,
    symbols: list[str],
    start: str,
    end: str,
    extra_query: str | None,
    page_size: int,
    max_pages: int,
    summary: IngestSummary,
) -> None:
    """Fetch NewsAPI articles per symbol and upsert into ``articles``."""
    from data_retrieval.newsapi_ingest import fetch_for_symbol
    from storage.articles_repo import upsert_articles

    for sym in symbols:
        try:
            df = fetch_for_symbol(
                sym,
                start,
                end,
                extra_query_terms=extra_query,
                # Improvement: pass explicit paging caps from config so users can
                # tune free-tier behavior without changing code.
                page_size=page_size,
                max_pages=max_pages,
            )
            summary.articles_newsapi_inserted += upsert_articles(conn, df, "newsapi")
        except Exception as e:
            msg = f"NewsAPI failed for {sym}: {e}"
            logger.exception(msg)
            summary.errors.append(msg)


def _ingest_yfinance_bars(
    conn: sqlite3.Connection,
    symbols: list[str],
    start: str,
    end: str,
    interval: str,
    summary: IngestSummary,
) -> None:
    """
    Download OHLCV via yfinance and upsert into ``bars``.

    **Note:** yfinance ``end`` is **exclusive**. We pass ``end + 1 calendar day`` so the
    CLI ``--end`` date behaves **inclusive**, matching Finnhub/NewsAPI date windows.
    See https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html
    """
    from data_retrieval.yfinance_ingest import fetch_ohlcv
    from storage.bars_repo import upsert_bars

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, start=start, end=end_exclusive, interval=interval)
            if df.empty:
                logger.warning("yfinance returned no bars for %s", sym)
                continue
            # fetch_ohlcv uses DatetimeIndex; bars_repo accepts index or column
            work = df.reset_index()
            if "timestamp" not in work.columns and work.columns.size > 0:
                first = work.columns[0]
                work = work.rename(columns={first: "timestamp"})
            summary.bars_yfinance_inserted += upsert_bars(
                conn, work, "yfinance", interval
            )
        except Exception as e:
            msg = f"yfinance bars failed for {sym}: {e}"
            logger.exception(msg)
            summary.errors.append(msg)


def _ingest_alpaca_bars(
    conn: sqlite3.Connection,
    symbols: list[str],
    start: str,
    end: str,
    interval: str,
    feed: str | None,
    summary: IngestSummary,
) -> None:
    """Download OHLCV via Alpaca market data and upsert into ``bars``."""
    from data_retrieval.alpaca_ingest import fetch_stock_bars
    from storage.bars_repo import upsert_bars

    iv = interval.strip().lower()
    if iv not in _ALPACA_INTERVALS:
        msg = (
            f"Alpaca ingest skipped: interval {interval!r} not in supported set "
            f"{sorted(_ALPACA_INTERVALS)}. Use yfinance for this interval or extend mapping."
        )
        logger.warning(msg)
        summary.errors.append(msg)
        return

    # Inclusive calendar ``end`` -> start of next day as exclusive upper bound (common API pattern).
    def _utc(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    start_ts = _utc(pd.Timestamp(start))
    end_ts = _utc(pd.Timestamp(end) + pd.Timedelta(days=1))

    for sym in symbols:
        try:
            df = fetch_stock_bars(
                sym,
                start_ts,
                end_ts,
                interval=iv,
                feed=feed,
            )
            if df.empty:
                logger.warning("Alpaca returned no bars for %s", sym)
                continue
            summary.bars_alpaca_inserted += upsert_bars(conn, df, "alpaca", iv)
        except Exception as e:
            msg = f"Alpaca bars failed for {sym}: {e}"
            logger.exception(msg)
            summary.errors.append(msg)


def run_ingest_pipeline(
    config: IngestConfig,
    *,
    init_db: bool = True,
    conn: sqlite3.Connection | None = None,
) -> IngestSummary:
    """
    **run_ingest_pipeline** — Main entry: optional DB connect, init schema, run enabled sources.

    Parameters
    ----------
    config
        What to pull and which sources are on.
    init_db
        If True (default), call :func:`storage.schema.init_schema` so tables exist.
    conn
        Optional open SQLite connection for tests; if ``None``, opens via :func:`storage.database.connect`
        and closes before return.

    Returns
    -------
    IngestSummary
        Insert counts and any captured error strings (individual symbol failures do not abort others).
    """
    from storage.database import connect
    from storage.schema import init_schema

    summary = IngestSummary()
    symbols = normalize_symbols(config.symbols)
    if not symbols:
        summary.errors.append("No symbols after normalization; nothing to do.")
        return summary

    own_conn = conn is None
    if own_conn:
        conn = connect()

    try:
        if init_db:
            init_schema(conn)

        if config.finnhub:
            _ingest_finnhub(conn, symbols, config.start, config.end, summary)
        if config.newsapi:
            _ingest_newsapi(
                conn,
                symbols,
                config.start,
                config.end,
                config.newsapi_extra_query,
                config.newsapi_page_size,
                config.newsapi_max_pages,
                summary,
            )
        if config.yfinance:
            _ingest_yfinance_bars(
                conn,
                symbols,
                config.start,
                config.end,
                config.bar_interval,
                summary,
            )
        if config.alpaca:
            _ingest_alpaca_bars(
                conn,
                symbols,
                config.start,
                config.end,
                config.bar_interval,
                config.alpaca_feed,
                summary,
            )
    finally:
        if own_conn and conn is not None:
            conn.close()

    logger.info(
        "Ingest complete: finnhub_articles=%s newsapi_articles=%s yf_bars=%s alpaca_bars=%s errors=%s",
        summary.articles_finnhub_inserted,
        summary.articles_newsapi_inserted,
        summary.bars_yfinance_inserted,
        summary.bars_alpaca_inserted,
        len(summary.errors),
    )
    return summary
