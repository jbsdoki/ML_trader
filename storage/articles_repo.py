"""
Persist normalized news DataFrames (Finnhub, NewsAPI, etc.) with deduplication.

Incoming rows should match the column names produced by ``data.finnhub_ingest``
and ``data.newsapi_ingest`` (``article_id``, ``datetime``, ``headline``, …).
The repo computes a stable ``dedupe_key`` per row so re-running ingest does not
create duplicates.

Useful references
-----------------
- SQLite ``INSERT OR IGNORE``:
  https://www.sqlite.org/lang_conflict.html
- Finnhub company news fields:
  https://finnhub.io/docs/api/company-news
- NewsAPI article object:
  https://newsapi.org/docs/get-started#article-objects-and-result-format
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _utc_iso(dt: Any) -> str:
    """
    Convert a pandas timestamp, ``datetime``, or ISO string to UTC ISO-8601 text.

    NaT / missing values fall back to the Unix epoch for NOT NULL constraint;
    callers should filter empty frames before insert when possible.
    """
    if dt is None or (isinstance(dt, float) and pd.isna(dt)):
        return "1970-01-01T00:00:00+00:00"
    ts = pd.Timestamp(dt)
    if ts is pd.NaT:
        return "1970-01-01T00:00:00+00:00"
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.isoformat()


def _ingested_now_iso() -> str:
    """UTC timestamp for ``ingested_at`` audit column."""
    return datetime.now(timezone.utc).isoformat()


def _scalar_missing_or_blank(val: Any) -> bool:
    """True if ``val`` is None, pandas NA/NaN, empty, or the literal string ``nan``."""
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return True
    return False


def article_dedupe_key(source_api: str, row: pd.Series) -> str:
    """
    Build a stable primary-key string for one article row.

    Prefer provider ``article_id`` when present (Finnhub numeric id, NewsAPI
    hash). Otherwise use normalized URL. Last resort: hash of headline + time
    so rarely-empty rows still get a key.
    """
    src = source_api.strip().lower()
    aid = row.get("article_id")
    if not _scalar_missing_or_blank(aid):
        return f"{src}:id:{str(aid).strip()}"

    url_raw = row.get("url")
    if not _scalar_missing_or_blank(url_raw):
        u = str(url_raw).strip()
        if u:
            return f"{src}:url:{u}"

    hl = row.get("headline")
    headline = "" if _scalar_missing_or_blank(hl) else str(hl).strip()
    when = _utc_iso(row.get("datetime"))
    return f"{src}:fallback:{when}:{headline[:200]}"


def upsert_articles(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source_api: str,
) -> int:
    """
    Insert new article rows; skip rows whose ``dedupe_key`` already exists.

    Uses ``INSERT OR IGNORE`` against the ``dedupe_key`` PRIMARY KEY — this is
    the dedupe mechanism: same logical story → same key → second insert ignored.

    Parameters
    ----------
    conn
        Open SQLite connection (tables must exist — run :func:`storage.schema.init_schema`).
    df
        DataFrame from ingest modules; empty DataFrame returns ``0``.
    source_api
        Short label stored in DB, e.g. ``\"finnhub\"`` or ``\"newsapi\"``.

    Returns
    -------
    int
        Number of rows **newly inserted** (SQLite ``total_changes`` delta). Re-runs
        that only hit conflicts typically return ``0``.
    """
    if df is None or df.empty:
        return 0

    before = conn.total_changes
    src = source_api.strip().lower()
    sql = """
        INSERT OR IGNORE INTO articles (
            dedupe_key, source_api, symbol, external_article_id, published_at,
            headline, summary, source_name, url, author, category, related,
            image_url, content_snippet, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    ingested = _ingested_now_iso()
    attempted = 0
    cur = conn.cursor()

    for _, row in df.iterrows():
        dedupe_key = article_dedupe_key(src, row)
        ext_id = row.get("article_id")
        ext_id_str = None if ext_id is None or (isinstance(ext_id, float) and pd.isna(ext_id)) else str(ext_id)

        sym = row.get("symbol")
        sym_str = None if sym is None or (isinstance(sym, float) and pd.isna(sym)) else str(sym).upper()

        cur.execute(
            sql,
            (
                dedupe_key,
                src,
                sym_str,
                ext_id_str,
                _utc_iso(row.get("datetime")),
                None if pd.isna(row.get("headline")) else str(row.get("headline")),
                None if pd.isna(row.get("summary")) else str(row.get("summary")),
                None if pd.isna(row.get("source")) else str(row.get("source")),
                None if pd.isna(row.get("url")) else str(row.get("url")),
                None if pd.isna(row.get("author")) else str(row.get("author")),
                None if pd.isna(row.get("category")) else str(row.get("category")),
                None if pd.isna(row.get("related")) else str(row.get("related")),
                None if pd.isna(row.get("image")) else str(row.get("image")),
                None if pd.isna(row.get("content_snippet")) else str(row.get("content_snippet")),
                ingested,
            ),
        )
        attempted += 1

    conn.commit()
    inserted = conn.total_changes - before
    logger.info(
        "articles upsert: source=%s rows_attempted=%s rows_inserted=%s",
        src,
        attempted,
        inserted,
    )
    return inserted


def fetch_articles_frame(
    conn: sqlite3.Connection,
    *,
    symbol: str | None = None,
    source_api: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load articles back into a DataFrame for notebooks or downstream sentiment.

    Optional filters by ``symbol`` and/or ``source_api``. ``limit`` caps rows
    (unordered — add ``ORDER BY`` in a later version if you need deterministic tests).
    """
    q = "SELECT * FROM articles WHERE 1=1"
    params: list[Any] = []
    if symbol:
        q += " AND symbol = ?"
        params.append(symbol.upper())
    if source_api:
        q += " AND source_api = ?"
        params.append(source_api.strip().lower())
    if limit is not None:
        q += f" LIMIT {int(limit)}"

    return pd.read_sql_query(q, conn, params=params)
