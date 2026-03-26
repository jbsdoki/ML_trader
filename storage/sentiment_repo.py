"""
Read/write helpers for article-level sentiment in SQLite.
This module is the storage bridge between:
- `articles` table (raw ingested news)
- `article_sentiment` table (derived model outputs)
Useful references:
- SQLite upsert: https://www.sqlite.org/lang_UPSERT.html
- pandas read_sql_query: https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

def _utc_iso(value: Any) -> str:
    """
    Convert date/time value to UTC ISO-8601 string.
    Used to keep query bounds and audit fields consistent.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.isoformat()


def _now_utc_iso() -> str:
    """
    Current UTC timestamp converted to ISO-8601 format
    """
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """
    Convert value to float or return None
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_finbert_input_text(headline: Any, summary: Any) -> str:
    """
    Create the text string sent to the FinBERT model

    Prefer the headline and summary if available,
    Otherwise use whichever exists
    return empty string if both missing
    """

    h = "" if headline is None or (isinstance(headline, float) and pd.isna(headline)) else str(headline).strip()
    s = "" if summary is None or (isinstance(summary, float) and pd.isna(summary)) else str(summary).strip()

    if h and s:
        return f"{h} {s}"
    if h:
        return h
    if s:
        return s
    return ""


def text_hash_for_article(text: str) -> str:
    """
    Create a hash of the text for deduplication
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_articles_for_sentiment(
    conn: sqlite3.Connection,
    *, #every argument after this must be passed by keyword (conn, model_id="finbert", etc...)'
    model_id: str,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    only_missing: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load article rows to score with the sentiment model

    if 'only_missing=True' then rows that alread have sentiment with this 
    model id will be skipped
    """
    query = """
        SELECT
            a.dedupe_key,
            a.symbol,
            a.source_api,
            a.published_at,
            a.headline,
            a.summary
        From articles a
    """

    params: list[Any] = []

    #left join filters out articles already scored with this model id
    if only_missing:
        query += """
            LEFT JOIN article_sentiment as s
            ON s.dedupe_key = a.dedupe_key
            AND s.model_id = ?
        """
        params.append(model_id)

    query += " WHERE 1=1"

    if only_missing:
        query += " AND s.dedupe_key IS NULL"

    #If passed symbols "AAPL, MSFT, etc" then build a list without spaces and all uppercase
    if symbols:
        cleaned = [x.strip().upper() for x in symbols if x and x.strip()]
        if cleaned:
            placeholders = ",".join(["?"] * len(cleaned)) #builds comma separated list for every symbol "?"
            query += f" AND a.symbol IN ({placeholders})"
            params.extend(cleaned)

    if start:
        query += " AND a.published_at >= ?"
        params.append(_utc_iso(start))

    if end:
        query += " AND a.published_at <= ?"
        params.append(_utc_iso(end))

    #After it has found all rows that match WHERE filter
    # sort those rows in ascending order (ASC)
    query += " ORDER BY a.published_at ASC"

    #Limit is max number of rows user asks for
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    return pd.read_sql_query(query, conn, params=params)


def upsert_article_sentiment(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> int:
    """
    Upsert article sentiment rows by (dedupe_key, model_id) 
    takes a list of sentiment results and writes them into the article_sentiment table

    returns number of rows inserted or updated
    
    Expected Row Keys are:
    dedupe_key (str), model_id (str), score (float or None), prob_pos (float or None), prob_neg (float or None), 
    prob_neutral (float or None), scored_at (str or None), text_hash (str or None), error (str or None)
    """
    if not rows:
        return 0

    sql = """
        INSERT INTO article_sentiment (
        dedupe_key, 
        model_id, 
        score, 
        prob_pos, 
        prob_neg, 
        prob_neutral, 
        scored_at, 
        text_hash, 
        error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (dedupe_key, model_id) DO UPDATE SET
            score           = excluded.score,
            prob_pos        = excluded.prob_pos,
            prob_neg        = excluded.prob_neg,
            prob_neutral    = excluded.prob_neutral,
            scored_at       = excluded.scored_at,
            text_hash       = excluded.text_hash,
            error           = excluded.error
    """

    #Total changes before upsert
    before = conn.total_changes
    now_iso = _now_utc_iso()
    
    values: list[tuple[Any, ...]] = []
    for r in rows:
        values.append(
            (
                r["dedupe_key"],
                r["model_id"],
                _safe_float(r.get("score")),
                _safe_float(r.get("prob_pos")),
                _safe_float(r.get("prob_neg")),
                _safe_float(r.get("prob_neutral")),
                now_iso,
                r.get("text_hash"),
                r.get("error"),
            )
        )

    cur = conn.cursor()
    cur.executemany(sql, values) #Runs upsert statement for every tuple
    conn.commit() #Saves the transaction

    return conn.total_changes - before



def fetch_article_sentiment_frame(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Read sentiment rows joined with article metadat for validation/debugging
    """
    query = """
        SELECT
            s.dedupe_key,
            s.model_id,
            s.score,
            s.prob_pos,
            s.prob_neg,
            s.prob_neutral,
            s.scored_at,
            s.text_hash,
            s.error,
            a.symbol,
            a.source_api,
            a.published_at
        From article_sentiment as s
        JOIN articles as a ON a.dedupe_key = s.dedupe_key
        WHERE s.model_id = ?
    """

    params: list[Any] = [model_id]

    if symbols:
        cleaned = [x.strip().upper() for x in symbols if x and x.strip()]
        if cleaned:
            placeholders = ",".join(["?"] * len(cleaned))
            query += f" AND a.symbol IN ({placeholders})"
            params.extend(cleaned)

    if start:
        query += " AND a.published_at >= ?"
        params.append(_utc_iso(start))

    if end:
        query += " AND a.published_at <= ?"
        params.append(_utc_iso(end))

    query += " ORDER BY a.published_at ASC"

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    return pd.read_sql_query(query, conn, params=params)
