"""
Create and migrate SQLite schema (tables, indexes, primary keys for dedupe).

Dedupe is enforced by **unique keys** SQLite understands: re-inserting the same
logical row becomes a no-op when using ``INSERT OR IGNORE`` in the repos.

Useful references
-----------------
- ``CREATE TABLE``:
  https://www.sqlite.org/lang_createtable.html
- ``CREATE INDEX``:
  https://www.sqlite.org/lang_createindex.html
- ``INSERT OR IGNORE`` / conflict resolution:
  https://www.sqlite.org/lang_conflict.html
"""

from __future__ import annotations

import logging
import sqlite3

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)

# SQL executed in order when initializing a fresh database.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS articles (
        dedupe_key TEXT PRIMARY KEY,
        source_api TEXT NOT NULL,
        symbol TEXT,
        external_article_id TEXT,
        published_at TEXT NOT NULL,
        headline TEXT,
        summary TEXT,
        source_name TEXT,
        url TEXT,
        author TEXT,
        category TEXT,
        related TEXT,
        image_url TEXT,
        content_snippet TEXT,
        ingested_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_articles_source_time
    ON articles (source_api, published_at);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_articles_symbol_time
    ON articles (symbol, published_at);
    """,
    """
    CREATE TABLE IF NOT EXISTS bars (
        source_api TEXT NOT NULL,
        symbol TEXT NOT NULL,
        bar_interval TEXT NOT NULL,
        bar_ts TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        vwap REAL,
        trade_count INTEGER,
        dividends REAL,
        stock_splits REAL,
        ingested_at TEXT NOT NULL,
        PRIMARY KEY (source_api, symbol, bar_interval, bar_ts)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_bars_symbol_interval_time
    ON bars (symbol, bar_interval, bar_ts);
    """,
    """
    CREATE TABLE IF NOT EXISTS article_sentiment (
        dedupe_key TEXT NOT NULL,
        model_id TEXT NOT NULL,
        score REAL,
        prob_pos REAL,
        prob_neg REAL,
        prob_neutral REAL,
        scored_at TEXT,
        text_hash TEXT,
        error TEXT,
        PRIMARY KEY (dedupe_key, model_id),
        FOREIGN KEY (dedupe_key) REFERENCES articles(dedupe_key)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_article_sentiment__model_id
    ON article_sentiment (model_id);
    """
)


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Apply all DDL statements idempotently (safe to call on every app start).

    Creates ``articles`` (one row per unique news item) ``bars`` (one row
    per OHLCV bar) and article_sentiment (one row per article and model). 
    Composite primary key on ``bars`` prevents duplicate bars for the same 
    vendor, symbol, interval, and bar open time.
    """
    cur = conn.cursor()
    try:
        for stmt in _SCHEMA_STATEMENTS:
            cur.execute(stmt)
        conn.commit()
    except sqlite3.Error:
        logger.exception("init_schema failed while applying DDL")
        raise


def schema_version(conn: sqlite3.Connection) -> int:
    """
    Return a simple integer schema version for future migrations.

    Currently returns ``1`` if ``articles`` exists, else ``0``. Extend this when
    you add ``PRAGMA user_version`` or a ``schema_migrations`` table.
    """
    cur = conn.execute(
        "SELECT COUNT(*) AS c FROM sqlite_master WHERE type='table' AND name='articles';"
    )
    row = cur.fetchone()
    return 1 if row is not None and int(row[0]) > 0 else 0
