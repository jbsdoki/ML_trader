"""
Schema and sample-row inspection using the in-memory ``sqlite_conn`` fixture only.

Does **not** open ``data_store/ml_trader.db``. One test inserts minimal rows in memory,
then writes ``sqlite_sample_export.json`` under ``logs/testing/data_store/`` (gitignored
via ``logs/*``) for local inspection; assertions use the same payload.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root and export dir for the sample JSON written by ``test_sample_rows_exported_to_logs_testing_data_store``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_EXPORT_DIR = _REPO_ROOT / "logs" / "testing" / "data_store"


def _user_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted user table names from ``sqlite_master`` (excludes ``sqlite_*`` internals)."""
    cur = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [str(r[0]) for r in cur.fetchall()]


def _pragma_table_columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    """Return column metadata for ``table`` via ``PRAGMA table_info`` (names, types, PK flags)."""
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    out: list[dict[str, Any]] = []
    for r in cur.fetchall():
        out.append(
            {
                "cid": r[0],
                "name": r[1],
                "type": r[2],
                "notnull": bool(r[3]),
                "default_value": r[4],
                "pk": bool(r[5]),
            }
        )
    return out


def test_expected_user_tables_exist(sqlite_conn: sqlite3.Connection) -> None:
    """Check that the initialized schema exposes the three app tables: articles, bars, article_sentiment."""
    names = _user_table_names(sqlite_conn)
    assert "articles" in names
    assert "bars" in names
    assert "article_sentiment" in names


def test_articles_table_columns(sqlite_conn: sqlite3.Connection) -> None:
    """Check ``articles`` has core columns used by ingest (dedupe key, source, publish time)."""
    col_names = [c["name"] for c in _pragma_table_columns(sqlite_conn, "articles")]
    assert "dedupe_key" in col_names
    assert "source_api" in col_names
    assert "published_at" in col_names


def test_bars_table_columns(sqlite_conn: sqlite3.Connection) -> None:
    """Check ``bars`` has the composite-key pieces (source, symbol, bar timestamp)."""
    col_names = [c["name"] for c in _pragma_table_columns(sqlite_conn, "bars")]
    assert "source_api" in col_names
    assert "symbol" in col_names
    assert "bar_ts" in col_names


def test_article_sentiment_table_columns(sqlite_conn: sqlite3.Connection) -> None:
    """Check ``article_sentiment`` keys articles by ``dedupe_key`` and ``model_id``."""
    col_names = [c["name"] for c in _pragma_table_columns(sqlite_conn, "article_sentiment")]
    assert "dedupe_key" in col_names
    assert "model_id" in col_names


def test_sample_rows_exported_to_logs_testing_data_store(
    sqlite_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """
    Verify we can insert minimal valid rows (in-memory only), read them back with ``SELECT``,
    serialize columns and sample rows to JSON, write that file to ``logs/testing/data_store/``
    for manual inspection, mirror the same JSON under ``tmp_path`` for assertions, and assert
    each table returned at least one row and the on-disk log path exists.
    """
    conn = sqlite_conn
    conn.execute(
        """
        INSERT INTO articles (
            dedupe_key, source_api, published_at, ingested_at
        ) VALUES (?, ?, ?, ?)
        """,
        (
            "inspect:test:1",
            "test",
            "2024-06-15T12:00:00+00:00",
            "2024-06-15T12:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO bars (
            source_api, symbol, bar_interval, bar_ts, close, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "test",
            "AAPL",
            "1d",
            "2024-06-14T00:00:00+00:00",
            100.0,
            "2024-06-15T12:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO article_sentiment (
            dedupe_key, model_id, score
        ) VALUES (?, ?, ?)
        """,
        ("inspect:test:1", "inspect_model", 0.25),
    )
    conn.commit()

    payload: dict[str, Any] = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables": {},
    }
    for table in ("articles", "bars", "article_sentiment"):
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT 2')
        col_names = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
        payload["tables"][table] = {"columns": col_names, "rows": rows}

    text = json.dumps(payload, indent=2, default=str)

    _LOG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_EXPORT_DIR / "sqlite_sample_export.json"
    log_file.write_text(text, encoding="utf-8")

    tmp_copy = tmp_path / "sqlite_sample_export.json"
    tmp_copy.write_text(text, encoding="utf-8")

    loaded = json.loads(tmp_copy.read_text(encoding="utf-8"))
    assert len(loaded["tables"]["articles"]["rows"]) >= 1
    assert len(loaded["tables"]["bars"]["rows"]) >= 1
    assert len(loaded["tables"]["article_sentiment"]["rows"]) >= 1
    assert log_file.is_file()
