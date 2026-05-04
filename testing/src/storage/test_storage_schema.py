"""Schema application and table presence."""

from __future__ import annotations

import sqlite3


def test_init_schema_creates_core_tables(sqlite_conn: sqlite3.Connection) -> None:
    cur = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {row[0] for row in cur.fetchall()}
    assert "articles" in names
    assert "bars" in names
    assert "article_sentiment" in names
