"""Articles repo: dedupe keys, upsert, read."""

from __future__ import annotations

import sqlite3

import pandas as pd

from storage.articles_repo import article_dedupe_key, fetch_articles_frame, upsert_articles


def test_article_dedupe_key_prefers_article_id() -> None:
    row = pd.Series({"article_id": "99", "datetime": "2024-01-01", "headline": "H"})
    k = article_dedupe_key("finnhub", row)
    assert k == "finnhub:id:99"


def test_article_dedupe_key_fallback_url() -> None:
    row = pd.Series(
        {
            "article_id": None,
            "datetime": "2024-01-01",
            "headline": "H",
            "url": "https://news.example/x",
        }
    )
    k = article_dedupe_key("newsapi", row)
    assert "newsapi:url:https://news.example/x" == k


def test_upsert_articles_inserts_and_is_idempotent(
    sqlite_conn: sqlite3.Connection,
    sample_article_df: pd.DataFrame,
) -> None:
    n1 = upsert_articles(sqlite_conn, sample_article_df, "finnhub")
    assert n1 == 1
    n2 = upsert_articles(sqlite_conn, sample_article_df, "finnhub")
    assert n2 == 0
    df = fetch_articles_frame(sqlite_conn, symbol="AAPL", source_api="finnhub")
    assert len(df) == 1
