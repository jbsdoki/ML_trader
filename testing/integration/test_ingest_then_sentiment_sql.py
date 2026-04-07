"""Integration: articles in SQLite -> fetch for sentiment -> upsert scores (no Torch)."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from storage.articles_repo import upsert_articles
from storage.sentiment_repo import fetch_articles_for_sentiment, upsert_article_sentiment


def test_article_to_sentiment_roundtrip(sqlite_conn: sqlite3.Connection) -> None:
    df = pd.DataFrame(
        [
            {
                "article_id": "int-1",
                "datetime": "2024-06-15T14:00:00+00:00",
                "headline": "Integration headline",
                "summary": "Integration summary",
                "source": "finnhub",
                "url": "https://example.com/int-1",
                "symbol": "AAPL",
            }
        ]
    )
    upsert_articles(sqlite_conn, df, "finnhub")

    pending = fetch_articles_for_sentiment(
        sqlite_conn, model_id="finbert", only_missing=True
    )
    assert len(pending) == 1
    dk = str(pending.iloc[0]["dedupe_key"])

    n = upsert_article_sentiment(
        sqlite_conn,
        [
            {
                "dedupe_key": dk,
                "model_id": "finbert",
                "score": 0.42,
                "prob_pos": 0.5,
                "prob_neg": 0.25,
                "prob_neutral": 0.25,
                "text_hash": "hash",
                "error": None,
            }
        ],
    )
    assert n >= 1

    again = fetch_articles_for_sentiment(
        sqlite_conn, model_id="finbert", only_missing=True
    )
    assert again.empty

    score = sqlite_conn.execute(
        "SELECT score FROM article_sentiment WHERE dedupe_key = ? AND model_id = ?",
        (dk, "finbert"),
    ).fetchone()
    assert score is not None
    assert float(score[0]) == pytest.approx(0.42)
