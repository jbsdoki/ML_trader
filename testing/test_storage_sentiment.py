"""Sentiment repo helpers and upsert."""

from __future__ import annotations

from storage.articles_repo import upsert_articles
from storage.sentiment_repo import (
    build_finbert_input_text,
    fetch_article_sentiment_frame,
    text_hash_for_article,
    upsert_article_sentiment,
)


def test_build_finbert_input_text_concat() -> None:
    t = build_finbert_input_text("Headline here", "Summary body")
    assert "Headline" in t and "Summary" in t


def test_text_hash_stable() -> None:
    h1 = text_hash_for_article("same")
    h2 = text_hash_for_article("same")
    assert h1 == h2 and len(h1) == 64


def test_upsert_article_sentiment_joins_with_article(
    sqlite_conn: sqlite3.Connection,
    sample_article_df,
) -> None:
    import pandas as pd

    upsert_articles(sqlite_conn, sample_article_df, "finnhub")
    df_keys = pd.read_sql_query("SELECT dedupe_key FROM articles", sqlite_conn)
    dk = str(df_keys.iloc[0]["dedupe_key"])
    rows = [
        {
            "dedupe_key": dk,
            "model_id": "finbert",
            "score": 0.25,
            "prob_pos": 0.6,
            "prob_neg": 0.2,
            "prob_neutral": 0.2,
            "text_hash": "abc",
            "error": None,
        }
    ]
    n = upsert_article_sentiment(sqlite_conn, rows)
    assert n == 1
    out = fetch_article_sentiment_frame(sqlite_conn, model_id="finbert", symbols=["AAPL"])
    assert len(out) == 1
    assert float(out.iloc[0]["score"]) == 0.25
