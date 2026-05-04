"""Sentiment repo: ``fetch_articles_for_sentiment`` filters and selection logic."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from storage.articles_repo import upsert_articles
from storage.sentiment_repo import (
    build_finbert_input_text,
    fetch_articles_for_sentiment,
    text_hash_for_article,
    upsert_article_sentiment,
)


def test_build_finbert_input_text_headline_only() -> None:
    assert build_finbert_input_text("Only headline", None) == "Only headline"


def test_build_finbert_input_text_summary_only() -> None:
    assert build_finbert_input_text(None, "Only summary") == "Only summary"


def test_build_finbert_input_text_empty() -> None:
    assert build_finbert_input_text(None, None) == ""


def test_text_hash_differs_for_different_strings() -> None:
    assert text_hash_for_article("a") != text_hash_for_article("b")


def test_fetch_articles_only_missing_excludes_scored(
    sqlite_conn: sqlite3.Connection,
    sample_article_df: pd.DataFrame,
) -> None:
    upsert_articles(sqlite_conn, sample_article_df, "finnhub")
    dk = str(
        pd.read_sql_query("SELECT dedupe_key FROM articles", sqlite_conn).iloc[0]["dedupe_key"]
    )
    upsert_article_sentiment(
        sqlite_conn,
        [
            {
                "dedupe_key": dk,
                "model_id": "finbert",
                "score": 0.1,
                "prob_pos": 0.2,
                "prob_neg": 0.3,
                "prob_neutral": 0.4,
                "text_hash": "x",
                "error": None,
            }
        ],
    )
    missing = fetch_articles_for_sentiment(
        sqlite_conn, model_id="finbert", only_missing=True
    )
    assert missing.empty


def test_fetch_articles_rescore_includes_already_scored(
    sqlite_conn: sqlite3.Connection,
    sample_article_df: pd.DataFrame,
) -> None:
    upsert_articles(sqlite_conn, sample_article_df, "finnhub")
    dk = str(
        pd.read_sql_query("SELECT dedupe_key FROM articles", sqlite_conn).iloc[0]["dedupe_key"]
    )
    upsert_article_sentiment(
        sqlite_conn,
        [
            {
                "dedupe_key": dk,
                "model_id": "finbert",
                "score": 0.1,
                "prob_pos": 0.2,
                "prob_neg": 0.3,
                "prob_neutral": 0.4,
                "text_hash": "x",
                "error": None,
            }
        ],
    )
    all_rows = fetch_articles_for_sentiment(
        sqlite_conn, model_id="finbert", only_missing=False
    )
    assert len(all_rows) == 1
    assert str(all_rows.iloc[0]["dedupe_key"]) == dk


def test_fetch_articles_filters_by_symbol(
    sqlite_conn: sqlite3.Connection,
) -> None:
    df = pd.DataFrame(
        [
            {
                "article_id": "1",
                "datetime": "2024-06-15T14:00:00+00:00",
                "headline": "H1",
                "summary": "S1",
                "source": "X",
                "url": "https://a.com/1",
                "symbol": "AAPL",
            },
            {
                "article_id": "2",
                "datetime": "2024-06-15T15:00:00+00:00",
                "headline": "H2",
                "summary": "S2",
                "source": "X",
                "url": "https://a.com/2",
                "symbol": "MSFT",
            },
        ]
    )
    upsert_articles(sqlite_conn, df.iloc[[0]], "finnhub")
    upsert_articles(sqlite_conn, df.iloc[[1]], "finnhub")
    out = fetch_articles_for_sentiment(
        sqlite_conn,
        model_id="finbert",
        symbols=["MSFT"],
        only_missing=True,
    )
    assert len(out) == 1
    assert str(out.iloc[0]["symbol"]) == "MSFT"


def test_fetch_articles_respects_limit(
    sqlite_conn: sqlite3.Connection,
) -> None:
    rows = []
    for i in range(5):
        rows.append(
            {
                "article_id": str(100 + i),
                "datetime": f"2024-06-{10+i:02d}T12:00:00+00:00",
                "headline": f"H{i}",
                "summary": "S",
                "source": "X",
                "url": None,
                "symbol": "AAPL",
            }
        )
    upsert_articles(sqlite_conn, pd.DataFrame(rows), "finnhub")
    out = fetch_articles_for_sentiment(
        sqlite_conn, model_id="finbert", only_missing=True, limit=2
    )
    assert len(out) == 2
