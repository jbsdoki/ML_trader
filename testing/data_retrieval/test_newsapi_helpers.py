"""NewsAPI ingest: pure helpers (no HTTP)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from data_retrieval import newsapi_ingest as n


def test_fmt_date_truncates_string_and_formats_date() -> None:
    assert n._fmt_date("2024-06-15T12:00:00") == "2024-06-15"
    assert n._fmt_date(date(2024, 6, 15)) == "2024-06-15"
    assert n._fmt_date(datetime(2024, 6, 15, 9, 30)) == "2024-06-15"


def test_article_id_stable_for_same_url_and_title() -> None:
    a = n._article_id("https://x.com/a", "Hello")
    b = n._article_id("https://x.com/a", "Hello")
    assert a == b
    assert len(a) == 32


def test_articles_to_df_maps_newsapi_shape() -> None:
    articles = [
        {
            "title": "T1",
            "description": "D1",
            "publishedAt": "2024-06-15T12:00:00Z",
            "url": "https://example.com/1",
            "source": {"name": "SRC"},
            "author": "A1",
            "content": None,
        }
    ]
    df = n._articles_to_df(articles, "AAPL")
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"
    assert df["headline"].iloc[0] == "T1"
    assert df["summary"].iloc[0] == "D1"
    assert df["source"].iloc[0] == "SRC"


def test_articles_to_df_empty() -> None:
    df = n._articles_to_df([], "MSFT")
    assert df.empty
    assert "article_id" in df.columns
