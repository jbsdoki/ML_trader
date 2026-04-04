"""End-to-end: bars + sentiment + training frame (in-memory DB)."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from features.training_frame import build_daily_training_frame
from storage.articles_repo import upsert_articles
from storage.bars_repo import upsert_bars
from storage.schema import init_schema
from storage.sentiment_repo import upsert_article_sentiment


def _seed_minimal_aapl_session(sqlite_conn: sqlite3.Connection) -> None:
    """One Finnhub article, one sentiment row, one yfinance daily bar (same calendar idea)."""
    article = pd.DataFrame(
        [
            {
                "article_id": "int-test-1",
                "datetime": "2024-06-14T20:00:00+00:00",
                "headline": "Integration test",
                "summary": "Body",
                "source": "Test",
                "url": "https://example.com/int1",
                "author": None,
                "category": None,
                "related": None,
                "image": None,
                "content_snippet": None,
                "symbol": "AAPL",
            }
        ]
    )
    upsert_articles(sqlite_conn, article, "finnhub")
    dk = pd.read_sql_query("SELECT dedupe_key FROM articles LIMIT 1", sqlite_conn).iloc[0, 0]
    upsert_article_sentiment(
        sqlite_conn,
        [
            {
                "dedupe_key": str(dk),
                "model_id": "finbert",
                "score": 0.1,
                "prob_pos": 0.5,
                "prob_neg": 0.2,
                "prob_neutral": 0.3,
                "text_hash": "x",
                "error": None,
            }
        ],
    )
    ts = pd.Timestamp("2024-06-15T13:30:00", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.25],
            "volume": [1e6],
        }
    )
    upsert_bars(sqlite_conn, bars, "yfinance", "1d")


def test_build_daily_training_frame_article_session_non_empty() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    _seed_minimal_aapl_session(conn)
    df = build_daily_training_frame(
        conn,
        model_id="finbert",
        symbols=["AAPL"],
        bar_interval="1d",
        bar_source_api="yfinance",
        sentiment_mode="article_session",
    )
    conn.close()
    assert len(df) >= 1
    assert "target_return_oc" in df.columns
    assert df["target_return_oc"].iloc[0] == pytest.approx(0.0025)
