"""Inference feature builders and matrix extraction."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from features.daily_sentiment_bars import build_sentiment_features_for_target_sessions
from features.inference import inference_feature_matrix, latest_bar_inference_frame
from storage.articles_repo import upsert_articles
from storage.bars_repo import upsert_bars
from storage.schema import init_schema
from storage.sentiment_repo import upsert_article_sentiment


def _seed_article_sentiment_bar(conn: sqlite3.Connection) -> None:
    article = pd.DataFrame(
        [
            {
                "article_id": "inf-1",
                "datetime": "2024-06-14T18:00:00+00:00",
                "headline": "H",
                "summary": "S",
                "source": "T",
                "url": "https://example.com/inf",
                "author": None,
                "category": None,
                "related": None,
                "image": None,
                "content_snippet": None,
                "symbol": "AAPL",
            }
        ]
    )
    upsert_articles(conn, article, "finnhub")
    dk = pd.read_sql_query("SELECT dedupe_key FROM articles LIMIT 1", conn).iloc[0, 0]
    upsert_article_sentiment(
        conn,
        [
            {
                "dedupe_key": str(dk),
                "model_id": "finbert",
                "score": 0.2,
                "prob_pos": 0.5,
                "prob_neg": 0.3,
                "prob_neutral": 0.2,
                "text_hash": "h",
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
            "low": [99.0],
            "close": [100.5],
            "volume": [1e6],
        }
    )
    upsert_bars(conn, bars, "yfinance", "1d")


def test_build_sentiment_features_for_target_sessions_has_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    _seed_article_sentiment_bar(conn)
    out = build_sentiment_features_for_target_sessions(
        conn,
        model_id="finbert",
        symbols=["AAPL"],
        nyse_session="2024-06-15",
        sentiment_mode="article_session",
    )
    conn.close()
    assert list(out.columns) == ["symbol", "nyse_session", "sentiment_mean", "sentiment_n", "sentiment_std"]
    assert len(out) == 1


def test_inference_feature_matrix_shape() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "sentiment_mean": [0.1],
            "sentiment_n": [2.0],
            "sentiment_std": [0.05],
        }
    )
    X, cols = inference_feature_matrix(df)
    assert cols == ["sentiment_mean", "sentiment_n", "sentiment_std"]
    assert X.shape == (1, 3)


def test_inference_feature_matrix_missing_column_raises() -> None:
    df = pd.DataFrame({"symbol": ["AAPL"], "sentiment_mean": [0.1]})
    with pytest.raises(ValueError, match="missing columns"):
        inference_feature_matrix(df)


def test_latest_bar_inference_frame_roundtrip() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    _seed_article_sentiment_bar(conn)
    out = latest_bar_inference_frame(
        conn,
        model_id="finbert",
        symbols=["AAPL"],
        bar_source_api="yfinance",
        sentiment_mode="article_session",
    )
    conn.close()
    assert len(out) == 1
    assert "bar_ts" in out.columns
