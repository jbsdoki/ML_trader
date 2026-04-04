"""Finnhub ingest: record normalization (no live API)."""

from __future__ import annotations

import pandas as pd

from data_retrieval import finnhub_ingest as fh


def test_news_records_to_df_empty() -> None:
    df = fh._news_records_to_df([], "AAPL")
    assert df.empty
    assert "article_id" in df.columns


def test_news_records_to_df_one_row_unix_ts() -> None:
    records = [
        {
            "id": 99,
            "datetime": 1718448000,
            "headline": "H",
            "summary": "S",
            "source": "Finnhub",
            "url": "https://example.com/n",
            "category": None,
            "image": None,
            "related": None,
        }
    ]
    df = fh._news_records_to_df(records, "aapl")
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"
    assert df["article_id"].iloc[0] == 99
    assert pd.notna(df["datetime"].iloc[0])
