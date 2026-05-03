"""
Shared fixtures: repo-root import path, in-memory SQLite with schema.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

@pytest.fixture
def sqlite_conn() -> sqlite3.Connection:
    """In-memory DB with articles, bars, article_sentiment tables."""
    from storage.schema import init_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_article_df() -> pd.DataFrame:
    """Minimal Finnhub-shaped row for articles_repo.upsert_articles."""
    return pd.DataFrame(
        [
            {
                "article_id": "test-article-1",
                "datetime": "2024-06-15T14:00:00+00:00",
                "headline": "Test headline",
                "summary": "Test summary",
                "source": "TestSource",
                "url": "https://example.com/a",
                "author": None,
                "category": None,
                "related": None,
                "image": None,
                "content_snippet": None,
                "symbol": "AAPL",
            }
        ]
    )


@pytest.fixture
def sample_bars_df() -> pd.DataFrame:
    """One daily bar row for bars_repo.upsert_bars (OHLCV + timestamp)."""
    ts = pd.Timestamp("2024-06-15T13:30:00", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["AAPL"],
            "open": [180.0],
            "high": [182.0],
            "low": [179.0],
            "close": [181.5],
            "volume": [1_000_000.0],
        }
    )
