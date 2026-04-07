"""Articles repo: ``article_dedupe_key`` edge cases (fallback when id/url missing)."""

from __future__ import annotations

import pandas as pd

from storage.articles_repo import article_dedupe_key


def test_article_dedupe_key_fallback_headline_and_time() -> None:
    row = pd.Series(
        {
            "article_id": None,
            "datetime": "2024-06-01T10:00:00+00:00",
            "headline": "Breaking story",
            "url": None,
        }
    )
    k = article_dedupe_key("finnhub", row)
    assert k.startswith("finnhub:fallback:")
    assert "Breaking story" in k


def test_article_dedupe_key_prefers_numeric_id_string() -> None:
    row = pd.Series(
        {
            "article_id": 42,
            "datetime": "2024-01-01",
            "headline": "H",
        }
    )
    k = article_dedupe_key("finnhub", row)
    assert k == "finnhub:id:42"
