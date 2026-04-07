"""``run_sentiment`` CLI: mocked DB + FinBERT for smoke test."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from storage.articles_repo import upsert_articles


def _load_run_sentiment_module():
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "scripts" / "run_sentiment.py"
    spec = importlib.util.spec_from_file_location("run_sentiment_cli", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _FakeFinBERTScorer:
    def __init__(self, device=None) -> None:
        pass

    def score_texts(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        return [
            {
                "score": 0.0,
                "prob_pos": 0.25,
                "prob_neg": 0.25,
                "prob_neutral": 0.5,
                "error": None,
            }
            for _ in texts
        ]


def test_run_sentiment_main_smoke(
    sqlite_conn: sqlite3.Connection,
    sample_article_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_articles(sqlite_conn, sample_article_df, "finnhub")

    def fake_connect(**kwargs):
        return sqlite_conn

    monkeypatch.setattr("storage.database.connect", fake_connect)
    monkeypatch.setattr("sentiment.finbert_scorer.FinBERTScorer", _FakeFinBERTScorer)

    mod = _load_run_sentiment_module()
    rc = mod.main(
        [
            "--symbols",
            "AAPL",
            "--limit",
            "10",
            "--no-init-schema",
        ]
    )
    assert rc == 0
    n = sqlite_conn.execute(
        "SELECT COUNT(*) FROM article_sentiment WHERE model_id = 'finbert'"
    ).fetchone()[0]
    assert int(n) == 1
