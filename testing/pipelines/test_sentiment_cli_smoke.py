"""``run_sentiment`` CLI: mocked DB + FinBERT for smoke test."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from storage.articles_repo import upsert_articles
from storage.schema import init_schema


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
    tmp_path: Path,
    sample_article_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Use a file-backed DB so ``main()`` can ``conn.close()`` and we still verify rows.
    (In-memory DBs are not shared across connections unless URI ``mode=memory`` is set.)
    """
    db_path = tmp_path / "sentiment_smoke.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    upsert_articles(conn, sample_article_df, "finnhub")
    conn.commit()
    conn.close()

    def fake_connect(**kwargs: Any) -> sqlite3.Connection:
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

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

    verify = sqlite3.connect(db_path)
    try:
        n = verify.execute(
            "SELECT COUNT(*) FROM article_sentiment WHERE model_id = 'finbert'"
        ).fetchone()[0]
    finally:
        verify.close()
    assert int(n) == 1
