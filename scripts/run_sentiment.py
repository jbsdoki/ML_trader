#!/usr/bin/env python3
"""
CLI: score ingested articles with FinBERT and upsert into ``article_sentiment``.

Run from the **repository root**::

    python scripts/run_sentiment.py --symbols AAPL,MSFT --limit 500

Optional date bounds (same ISO handling as ``storage.sentiment_repo``)::

    python scripts/run_sentiment.py --start 2025-01-01 --end 2025-03-01

**Cron / VPS:** use the venv interpreter, absolute path, and set ``ML_TRADER_DATA_DIR``
so the DB path is stable (see ``storage/database.py``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_symbols_csv(raw: str | None) -> list[str] | None:
    if not raw or not str(raw).strip():
        return None
    out = [x.strip().upper() for x in str(raw).split(",") if x.strip()]
    return out or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run FinBERT on articles missing sentiment (by default) and upsert results.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="finbert",
        help="Stored in article_sentiment.model_id (default: finbert).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated tickers; omit to include all symbols.",
    )
    parser.add_argument("--start", type=str, default=None, help="Lower bound on published_at (YYYY-MM-DD or ISO)")
    parser.add_argument("--end", type=str, default=None, help="Upper bound on published_at")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Also score articles that already have a row for --model-id (default: only missing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to process (after SQL filters).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="FinBERT inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device: cuda, cpu, or omit for auto.',
    )
    parser.add_argument(
        "--no-init-schema",
        action="store_true",
        help="Skip CREATE TABLE (use if schema already applied).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger(__name__)

    # Load ML_TRADER_DATA_DIR and API keys from .env when present.
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        log.warning("python-dotenv not installed; .env not loaded")

    # Open SQLite and ensure tables exist unless the user opted out.
    from storage.database import connect
    from storage.schema import init_schema
    from storage.sentiment_repo import (
        build_finbert_input_text,
        fetch_articles_for_sentiment,
        text_hash_for_article,
        upsert_article_sentiment,
    )

    conn = connect()
    if not args.no_init_schema:
        init_schema(conn)

    model_id = args.model_id.strip()
    symbols = _parse_symbols_csv(args.symbols)

    # Pull candidate rows: by default only articles without sentiment for this model_id.
    df = fetch_articles_for_sentiment(
        conn,
        model_id=model_id,
        symbols=symbols,
        start=args.start,
        end=args.end,
        only_missing=not args.rescore,
        limit=args.limit,
    )

    if df.empty:
        log.info("No articles to score (filters returned 0 rows).")
        conn.close()
        return 0

    # Build one string per article (headline + summary) for the model and for hashing.
    texts: list[str] = []
    for _, row in df.iterrows():
        raw = build_finbert_input_text(row.get("headline"), row.get("summary"))
        texts.append((raw or "").strip())

    # Load FinBERT once and run batched forward passes.
    from sentiment.finbert_scorer import FinBERTScorer

    try:
        scorer = FinBERTScorer(device=args.device)
        scored = scorer.score_texts(texts, batch_size=max(1, int(args.batch_size)))
    except ImportError as e:
        log.error("%s", e)
        conn.close()
        return 1

    # Align DB keys and metadata with model outputs for upsert.
    upsert_rows: list[dict[str, Any]] = []
    for i, (_, article_row) in enumerate(df.iterrows()):
        text = texts[i]
        r = scored[i]
        upsert_rows.append(
            {
                "dedupe_key": str(article_row["dedupe_key"]),
                "model_id": model_id,
                "score": r.get("score"),
                "prob_pos": r.get("prob_pos"),
                "prob_neg": r.get("prob_neg"),
                "prob_neutral": r.get("prob_neutral"),
                "text_hash": text_hash_for_article(text) if text else None,
                "error": r.get("error"),
            }
        )

    # Persist (ON CONFLICT updates scores and text_hash for this model_id).
    n = upsert_article_sentiment(conn, upsert_rows)
    conn.close()

    print(f"Scored articles: {len(upsert_rows)}")
    print(f"Upsert operations (row changes): {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
