#!/usr/bin/env python3
"""
CLI entrypoint: **ingest** news and OHLCV into SQLite (deduped).

Run from the **repository root** so imports resolve, e.g.::

    python scripts/run_ingest.py --symbols AAPL,MSFT --start 2025-01-01 --end 2025-03-18

Or with YAML (requires PyYAML)::

    python scripts/run_ingest.py --config config.yaml

**Cron / VPS:** use the venv interpreter and an absolute path; set ``ML_TRADER_DATA_DIR``
in ``.env`` so the DB path does not depend on cron's working directory.

Useful links
------------
- ``argparse`` tutorial: https://docs.python.org/3/howto/argparse.html
- ``python-dotenv``: https://pypi.org/project/python-dotenv/
- SQLite file location (``ML_TRADER_DATA_DIR``): see ``storage/database.py`` docstring.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repo root on sys.path when executing ``python scripts/run_ingest.py``
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _configure_logging(verbose: bool) -> None:
    """Configure root logger for console output (level INFO or DEBUG)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """
    Parse CLI arguments, load ``.env``, run :func:`pipelines.ingest_pipeline.run_ingest_pipeline`.

    Returns
    -------
    int
        ``0`` on success (errors may still be partial; check stderr and exit code policy).
        ``1`` if arguments invalid or YAML load fails.
        ``2`` if summary contains errors (API failures); still ``0`` if you prefer lenient — here we use 2.
    """
    parser = argparse.ArgumentParser(
        description="Fetch market/news data and persist to SQLite (deduped upserts).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file with symbols, start, end, bar_interval, sources (overrides other flags if set).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (inclusive for news APIs)")
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="OHLCV bar interval for yfinance/Alpaca (e.g. 1d, 1h)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="finnhub,newsapi,yfinance,alpaca",
        help="Comma-separated: finnhub, newsapi, yfinance, alpaca",
    )
    parser.add_argument(
        "--newsapi-extra-query",
        type=str,
        default=None,
        help="Optional OR-query fragment for NewsAPI fetch_for_symbol (e.g. company name)",
    )
    parser.add_argument(
        "--newsapi-page-size",
        type=int,
        default=100,
        help="NewsAPI page size (max 100).",
    )
    parser.add_argument(
        "--newsapi-max-pages",
        type=int,
        default=1,
        help="NewsAPI pages to request. Default 1 is free-tier friendly.",
    )
    parser.add_argument(
        "--alpaca-feed",
        type=str,
        default=None,
        help="Optional Alpaca data feed (e.g. sip, iex)",
    )
    parser.add_argument(
        "--no-init-schema",
        action="store_true",
        help="Do not run CREATE TABLE (use only if schema already applied)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        logging.getLogger(__name__).warning("python-dotenv not installed; .env not loaded")

    from pipelines.ingest_pipeline import (
        IngestConfig,
        load_ingest_config_yaml,
        normalize_symbols,
        parse_sources_csv,
        run_ingest_pipeline,
    )

    if args.config is not None:
        if not args.config.is_file():
            logging.error("Config file not found: %s", args.config)
            return 1
        try:
            config = load_ingest_config_yaml(args.config)
        except Exception as e:
            logging.exception("Failed to load config: %s", e)
            return 1
        # Allow CLI to override sources when both given? Keep YAML-only for config path.
    else:
        if not args.symbols or not args.start or not args.end:
            logging.error("Provide --config PATH or all of --symbols, --start, --end")
            return 1
        flags = parse_sources_csv(args.sources)
        config = IngestConfig(
            symbols=normalize_symbols(
                [x.strip() for x in args.symbols.split(",") if x.strip()]
            ),
            start=args.start.strip()[:10],
            end=args.end.strip()[:10],
            bar_interval=args.interval.strip(),
            finnhub=flags["finnhub"],
            newsapi=flags["newsapi"],
            yfinance=flags["yfinance"],
            alpaca=flags["alpaca"],
            alpaca_feed=args.alpaca_feed,
            newsapi_extra_query=args.newsapi_extra_query,
            # Improvement: expose paging knobs in CLI so free-tier users can avoid
            # page-2 errors without editing Python modules.
            newsapi_page_size=max(1, min(int(args.newsapi_page_size), 100)),
            newsapi_max_pages=max(1, int(args.newsapi_max_pages)),
        )

    summary = run_ingest_pipeline(config, init_db=not args.no_init_schema)

    print("Ingest summary:")
    print(f"  finnhub_articles_inserted={summary.articles_finnhub_inserted}")
    print(f"  newsapi_articles_inserted={summary.articles_newsapi_inserted}")
    print(f"  yfinance_bars_inserted={summary.bars_yfinance_inserted}")
    print(f"  alpaca_bars_inserted={summary.bars_alpaca_inserted}")
    print(f"  error_count={len(summary.errors)}")
    for err in summary.errors:
        print(f"  - {err}")

    return 2 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
