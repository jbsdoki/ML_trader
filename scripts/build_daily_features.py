#!/usr/bin/env python3
"""
Build a daily table: OHLC bars + sentiment aggregates (left join).

Default sentiment uses articles from **prior session close through before
target session open** (NYSE), so overnight / post-close news rolls into the
next session's bar row. Use ``--sentiment-mode article_session`` for the
legacy same-calendar-session bucket.

Example::

    python scripts/build_daily_features.py --symbols AAPL,MSFT --model-id finbert \\
        --bar-source alpaca --out data_store/daily_features.csv

Parquet (``.parquet``) is supported if ``pyarrow`` is installed. Requires
``exchange_calendars`` (see requirements.txt).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


def _parse_symbols(raw: str) -> list[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join daily bars to NYSE-session sentiment aggregates.",
    )
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated tickers")
    parser.add_argument("--model-id", type=str, default="finbert", help="article_sentiment.model_id")
    parser.add_argument("--bar-interval", type=str, default="1d", help="bars.bar_interval filter")
    parser.add_argument(
        "--bar-source",
        type=str,
        default=None,
        help="Optional bars source_api (yfinance, alpaca, ...) to avoid duplicate vendors",
    )
    parser.add_argument("--published-start", type=str, default=None, help="Lower bound articles.published_at")
    parser.add_argument("--published-end", type=str, default=None, help="Upper bound articles.published_at")
    parser.add_argument("--bar-start", type=str, default=None, help="Lower bound bars.bar_ts")
    parser.add_argument("--bar-end", type=str, default=None, help="Upper bound bars.bar_ts")
    parser.add_argument(
        "--sentiment-mode",
        type=str,
        choices=("open_cutoff", "article_session"),
        default="open_cutoff",
        help="open_cutoff: close(T-1)<=pub<open(T); article_session: bucket by article day",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write CSV or Parquet (.parquet needs pyarrow); omit to print row count only",
    )
    parser.add_argument("--no-init-schema", action="store_true", help="Skip CREATE TABLE if missing")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        logging.getLogger(__name__).warning("python-dotenv not installed; .env not loaded")

    from features.daily_sentiment_bars import build_daily_bars_sentiment_frame
    from storage.database import connect
    from storage.schema import init_schema

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        logging.error("Provide at least one symbol")
        return 1

    conn = connect()
    if not args.no_init_schema:
        init_schema(conn)

    df = build_daily_bars_sentiment_frame(
        conn,
        model_id=args.model_id.strip(),
        symbols=symbols,
        bar_interval=args.bar_interval.strip(),
        bar_source_api=args.bar_source.strip() if args.bar_source else None,
        published_start=args.published_start,
        published_end=args.published_end,
        bar_ts_start=args.bar_start,
        bar_ts_end=args.bar_end,
        sentiment_mode=args.sentiment_mode,
    )
    conn.close()

    print(f"rows={len(df)}")
    if df.empty:
        return 0

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        suf = args.out.suffix.lower()
        if suf == ".parquet":
            try:
                df.to_parquet(args.out, index=False)
            except ImportError:
                logging.error("Parquet requires pyarrow: pip install pyarrow")
                return 1
        else:
            df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
