#!/usr/bin/env python3
"""
Load a trained XGBoost JSON model and predict open-to-close return from sentiment features.

**latest_bar** — one row per symbol: most recent complete daily bar + aligned sentiment
(same as training rows; use after the session close).

**session** — sentiment only for a target NYSE session date (no bar required; use at open
before that day's bar exists).

Examples::

    python scripts/predict_daily.py --model data_store/xgb_daily.json --mode latest_bar \\
        --symbols AAPL --bar-source alpaca

    python scripts/predict_daily.py --model data_store/xgb_daily.json --mode session \\
        --session-date 2025-03-28 --symbols AAPL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

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
        description="Predict open-to-close return using saved XGBoost regressor + sentiment features.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to XGBoost JSON model (from train_daily_xgb.py --save-model)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("latest_bar", "session"),
        default="latest_bar",
        help="latest_bar: last complete bar per symbol; session: sentiment for --session-date only",
    )
    parser.add_argument("--symbols", type=str, default="AAPL", help="Comma-separated tickers")
    parser.add_argument("--model-id", type=str, default="finbert", help="article_sentiment.model_id")
    parser.add_argument("--bar-interval", type=str, default="1d")
    parser.add_argument("--bar-source", type=str, default="alpaca", help="bars.source_api")
    parser.add_argument("--published-start", type=str, default=None)
    parser.add_argument("--published-end", type=str, default=None)
    parser.add_argument("--bar-start", type=str, default=None)
    parser.add_argument("--bar-end", type=str, default=None)
    parser.add_argument(
        "--session-date",
        type=str,
        default=None,
        help="NYSE session date (YYYY-MM-DD) for --mode session",
    )
    parser.add_argument(
        "--sentiment-mode",
        type=str,
        choices=("open_cutoff", "article_session"),
        default="open_cutoff",
    )
    parser.add_argument("--no-init-schema", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger(__name__)

    if args.mode == "session" and not args.session_date:
        log.error("--session-date is required when --mode session")
        return 1

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        log.warning("python-dotenv not installed; .env not loaded")

    from features.inference import (
        build_sentiment_features_for_target_sessions,
        inference_feature_matrix,
        latest_bar_inference_frame,
    )
    from models.xgboost import XGBoostRegressorModel
    from storage.database import connect
    from storage.schema import init_schema

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        log.error("Provide at least one symbol")
        return 1

    if not args.model.is_file():
        log.error("Model file not found: %s", args.model)
        return 1

    conn = connect()
    if not args.no_init_schema:
        init_schema(conn)

    if args.mode == "latest_bar":
        feat_df = latest_bar_inference_frame(
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
    else:
        feat_df = build_sentiment_features_for_target_sessions(
            conn,
            model_id=args.model_id.strip(),
            symbols=symbols,
            nyse_session=args.session_date.strip(),
            published_start=args.published_start,
            published_end=args.published_end,
            sentiment_mode=args.sentiment_mode,
        )

    conn.close()

    if feat_df.empty:
        log.error("No inference rows (check bars, sentiment, filters, or session date).")
        return 1

    try:
        X, cols = inference_feature_matrix(feat_df)
    except ValueError as e:
        log.error("%s", e)
        return 1

    reg = XGBoostRegressorModel()
    reg.load(str(args.model))
    pred = reg.predict(X)

    meta_cols = [c for c in ("symbol", "bar_ts", "nyse_session") if c in feat_df.columns]
    out = feat_df.loc[:, meta_cols].copy()
    out["predicted_return_oc"] = pred

    print("features:", cols)
    print(out.to_string(index=False))
    print(
        "pred_mean=%.6f pred_std=%.6f"
        % (float(np.mean(pred)), float(np.std(pred)) if len(pred) > 1 else 0.0)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
