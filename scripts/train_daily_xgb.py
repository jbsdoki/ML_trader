#!/usr/bin/env python3
"""
Train an XGBoost regressor on daily sentiment + open-to-close return (v1).

Default: single symbol AAPL, yfinance bars, FinBERT sentiment columns, time-ordered
train/test split. Sentiment NaNs are passed through (missing days with no news).

Example::

    python scripts/train_daily_xgb.py --symbols AAPL --model-id finbert \\
        --bar-source yfinance --train-frac 0.8 --save-model data_store/xgb_daily.json
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
        description="Train XGBoost regressor: sentiment features -> open-to-close return.",
    )
    parser.add_argument("--symbols", type=str, default="AAPL", help="Comma-separated tickers")
    parser.add_argument("--model-id", type=str, default="finbert", help="article_sentiment.model_id")
    parser.add_argument("--bar-interval", type=str, default="1d")
    parser.add_argument(
        "--bar-source",
        type=str,
        default="yfinance",
        help="bars.source_api filter (default: yfinance)",
    )
    parser.add_argument("--published-start", type=str, default=None)
    parser.add_argument("--published-end", type=str, default=None)
    parser.add_argument("--bar-start", type=str, default=None)
    parser.add_argument("--bar-end", type=str, default=None)
    parser.add_argument(
        "--sentiment-mode",
        type=str,
        choices=("open_cutoff", "article_session"),
        default="open_cutoff",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of rows (by bar_ts) for training; remainder is test",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Write XGBoost JSON model (native save_model)",
    )
    parser.add_argument("--no-init-schema", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger(__name__)

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        log.warning("python-dotenv not installed; .env not loaded")

    from features.training_frame import (
        build_daily_training_frame,
        default_training_feature_columns,
        time_series_split_by_bar_ts,
    )
    from models.xgboost import XGBoostRegressorModel
    from storage.database import connect
    from storage.schema import init_schema

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        log.error("Provide at least one symbol")
        return 1

    conn = connect()
    if not args.no_init_schema:
        init_schema(conn)

    df = build_daily_training_frame(
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

    if df.empty:
        log.error("No training rows (check bars, sentiment, and date filters).")
        return 1

    feat_cols = default_training_feature_columns()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.error("Frame missing columns: %s", missing)
        return 1

    train_df, test_df = time_series_split_by_bar_ts(df, train_frac=args.train_frac)

    X_train = train_df[feat_cols].to_numpy(dtype=np.float64, copy=True)
    y_train = train_df["target_return_oc"].to_numpy(dtype=np.float64, copy=True)
    X_test = test_df[feat_cols].to_numpy(dtype=np.float64, copy=True)
    y_test = test_df["target_return_oc"].to_numpy(dtype=np.float64, copy=True)

    reg = XGBoostRegressorModel()
    reg.train(X_train, y_train)

    print(f"train_rows={len(train_df)} test_rows={len(test_df)} features={feat_cols}")

    if len(test_df) == 0:
        log.warning("Empty test set; skip metrics")
        return 0

    pred = reg.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    std_y = float(np.std(y_test, ddof=0))
    std_p = float(np.std(pred, ddof=0))
    if len(y_test) >= 2 and std_y > 1e-12 and std_p > 1e-12:
        corr = float(np.corrcoef(pred, y_test)[0, 1])
    else:
        corr = float("nan")

    print(f"test_mae={mae:.6f} test_rmse={rmse:.6f} test_corr_pred_y={corr:.6f}")

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        reg.save(str(args.save_model))
        print(f"wrote {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
