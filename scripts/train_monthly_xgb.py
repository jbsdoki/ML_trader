#!/usr/bin/env python3
"""
Train an XGBoost regressor on **month-end** rows: prior-month features -> forward return.

Uses :func:`features.monthly_forward_frame.build_monthly_forward_frame` (default
21 NYSE sessions forward). Writes ``--save-model`` JSON plus a sidecar
``*.meta.json`` with feature names and training settings for
``predict_monthly.py``.

Example::

    python scripts/train_monthly_xgb.py --symbols AAPL --model-id finbert \\
        --bar-source alpaca --forward-sessions 21 --train-frac 0.8 \\
        --save-model data_store/xgb_monthly.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_symbols(raw: str) -> list[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train XGBoost regressor: monthly prior-month features -> forward return.",
    )
    parser.add_argument("--symbols", type=str, default="AAPL", help="Comma-separated tickers")
    parser.add_argument("--model-id", type=str, default="finbert", help="article_sentiment.model_id")
    parser.add_argument("--bar-interval", type=str, default="1d")
    parser.add_argument(
        "--bar-source",
        type=str,
        default="alpaca",
        help="bars.source_api filter (empty string = any source)",
    )
    parser.add_argument("--bar-start", type=str, default=None)
    parser.add_argument("--bar-end", type=str, default=None)
    parser.add_argument("--anchor-start", type=str, default=None)
    parser.add_argument("--anchor-end", type=str, default=None)
    parser.add_argument(
        "--forward-sessions",
        type=int,
        default=21,
        help="Label horizon: NYSE sessions after month-end anchor",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of rows (by anchor time) for training; remainder is test",
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

    from features.monthly_forward_frame import (
        build_monthly_forward_frame,
        default_monthly_training_feature_columns,
        time_series_split_by_anchor,
    )
    from models.xgboost import XGBoostRegressorModel
    from storage.database import connect
    from storage.schema import init_schema

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        log.error("Provide at least one symbol")
        return 1

    if args.forward_sessions < 1:
        log.error("--forward-sessions must be >= 1")
        return 1

    bar_src = args.bar_source.strip() if args.bar_source and args.bar_source.strip() else None

    conn = connect()
    if not args.no_init_schema:
        init_schema(conn)

    df = build_monthly_forward_frame(
        conn,
        symbols=symbols,
        model_id=args.model_id.strip(),
        bar_interval=args.bar_interval.strip(),
        bar_source_api=bar_src,
        bar_ts_start=args.bar_start,
        bar_ts_end=args.bar_end,
        anchor_session_start=args.anchor_start,
        anchor_session_end=args.anchor_end,
        forward_sessions=args.forward_sessions,
        require_forward_return=True,
    )
    conn.close()

    if df.empty:
        log.error("No training rows (check bars, sentiment, horizon, and date filters).")
        return 1

    feat_cols = default_monthly_training_feature_columns()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.error("Frame missing columns: %s", missing)
        return 1

    if df["forward_return"].isna().any():
        log.error("Labeled frame contains NaN forward_return; check data quality.")
        return 1

    try:
        train_df, test_df = time_series_split_by_anchor(df, train_frac=args.train_frac)
    except ValueError as e:
        log.error("%s", e)
        return 1

    X_train = train_df[feat_cols].to_numpy(dtype=np.float64, copy=True)
    y_train = train_df["forward_return"].to_numpy(dtype=np.float64, copy=True)
    X_test = test_df[feat_cols].to_numpy(dtype=np.float64, copy=True)
    y_test = test_df["forward_return"].to_numpy(dtype=np.float64, copy=True)

    reg = XGBoostRegressorModel()
    reg.train(X_train, y_train)

    print(f"train_rows={len(train_df)} test_rows={len(test_df)} features={feat_cols}")

    if len(test_df) == 0:
        log.warning("Empty test set; skip metrics")
    else:
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
        meta_path = args.save_model.with_suffix(".meta.json")
        _write_meta(
            meta_path,
            {
                "schema_version": 1,
                "feature_columns": feat_cols,
                "target_column": "forward_return",
                "forward_sessions": args.forward_sessions,
                "bar_interval": args.bar_interval.strip(),
                "bar_source_api": bar_src,
                "symbols_trained": symbols,
                "model_id": args.model_id.strip(),
            },
        )
        print(f"wrote {args.save_model}")
        print(f"wrote {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
