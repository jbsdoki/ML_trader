#!/usr/bin/env python3
"""
Load a monthly XGBoost model and predict **forward return** from prior-month features.

By default builds rows with ``require_forward_return=False`` so the **latest**
month-end anchor is included even when future bars (the label) are not yet in
the database. Use ``--labeled-only`` to restrict to rows with a realized
``forward_return`` (backtest / QA).

Reads feature order from ``<model>.meta.json`` (written by ``train_monthly_xgb.py``)
or falls back to :func:`features.monthly_forward_frame.default_monthly_training_feature_columns`.

Example::

    python scripts/predict_monthly.py --model data_store/xgb_monthly.json \\
        --symbols AAPL --bar-source alpaca --tau 0.01
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


def _load_meta(model_path: Path) -> dict | None:
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _feature_columns(model_path: Path) -> tuple[list[str], int]:
    meta = _load_meta(model_path)
    if meta and "feature_columns" in meta:
        cols = list(meta["feature_columns"])
        fs = int(meta.get("forward_sessions", 21))
        return cols, fs
    from features.monthly_forward_frame import default_monthly_training_feature_columns

    return default_monthly_training_feature_columns(), 21


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict monthly forward return (XGBoost) + optional τ action.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to XGBoost JSON model from train_monthly_xgb.py",
    )
    parser.add_argument("--symbols", type=str, default="AAPL", help="Comma-separated tickers")
    parser.add_argument("--model-id", type=str, default="finbert")
    parser.add_argument("--bar-interval", type=str, default="1d")
    parser.add_argument("--bar-source", type=str, default="alpaca", help="bars.source_api (empty = any)")
    parser.add_argument("--bar-start", type=str, default=None)
    parser.add_argument("--bar-end", type=str, default=None)
    parser.add_argument("--anchor-start", type=str, default=None)
    parser.add_argument("--anchor-end", type=str, default=None)
    parser.add_argument(
        "--forward-sessions",
        type=int,
        default=None,
        help="Override label horizon (default: from meta or 21)",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Only rows with realized forward_return (require future bars in DB)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Emit every anchor row; default is latest month-end per symbol only",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="If set, append policy action {-1,0,1} for symmetric band",
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

    from features.monthly_forward_frame import build_monthly_forward_frame
    from models.xgboost import XGBoostRegressorModel
    from policy.threshold import pred_to_action
    from storage.database import connect
    from storage.schema import init_schema

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        log.error("Provide at least one symbol")
        return 1

    if not args.model.is_file():
        log.error("Model file not found: %s", args.model)
        return 1

    feat_cols, meta_fs = _feature_columns(args.model)
    forward_sessions = args.forward_sessions if args.forward_sessions is not None else meta_fs
    if forward_sessions < 1:
        log.error("forward-sessions must be >= 1")
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
        forward_sessions=forward_sessions,
        require_forward_return=args.labeled_only,
    )
    conn.close()

    if df.empty:
        log.error("No inference rows (check bars, sentiment, and filters).")
        return 1

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.error("Frame missing feature columns: %s", missing)
        return 1

    if args.all_rows:
        out = df.copy()
    else:
        out = (
            df.sort_values(["symbol", "anchor_nyse_session"])
            .groupby("symbol", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    X = out[feat_cols].to_numpy(dtype=np.float64, copy=True)
    reg = XGBoostRegressorModel()
    reg.load(str(args.model))
    pred = reg.predict(X)
    out = out.copy()
    out["predicted_forward_return"] = pred

    if args.tau is not None:
        tau = float(args.tau)
        out["action"] = [pred_to_action(float(p), tau) for p in pred]

    cols = ["symbol", "anchor_nyse_session", "predicted_forward_return"]
    if "forward_return" in out.columns:
        cols.append("forward_return")
    if "action" in out.columns:
        cols.append("action")
    print(out[cols].to_string(index=False))
    print(
        "pred_mean=%.6f pred_std=%.6f"
        % (float(np.mean(pred)), float(np.std(pred)) if len(pred) > 1 else 0.0)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
