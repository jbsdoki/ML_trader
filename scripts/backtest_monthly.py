#!/usr/bin/env python3
"""
Walk-forward backtest: monthly XGBoost -> symmetric τ policy -> net returns with costs.

For each fold, fits a fresh regressor on the training window, predicts the held-out
month, maps the prediction to ``{-1,0,+1}`` via :func:`policy.threshold.pred_to_action`,
and scores :func:`backtest.simulator.period_net_return` against realized
``forward_return`` (default **10 bps** round-trip when non-flat).

Example::

    python scripts/backtest_monthly.py --symbols AAPL --model-id finbert \\
        --bar-source alpaca --taus 0,0.005,0.01,0.02
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def _parse_taus(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward monthly XGBoost backtest with τ sweep and costs.",
    )
    parser.add_argument("--symbols", type=str, default="AAPL")
    parser.add_argument("--model-id", type=str, default="finbert")
    parser.add_argument("--bar-interval", type=str, default="1d")
    parser.add_argument("--bar-source", type=str, default="alpaca")
    parser.add_argument("--bar-start", type=str, default=None)
    parser.add_argument("--bar-end", type=str, default=None)
    parser.add_argument("--anchor-start", type=str, default=None)
    parser.add_argument("--anchor-end", type=str, default=None)
    parser.add_argument("--forward-sessions", type=int, default=21)
    parser.add_argument(
        "--taus",
        type=str,
        default="0,0.005,0.01,0.02",
        help="Comma-separated τ values (fraction; symmetric dead zone)",
    )
    parser.add_argument("--min-train-rows", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=12)
    parser.add_argument("--max-history-days", type=int, default=366)
    parser.add_argument("--round-trip-cost-bps", type=float, default=10.0)
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

    from backtest.simulator import equity_multiple, period_net_return
    from backtest.walkforward import iter_monthly_walk_forward_splits
    from features.monthly_forward_frame import (
        build_monthly_forward_frame,
        default_monthly_training_feature_columns,
    )
    from models.xgboost import XGBoostRegressorModel
    from policy.threshold import pred_to_action
    from storage.database import connect
    from storage.schema import init_schema

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        log.error("Provide at least one symbol")
        return 1

    taus = _parse_taus(args.taus)
    if not taus:
        log.error("Provide at least one τ in --taus")
        return 1
    for tau in taus:
        if tau < 0:
            log.error("τ values must be >= 0, got %s", taus)
            return 1

    if args.forward_sessions < 1:
        log.error("--forward-sessions must be >= 1")
        return 1

    if args.min_train_rows < 1 or args.max_train_rows < args.min_train_rows:
        log.error("Invalid train window bounds")
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
        log.error("No labeled rows for backtest.")
        return 1

    feat_cols = default_monthly_training_feature_columns()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.error("Frame missing columns: %s", missing)
        return 1

    if df["forward_return"].isna().any():
        log.error("Frame has NaN forward_return; drop or fix data.")
        return 1

    max_hist = None if args.max_history_days < 0 else pd.Timedelta(days=int(args.max_history_days))

    folds = list(
        iter_monthly_walk_forward_splits(
            df,
            min_train_rows=args.min_train_rows,
            max_train_rows=args.max_train_rows,
            max_history=max_hist,
        )
    )
    if not folds:
        log.error(
            "No walk-forward folds (need more rows than min-train-rows=%s).",
            args.min_train_rows,
        )
        return 1

    series_by_tau: dict[float, list[float]] = {tau: [] for tau in taus}

    for train_df, test_row in folds:
        X_train = train_df[feat_cols].to_numpy(dtype=np.float64, copy=True)
        y_train = train_df["forward_return"].to_numpy(dtype=np.float64, copy=True)
        X_test = test_row[feat_cols].to_numpy(dtype=np.float64, copy=True).reshape(1, -1)
        realized = float(test_row["forward_return"])

        reg = XGBoostRegressorModel()
        reg.train(X_train, y_train)
        pred = float(reg.predict(X_test)[0])

        for tau in taus:
            act = pred_to_action(pred, tau)
            net = period_net_return(act, realized, round_trip_cost_bps=args.round_trip_cost_bps)
            series_by_tau[tau].append(net)

    print(f"folds={len(folds)} symbols={symbols} cost_bps={args.round_trip_cost_bps}")
    for tau in taus:
        arr = np.array(series_by_tau[tau], dtype=np.float64)
        mean_r = float(np.mean(arr))
        em = equity_multiple(arr)
        print(
            f"tau={tau:g} mean_net_return={mean_r:.6f} equity_multiple={em:.6f} n={len(arr)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
