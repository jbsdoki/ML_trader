"""
Assemble a supervised frame: daily bars + sentiment (open-cutoff) + intraday return label.
"""

from __future__ import annotations

import sqlite3
from typing import Literal

import pandas as pd

from .daily_sentiment_bars import build_daily_bars_sentiment_frame
from .training_labels import add_open_to_close_return


def default_training_feature_columns() -> list[str]:
    """Feature names for v1 daily sentiment + activity."""
    return ["sentiment_mean", "sentiment_n", "sentiment_std"]


def build_daily_training_frame(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    symbols: list[str],
    bar_interval: str = "1d",
    bar_source_api: str | None = "yfinance",
    published_start: str | None = None,
    published_end: str | None = None,
    bar_ts_start: str | None = None,
    bar_ts_end: str | None = None,
    sentiment_mode: Literal["open_cutoff", "article_session"] = "open_cutoff",
) -> pd.DataFrame:
    """
    Bars joined to sentiment, plus ``target_return_oc`` = open-to-close return same session.

    Drops rows with invalid OHLC for the label. Does not impute sentiment (XGBoost
    can consume NaN in ``sentiment_*`` when enabled).
    """
    base = build_daily_bars_sentiment_frame(
        conn,
        model_id=model_id,
        symbols=symbols,
        bar_interval=bar_interval,
        bar_source_api=bar_source_api,
        published_start=published_start,
        published_end=published_end,
        bar_ts_start=bar_ts_start,
        bar_ts_end=bar_ts_end,
        sentiment_mode=sentiment_mode,
    )
    if base.empty:
        return base

    labeled = add_open_to_close_return(base)
    o = pd.to_numeric(labeled["open"], errors="coerce")
    c = pd.to_numeric(labeled["close"], errors="coerce")
    ok = o.notna() & c.notna() & (o > 0) & labeled["target_return_oc"].notna()
    return labeled.loc[ok].reset_index(drop=True)


def time_series_split_by_bar_ts(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    bar_ts_col: str = "bar_ts",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on ``bar_ts`` (sorted). Last ``1 - train_frac`` is test.
    """
    if df.empty:
        return df.copy(), df.copy()
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1).")
    ordered = df.sort_values(bar_ts_col, kind="mergesort")
    n = len(ordered)
    cut = max(1, min(n - 1, int(n * train_frac)))
    train = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()
    return train, test
