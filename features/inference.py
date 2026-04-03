"""
Build feature rows for inference (aligned with :func:`default_training_feature_columns`).
"""

from __future__ import annotations

import sqlite3
from typing import Literal

import numpy as np
import pandas as pd

from .daily_sentiment_bars import build_daily_bars_sentiment_frame, build_sentiment_features_for_target_sessions
from .training_frame import default_training_feature_columns


def latest_bar_inference_frame(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    symbols: list[str],
    bar_interval: str = "1d",
    bar_source_api: str | None = "alpaca",
    published_start: str | None = None,
    published_end: str | None = None,
    bar_ts_start: str | None = None,
    bar_ts_end: str | None = None,
    sentiment_mode: Literal["open_cutoff", "article_session"] = "open_cutoff",
) -> pd.DataFrame:
    """
    Same join as training, then **one row per symbol**: the latest ``bar_ts`` (complete sessions).
    """
    df = build_daily_bars_sentiment_frame(
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
    if df.empty:
        return df
    df = df.sort_values(["symbol", "bar_ts"], kind="mergesort")
    return df.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


def inference_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Extract ``(X, column_names)`` in training order. Raises if columns missing.
    """
    cols = default_training_feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Inference frame missing columns {missing}; got {list(df.columns)}")
    X = df[cols].to_numpy(dtype=np.float64, copy=True)
    return X, cols


__all__ = [
    "build_sentiment_features_for_target_sessions",
    "inference_feature_matrix",
    "latest_bar_inference_frame",
]
