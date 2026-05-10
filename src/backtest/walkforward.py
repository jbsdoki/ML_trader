"""
Time-ordered **walk-forward** splits for **monthly** rows.

The input frame is expected to be sorted by a monotonic **anchor** timestamp
(typically ``anchor_nyse_session`` from ``features.monthly_forward_frame``). For
each test row *t*, training rows are all strictly earlier anchors, optionally
restricted to a rolling **history cap** (e.g. one year before *t*), then
truncated to the **most recent** ``max_train_rows`` months so folds stay small
and comparable to a 6–12 month training window.

This module does **not** fit models; it only yields ``(train_df, test_row)``
pairs for a caller (script or test) to train and score.
"""

from __future__ import annotations

from typing import Any, Iterator

import pandas as pd


def _sorted_frame(df: pd.DataFrame, anchor_col: str) -> pd.DataFrame:
    return df.sort_values(anchor_col).reset_index(drop=True)


def _coerce_anchor_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _train_window_indices(
    work: pd.DataFrame,
    test_idx: int,
    *,
    anchor_col: str,
    min_train_rows: int,
    max_train_rows: int,
    max_history: pd.Timedelta | None,
) -> pd.Index | None:
    test_anchor = pd.Timestamp(work.at[test_idx, anchor_col])
    pool = work.loc[: test_idx - 1]
    if pool.empty:
        return None
    if max_history is not None:
        cutoff = test_anchor - max_history
        mask = _coerce_anchor_series(pool[anchor_col]) >= cutoff
        pool = pool.loc[mask]
    if len(pool) < min_train_rows:
        return None
    if len(pool) > max_train_rows:
        pool = pool.iloc[-max_train_rows:]
    return pool.index


def iter_monthly_walk_forward_splits(
    df: pd.DataFrame,
    *,
    anchor_col: str = "anchor_nyse_session",
    min_train_rows: int = 6,
    max_train_rows: int = 12,
    max_history: pd.Timedelta | None = pd.Timedelta(days=366),
) -> Iterator[tuple[pd.DataFrame, pd.Series]]:
    """
    Yield ``(train_df, test_row)`` for expanding / rolling monthly backtests.

    Parameters
    ----------
    df
        One row per (symbol, month) or pooled symbols; must include ``anchor_col``.
    anchor_col
        Column used for chronological ordering and ``max_history`` cutoff.
    min_train_rows
        Skip folds until at least this many training rows remain after filters.
    max_train_rows
        After the history cap, keep only the **last** (most recent) this many rows.
    max_history
        Drop training rows older than this span before the test anchor. ``None``
        keeps all prior rows (subject to ``max_train_rows``).

    Yields
    ------
    train_df
        Training subset for this fold (copy).
    test_row
        A single-row ``Series`` (``iloc[i]``) for the held-out month.
    """
    if min_train_rows < 1 or max_train_rows < min_train_rows:
        return
    if df.empty or anchor_col not in df.columns:
        return

    work = _sorted_frame(df, anchor_col)
    n = len(work)
    for i in range(min_train_rows, n):
        idx = _train_window_indices(
            work,
            i,
            anchor_col=anchor_col,
            min_train_rows=min_train_rows,
            max_train_rows=max_train_rows,
            max_history=max_history,
        )
        if idx is None:
            continue
        train_df = work.loc[idx].copy()
        yield train_df, work.iloc[i]


def fold_count(
    df: pd.DataFrame,
    *,
    anchor_col: str = "anchor_nyse_session",
    min_train_rows: int = 6,
    max_train_rows: int = 12,
    max_history: pd.Timedelta | None = pd.Timedelta(days=366),
) -> int:
    """Number of folds ``iter_monthly_walk_forward_splits`` would emit."""
    return sum(
        1
        for _ in iter_monthly_walk_forward_splits(
            df,
            anchor_col=anchor_col,
            min_train_rows=min_train_rows,
            max_train_rows=max_train_rows,
            max_history=max_history,
        )
    )


def split_train_test_by_anchor(
    df: pd.DataFrame,
    *,
    anchor_col: str,
    test_anchor_min: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single cut: train strictly before ``test_anchor_min``, test at or after.

    Useful for a simple holdout backtest without walking every month.
    """
    work = _sorted_frame(df, anchor_col)
    cut = pd.Timestamp(test_anchor_min)
    ts = _coerce_anchor_series(work[anchor_col])
    return work.loc[ts < cut].copy(), work.loc[ts >= cut].copy()
