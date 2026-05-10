"""Tests for ``backtest.walkforward`` split iterators."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.walkforward import (
    fold_count,
    iter_monthly_walk_forward_splits,
    split_train_test_by_anchor,
)


def _monthly_df(n: int, start: str = "2024-01-31") -> pd.DataFrame:
    anchors = pd.date_range(start=start, periods=n, freq="ME", normalize=True)
    return pd.DataFrame(
        {
            "anchor_nyse_session": anchors,
            "x": range(n),
        }
    )


def test_iter_monthly_walk_forward_splits_empty() -> None:
    assert list(iter_monthly_walk_forward_splits(pd.DataFrame())) == []
    assert list(iter_monthly_walk_forward_splits(pd.DataFrame({"a": [1]}))) == []


def test_iter_monthly_walk_forward_invalid_window_yields_nothing() -> None:
    df = _monthly_df(5)
    assert list(iter_monthly_walk_forward_splits(df, min_train_rows=10, max_train_rows=12)) == []


def test_iter_monthly_walk_forward_respects_min_train() -> None:
    df = _monthly_df(8)
    folds = list(
        iter_monthly_walk_forward_splits(
            df,
            min_train_rows=2,
            max_train_rows=12,
            max_history=None,
        )
    )
    assert len(folds) == 6
    train0, test0 = folds[0]
    assert len(train0) == 2
    assert test0["x"] == 2


def test_iter_monthly_walk_forward_max_train_rows_truncates() -> None:
    df = _monthly_df(10)
    folds = list(
        iter_monthly_walk_forward_splits(
            df,
            min_train_rows=2,
            max_train_rows=3,
            max_history=None,
        )
    )
    last_train, _ = folds[-1]
    assert len(last_train) == 3
    assert last_train["x"].tolist() == [6, 7, 8]


def test_iter_monthly_walk_forward_max_history_filters_old_rows() -> None:
    df = pd.DataFrame(
        {
            "anchor_nyse_session": pd.to_datetime(
                [
                    "2022-01-31",
                    "2024-01-31",
                    "2024-02-29",
                    "2024-03-31",
                ]
            ),
            "x": [0, 1, 2, 3],
        }
    )
    folds = list(
        iter_monthly_walk_forward_splits(
            df,
            min_train_rows=2,
            max_train_rows=12,
            max_history=pd.Timedelta(days=120),
        )
    )
    train, test = folds[0]
    assert int(test["x"]) == 3
    assert 0 not in train["x"].tolist()
    assert set(train["x"].tolist()) == {1, 2}


def test_fold_count_matches_iterator() -> None:
    df = _monthly_df(9)
    n = fold_count(df, min_train_rows=3, max_train_rows=12, max_history=None)
    assert n == len(
        list(iter_monthly_walk_forward_splits(df, min_train_rows=3, max_train_rows=12, max_history=None))
    )


def test_split_train_test_by_anchor() -> None:
    df = _monthly_df(5, start="2024-01-31")
    train, test = split_train_test_by_anchor(df, anchor_col="anchor_nyse_session", test_anchor_min="2024-04-15")
    assert len(train) == 3
    assert len(test) == 2


def test_iter_sorts_by_anchor() -> None:
    df = pd.DataFrame(
        {
            "anchor_nyse_session": pd.to_datetime(["2024-03-31", "2024-01-31", "2024-02-29"]),
            "x": [2, 0, 1],
        }
    )
    folds = list(
        iter_monthly_walk_forward_splits(
            df,
            min_train_rows=1,
            max_train_rows=12,
            max_history=None,
        )
    )
    _, test = folds[0]
    assert int(test["x"]) == 1
