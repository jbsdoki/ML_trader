"""Bars repo: composite primary key dedupes duplicate upserts."""

from __future__ import annotations

import sqlite3

import pandas as pd

from storage.bars_repo import upsert_bars


def test_bars_second_upsert_same_pk_inserts_zero(
    sqlite_conn: sqlite3.Connection,
    sample_bars_df: pd.DataFrame,
) -> None:
    n1 = upsert_bars(sqlite_conn, sample_bars_df, "yfinance", "1d")
    assert n1 == 1
    n2 = upsert_bars(sqlite_conn, sample_bars_df, "yfinance", "1d")
    assert n2 == 0
    cnt = sqlite_conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert cnt == 1


def test_bars_different_source_same_ts_allowed(
    sqlite_conn: sqlite3.Connection,
    sample_bars_df: pd.DataFrame,
) -> None:
    upsert_bars(sqlite_conn, sample_bars_df, "yfinance", "1d")
    n = upsert_bars(sqlite_conn, sample_bars_df, "alpaca", "1d")
    assert n == 1
    cnt = sqlite_conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert cnt == 2
