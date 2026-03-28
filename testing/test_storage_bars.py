"""Bars repo: upsert, single- and multi-symbol fetch."""

from __future__ import annotations

import sqlite3

import pandas as pd

from storage.bars_repo import (
    fetch_bars_frame,
    fetch_bars_multi_symbol_frame,
    upsert_bars,
)


def test_upsert_bars_and_fetch_roundtrip(
    sqlite_conn: sqlite3.Connection,
    sample_bars_df: pd.DataFrame,
) -> None:
    inserted = upsert_bars(sqlite_conn, sample_bars_df, "yfinance", "1d")
    assert inserted == 1
    out = fetch_bars_frame(sqlite_conn, symbol="AAPL", bar_interval="1d", source_api="yfinance")
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 181.5


def test_fetch_bars_multi_symbol(
    sqlite_conn: sqlite3.Connection,
) -> None:
    ts = pd.Timestamp("2024-06-16T13:30:00", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["AAPL", "MSFT"],
            "open": [180.0, 400.0],
            "high": [182.0, 405.0],
            "low": [179.0, 398.0],
            "close": [181.0, 402.0],
            "volume": [1e6, 2e6],
        }
    )
    upsert_bars(sqlite_conn, df, "alpaca", "1d")
    aapl = fetch_bars_multi_symbol_frame(
        sqlite_conn,
        symbols=["AAPL"],
        bar_interval="1d",
        source_api="alpaca",
    )
    assert len(aapl) == 1
    both = fetch_bars_multi_symbol_frame(
        sqlite_conn,
        symbols=["AAPL", "MSFT"],
        bar_interval="1d",
        source_api="alpaca",
    )
    assert len(both) == 2
