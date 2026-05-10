"""Tests for ``features.monthly_forward_frame``."""

from __future__ import annotations

import sqlite3

import exchange_calendars as xcals
import pandas as pd
import pytest

from features.monthly_forward_frame import (
    build_monthly_forward_frame,
    default_monthly_training_feature_columns,
    time_series_split_by_anchor,
)
from storage.articles_repo import upsert_articles
from storage.bars_repo import upsert_bars
from storage.sentiment_repo import fetch_articles_for_sentiment, upsert_article_sentiment


def _xnys():
    return xcals.get_calendar("XNYS")


def _bars_df_for_sessions(
    sessions: pd.DatetimeIndex,
    *,
    symbol: str,
    close_by_session: dict[pd.Timestamp, float],
) -> pd.DataFrame:
    cal = _xnys()
    rows: list[dict] = []
    for s in sessions:
        s_norm = pd.Timestamp(s).normalize()
        o = cal.session_open(s_norm)
        c = float(close_by_session[s_norm])
        h = c + 0.25
        low = c - 0.25
        op = c - 0.1
        rows.append(
            {
                "timestamp": o,
                "symbol": symbol,
                "open": op,
                "high": h,
                "low": low,
                "close": c,
                "volume": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def _exit_session_n_forward(anchor: pd.Timestamp, n: int) -> pd.Timestamp:
    cal = _xnys()
    cur = pd.Timestamp(anchor).normalize()
    for _ in range(n):
        cur = cal.next_session(cur)
    return cur.normalize()


def test_build_monthly_forward_frame_forward_return_and_sentiment(sqlite_conn: sqlite3.Connection) -> None:
    symbol = "MFTEST"
    cal = _xnys()
    anchor = pd.Timestamp("2024-01-31").normalize()
    assert cal.is_session(anchor)
    exit_s = _exit_session_n_forward(anchor, 21)

    sessions = cal.sessions_in_range(pd.Timestamp("2023-12-01").normalize(), exit_s.normalize())
    close_map: dict[pd.Timestamp, float] = {pd.Timestamp(s).normalize(): 50.0 + float(i) * 0.01 for i, s in enumerate(sessions)}
    close_map[anchor] = 100.0
    close_map[exit_s.normalize()] = 110.0

    bars = _bars_df_for_sessions(sessions, symbol=symbol, close_by_session=close_map)
    upsert_bars(sqlite_conn, bars, "alpaca", "1d")

    art = pd.DataFrame(
        [
            {
                "article_id": "mf-dec-1",
                "datetime": "2023-12-15T16:00:00+00:00",
                "headline": "h",
                "summary": "s",
                "source": "finnhub",
                "url": "https://example.com/mf-1",
                "symbol": symbol,
            },
            {
                "article_id": "mf-jan-1",
                "datetime": "2024-01-10T16:00:00+00:00",
                "headline": "h2",
                "summary": "s2",
                "source": "finnhub",
                "url": "https://example.com/mf-2",
                "symbol": symbol,
            },
        ]
    )
    upsert_articles(sqlite_conn, art, "finnhub")
    pending = fetch_articles_for_sentiment(sqlite_conn, model_id="finbert", only_missing=True)
    scores = []
    for _, row in pending.iterrows():
        scores.append(
            {
                "dedupe_key": str(row["dedupe_key"]),
                "model_id": "finbert",
                "score": 0.25 if str(row.get("headline", "")) == "h" else 0.75,
                "prob_pos": 0.5,
                "prob_neg": 0.25,
                "prob_neutral": 0.25,
                "text_hash": "t",
                "error": None,
            }
        )
    upsert_article_sentiment(sqlite_conn, scores)

    out = build_monthly_forward_frame(
        sqlite_conn,
        symbols=[symbol],
        model_id="finbert",
        bar_interval="1d",
        bar_source_api="alpaca",
        anchor_session_start="2024-01-01",
        anchor_session_end="2024-01-31",
        forward_sessions=21,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == symbol
    assert pd.Timestamp(row["anchor_nyse_session"]).normalize() == anchor
    assert row["forward_return"] == pytest.approx(110.0 / 100.0 - 1.0)
    assert row["prior_month_sentiment_n"] == 1.0
    assert row["prior_month_sentiment_mean"] == pytest.approx(0.25)
    assert row["prior_month_bar_days"] >= 2.0


def test_build_monthly_forward_frame_drops_row_without_forward_history(sqlite_conn: sqlite3.Connection) -> None:
    symbol = "MFSHORT"
    sessions = _xnys().sessions_in_range(pd.Timestamp("2023-12-01").normalize(), pd.Timestamp("2023-12-31").normalize())
    close_map = {pd.Timestamp(s).normalize(): 100.0 for s in sessions}
    bars = _bars_df_for_sessions(sessions, symbol=symbol, close_by_session=close_map)
    upsert_bars(sqlite_conn, bars, "alpaca", "1d")

    out = build_monthly_forward_frame(
        sqlite_conn,
        symbols=[symbol],
        model_id="finbert",
        bar_interval="1d",
        bar_source_api="alpaca",
        forward_sessions=21,
    )
    assert out.empty


def test_build_monthly_forward_frame_empty_inputs() -> None:
    conn = sqlite3.connect(":memory:")
    from storage.schema import init_schema

    init_schema(conn)
    assert build_monthly_forward_frame(conn, symbols=[], model_id="finbert").empty
    assert build_monthly_forward_frame(conn, symbols=["A"], model_id="finbert", forward_sessions=0).empty
    conn.close()


def test_default_monthly_training_feature_columns_order() -> None:
    cols = default_monthly_training_feature_columns()
    assert cols[0] == "prior_month_bar_ret_mean"
    assert "forward_return" not in cols
    assert len(cols) == 6


def test_time_series_split_by_anchor() -> None:
    df = pd.DataFrame(
        {
            "anchor_nyse_session": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]),
            "forward_return": [0.01, 0.02, 0.03, 0.04],
        }
    )
    train, test = time_series_split_by_anchor(df, train_frac=0.5)
    assert len(train) == 2
    assert len(test) == 2


def test_build_monthly_forward_frame_no_bars(sqlite_conn: sqlite3.Connection) -> None:
    out = build_monthly_forward_frame(
        sqlite_conn,
        symbols=["NONE"],
        model_id="finbert",
        bar_interval="1d",
    )
    assert out.empty
