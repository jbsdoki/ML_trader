"""Daily bars + sentiment join: empty DB and symbol edge cases."""

from __future__ import annotations

from features.daily_sentiment_bars import build_daily_bars_sentiment_frame


def test_build_daily_bars_sentiment_frame_no_bars_returns_empty(sqlite_conn) -> None:
    out = build_daily_bars_sentiment_frame(
        sqlite_conn,
        model_id="finbert",
        symbols=["AAPL"],
        bar_interval="1d",
        bar_source_api="yfinance",
    )
    assert out.empty


def test_build_daily_bars_sentiment_frame_empty_symbols_returns_empty(sqlite_conn) -> None:
    out = build_daily_bars_sentiment_frame(
        sqlite_conn,
        model_id="finbert",
        symbols=[],
        bar_interval="1d",
    )
    assert out.empty
