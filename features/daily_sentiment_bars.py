"""
Daily NYSE-session sentiment aggregates joined to OHLC bars (one row per bar).
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from storage.bars_repo import fetch_bars_multi_symbol_frame
from storage.sentiment_repo import fetch_article_sentiment_frame

from .nyse_session import nyse_session_label_series


def _sentiment_with_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["nyse_session"] = nyse_session_label_series(out["published_at"])
    return out


def _aggregate_sentiment_by_nyse_session(df: pd.DataFrame) -> pd.DataFrame:
    work = _sentiment_with_session_labels(df)
    ok = work["error"].isna() & work["score"].notna()
    sub = work.loc[ok, ["symbol", "nyse_session", "score"]]
    if sub.empty:
        return pd.DataFrame(columns=["symbol", "nyse_session", "sentiment_mean", "sentiment_n", "sentiment_std"])

    g = sub.groupby(["symbol", "nyse_session"], as_index=False).agg(
        sentiment_mean=("score", "mean"),
        sentiment_n=("score", "count"),
        sentiment_std=("score", "std"),
    )
    return g


def _bars_with_nyse_session(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["nyse_session"] = nyse_session_label_series(out["bar_ts"])
    return out


def build_daily_bars_sentiment_frame(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    symbols: list[str],
    bar_interval: str = "1d",
    bar_source_api: str | None = None,
    published_start: str | None = None,
    published_end: str | None = None,
    bar_ts_start: str | None = None,
    bar_ts_end: str | None = None,
) -> pd.DataFrame:
    """
    Return bars with left-joined daily sentiment aggregates (NYSE session buckets).

    Parameters
    ----------
    conn
        SQLite connection with ``articles``, ``article_sentiment``, and ``bars``.
    model_id
        Sentiment model id stored in ``article_sentiment`` (e.g. ``finbert``).
    symbols
        Uppercase tickers to include (both news and bars).
    bar_interval
        Bar frequency key stored in ``bars`` (e.g. ``1d``).
    bar_source_api
        If set, restrict bars to this vendor (recommended when multiple sources exist).
    published_start, published_end
        Optional bounds on ``articles.published_at`` (ISO strings or dates).
    bar_ts_start, bar_ts_end
        Optional bounds on ``bars.bar_ts``.
    """
    sym_clean = [x.strip().upper() for x in symbols if x and str(x).strip()]
    if not sym_clean:
        return pd.DataFrame()

    scored = fetch_article_sentiment_frame(
        conn,
        model_id=model_id,
        symbols=sym_clean,
        start=published_start,
        end=published_end,
    )
    agg = _aggregate_sentiment_by_nyse_session(scored)

    bars = fetch_bars_multi_symbol_frame(
        conn,
        symbols=sym_clean,
        bar_interval=bar_interval,
        source_api=bar_source_api,
        bar_ts_start=bar_ts_start,
        bar_ts_end=bar_ts_end,
    )
    if bars.empty:
        return bars

    bars_labeled = _bars_with_nyse_session(bars)
    merged = bars_labeled.merge(agg, on=["symbol", "nyse_session"], how="left")
    return merged
