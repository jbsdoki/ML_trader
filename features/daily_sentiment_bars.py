"""
Daily NYSE-session sentiment aggregates joined to OHLC bars (one row per bar).

Default sentiment mode ``open_cutoff`` assigns each article to **target session *T***
(the bar row's session) when ``session_close(T-1) <= published_at < session_open(T)``,
so overnight and post-close news on *T-1* count toward features for session *T*
(freeze at NYSE open of *T*).

Legacy mode ``article_session`` groups by the article's own NYSE session bucket.
"""

from __future__ import annotations

import sqlite3
from typing import Literal

import pandas as pd

from storage.bars_repo import fetch_bars_multi_symbol_frame
from storage.sentiment_repo import fetch_article_sentiment_frame

from .nyse_session import (
    nyse_sentiment_window_bounds_for_target_session,
    nyse_session_label_series,
)


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


def _normalize_nyse_session_cell(v: object) -> pd.Timestamp:
    t = pd.Timestamp(v)
    if t.tzinfo is not None:
        t = pd.Timestamp(t.tz_convert("UTC").date())
    return pd.Timestamp(t.date()).normalize()


def _aggregate_sentiment_open_cutoff(
    scored: pd.DataFrame,
    bar_keys: pd.DataFrame,
) -> pd.DataFrame:
    """
    One aggregate row per (symbol, nyse_session) in ``bar_keys`` using the
    prior-close-to-open(T) publication window.
    """
    cols = ["symbol", "nyse_session", "sentiment_mean", "sentiment_n", "sentiment_std"]
    if scored.empty or bar_keys.empty:
        return pd.DataFrame(columns=cols)

    pub = pd.to_datetime(scored["published_at"], utc=True, errors="coerce")
    work = scored.assign(_pub=pub)
    base = work["error"].isna() & work["score"].notna() & work["_pub"].notna()

    work["_target_session"] = pd.NaT
    keys = bar_keys[["symbol", "nyse_session"]].drop_duplicates()

    for row in keys.itertuples(index=False):
        sym = str(row.symbol).strip().upper()
        t_norm = _normalize_nyse_session_cell(row.nyse_session)
        lower, upper = nyse_sentiment_window_bounds_for_target_session(t_norm)
        m = base & (work["symbol"].str.strip().str.upper() == sym)
        if lower is not None:
            m &= work["_pub"] >= lower
        m &= work["_pub"] < upper
        work.loc[m, "_target_session"] = t_norm

    sub = work.loc[work["_target_session"].notna(), ["symbol", "_target_session", "score"]]
    if sub.empty:
        return pd.DataFrame(columns=cols)

    g = sub.groupby(["symbol", "_target_session"], as_index=False).agg(
        sentiment_mean=("score", "mean"),
        sentiment_n=("score", "count"),
        sentiment_std=("score", "std"),
    )
    return g.rename(columns={"_target_session": "nyse_session"})


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
    sentiment_mode: Literal["open_cutoff", "article_session"] = "open_cutoff",
) -> pd.DataFrame:
    """
    Return bars with left-joined daily sentiment aggregates.

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
        For ``open_cutoff``, omit a tight lower bound so overnight news before the
        first bar session is available (or set early enough).
    bar_ts_start, bar_ts_end
        Optional bounds on ``bars.bar_ts``.
    sentiment_mode
        ``open_cutoff`` (default): sentiment for target session *T* uses articles with
        ``session_close(T-1) <= published_at < session_open(T)``.
        ``article_session``: legacy bucket by each article's NYSE session label.
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
    if sentiment_mode == "open_cutoff":
        agg = _aggregate_sentiment_open_cutoff(scored, bars_labeled)
    else:
        agg = _aggregate_sentiment_by_nyse_session(scored)

    merged = bars_labeled.merge(agg, on=["symbol", "nyse_session"], how="left")
    return merged
