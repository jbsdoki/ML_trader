"""
Build a **month-end–anchored** supervised frame for monthly ML.

Each row is one (symbol, month-end NYSE session):

- **Features** use only information from the **prior calendar month**: daily bar
  return statistics and article sentiment aggregates (``published_at`` in that
  month). All of that is strictly before the anchor close, so there is no
  lookahead in the feature window definition.

- **Label** is the **close-to-close** simple return from the anchor session’s
  close to the close **21 trading sessions later** on the XNYS calendar (the
  ``forward_sessions`` parameter defaults to 21).

Rows are dropped when the forward horizon is missing from stored bars (no
partial labels). Optional ``bar_ts_start`` / ``bar_ts_end`` narrow the SQLite
query; anchors are still required to be month-end trading days that have a bar.

See also ``features.daily_sentiment_bars`` for daily bar + sentiment joins; this
module is the **monthly** aggregation and forward-return target used in the
walk-forward backtest scripts.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from storage.bars_repo import fetch_bars_multi_symbol_frame
from storage.sentiment_repo import fetch_article_sentiment_frame

from .nyse_session import nyse_session_label_series


def _xnys() -> Any:
    return xcals.get_calendar("XNYS")


def _normalize_session_date(session: Any) -> pd.Timestamp:
    t = pd.Timestamp(session)
    if t.tzinfo is not None:
        t = pd.Timestamp(t.tz_convert("UTC").date())
    return pd.Timestamp(t.date()).normalize()


def _prior_calendar_month_bounds(anchor_session: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    First and last **calendar** dates of the month immediately before the
    anchor’s calendar month (naive dates, NYSE session calendar date).
    """
    a = _normalize_session_date(anchor_session)
    first_this_month = a.replace(day=1)
    last_prior = first_this_month - pd.Timedelta(days=1)
    first_prior = last_prior.replace(day=1)
    return first_prior.normalize(), last_prior.normalize()


def _utc_end_of_day(d: pd.Timestamp) -> pd.Timestamp:
    dn = _normalize_session_date(d)
    return pd.Timestamp(dn.year, dn.month, dn.day, 23, 59, 59, 999999, tz="UTC")


def _month_end_nyse_sessions(cal: Any, range_start: pd.Timestamp, range_end: pd.Timestamp) -> list[pd.Timestamp]:
    """Last XNYS session whose **calendar date** falls in each month between range_start and range_end."""
    rs = _normalize_session_date(range_start)
    re = _normalize_session_date(range_end)
    out: list[pd.Timestamp] = []
    cur = rs.replace(day=1)
    end_month = re.replace(day=1)
    while cur <= end_month:
        month_end = cur + pd.offsets.MonthEnd(0)
        seg = cal.sessions_in_range(cur.normalize(), month_end.normalize())
        if len(seg) > 0:
            out.append(pd.Timestamp(seg[-1]).normalize())
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def _session_n_forward(cal: Any, session: Any, n: int) -> pd.Timestamp:
    cur = _normalize_session_date(session)
    for _ in range(int(n)):
        cur = cal.next_session(cur)
    return _normalize_session_date(cur)


def _dedupe_bars_by_session(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    work = bars.copy()
    work["bar_ts"] = pd.to_datetime(work["bar_ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["bar_ts"])
    work["nyse_session"] = nyse_session_label_series(work["bar_ts"])
    return work.sort_values("nyse_session").drop_duplicates(subset=["nyse_session"], keep="last")


def _session_to_close_map(deduped: pd.DataFrame) -> pd.Series:
    return deduped.set_index("nyse_session")["close"].astype(float)


def _forward_simple_return(
    cal: Any,
    close_by_session: pd.Series,
    anchor_session: Any,
    forward_sessions: int,
) -> float | None:
    a = _normalize_session_date(anchor_session)
    exit_s = _session_n_forward(cal, a, forward_sessions)
    try:
        c0 = float(close_by_session.loc[a])
        c1 = float(close_by_session.loc[exit_s])
    except (KeyError, TypeError, ValueError):
        return None
    if c0 <= 0 or pd.isna(c0) or pd.isna(c1):
        return None
    return c1 / c0 - 1.0


def _daily_returns_in_session_order(deduped: pd.DataFrame) -> pd.Series:
    """Close-to-close returns aligned with ``deduped`` rows (sorted by session)."""
    closes = deduped["close"].astype(float)
    return closes.pct_change()


def _prior_month_bar_stats(deduped: pd.DataFrame, first_prior: pd.Timestamp, last_prior: pd.Timestamp) -> dict[str, float]:
    work = deduped.copy()
    mask = (work["nyse_session"] >= first_prior) & (work["nyse_session"] <= last_prior)
    sub = work.loc[mask]
    if len(sub) < 2:
        rets = _daily_returns_in_session_order(sub)
        valid = rets.dropna()
        if valid.empty:
            return {"prior_month_bar_ret_mean": float("nan"), "prior_month_bar_ret_std": float("nan"), "prior_month_bar_days": 0.0}
        return {
            "prior_month_bar_ret_mean": float(valid.mean()),
            "prior_month_bar_ret_std": float("nan"),
            "prior_month_bar_days": float(len(sub)),
        }
    rets = _daily_returns_in_session_order(sub).dropna()
    return {
        "prior_month_bar_ret_mean": float(rets.mean()) if not rets.empty else float("nan"),
        "prior_month_bar_ret_std": float(rets.std(ddof=1)) if len(rets) > 1 else float("nan"),
        "prior_month_bar_days": float(len(sub)),
    }


def _aggregate_sentiment_scores(scored: pd.DataFrame, first_prior: pd.Timestamp, last_prior: pd.Timestamp) -> dict[str, float]:
    pub = pd.to_datetime(scored["published_at"], utc=True, errors="coerce")
    lower = pd.Timestamp(first_prior).tz_localize("UTC")
    upper = _utc_end_of_day(last_prior)
    ok = scored["error"].isna() & scored["score"].notna() & pub.notna()
    m = ok & (pub >= lower) & (pub <= upper)
    vals = scored.loc[m, "score"].astype(float)
    if vals.empty:
        return {
            "prior_month_sentiment_mean": float("nan"),
            "prior_month_sentiment_std": float("nan"),
            "prior_month_sentiment_n": 0.0,
        }
    return {
        "prior_month_sentiment_mean": float(vals.mean()),
        "prior_month_sentiment_std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
        "prior_month_sentiment_n": float(len(vals)),
    }


def _load_scored_articles(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    symbol: str,
    published_start: str,
    published_end: str,
) -> pd.DataFrame:
    return fetch_article_sentiment_frame(
        conn,
        model_id=model_id,
        symbols=[symbol],
        start=published_start,
        end=published_end,
    )


def _row_for_anchor(
    cal: Any,
    *,
    symbol: str,
    anchor_session: pd.Timestamp,
    deduped_bars: pd.DataFrame,
    scored_cache: pd.DataFrame,
    forward_sessions: int,
    require_forward_return: bool,
) -> dict[str, Any] | None:
    first_prior, last_prior = _prior_calendar_month_bounds(anchor_session)
    close_map = _session_to_close_map(deduped_bars)
    y = _forward_simple_return(cal, close_map, anchor_session, forward_sessions)
    if y is None and require_forward_return:
        return None
    bar_part = _prior_month_bar_stats(deduped_bars, first_prior, last_prior)
    sent_part = _aggregate_sentiment_scores(scored_cache, first_prior, last_prior)
    fwd = float(y) if y is not None else float("nan")
    return {
        "symbol": symbol.upper(),
        "anchor_nyse_session": _normalize_session_date(anchor_session),
        **bar_part,
        **sent_part,
        "forward_return": fwd,
    }


def default_monthly_training_feature_columns() -> list[str]:
    """Feature columns used by monthly XGBoost scripts (order-stable)."""
    return [
        "prior_month_bar_ret_mean",
        "prior_month_bar_ret_std",
        "prior_month_bar_days",
        "prior_month_sentiment_mean",
        "prior_month_sentiment_n",
        "prior_month_sentiment_std",
    ]


def time_series_split_by_anchor(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    anchor_col: str = "anchor_nyse_session",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on ``anchor_col`` (sorted). Last ``1 - train_frac`` is test.
    """
    if df.empty:
        return df.copy(), df.copy()
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1).")
    ordered = df.sort_values(anchor_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    cut = max(1, min(n - 1, int(n * train_frac)))
    train = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()
    return train, test


def build_monthly_forward_frame(
    conn: sqlite3.Connection,
    *,
    symbols: list[str],
    model_id: str,
    bar_interval: str = "1d",
    bar_source_api: str | None = None,
    bar_ts_start: str | None = None,
    bar_ts_end: str | None = None,
    anchor_session_start: str | None = None,
    anchor_session_end: str | None = None,
    forward_sessions: int = 21,
    require_forward_return: bool = True,
) -> pd.DataFrame:
    """
    Assemble the monthly supervised frame for all symbols.

    Parameters
    ----------
    conn
        SQLite connection with ``bars``, ``articles``, and ``article_sentiment``.
    symbols
        Equity tickers (uppercased internally).
    model_id
        Sentiment model id (e.g. ``finbert``) for ``fetch_article_sentiment_frame``.
    bar_interval, bar_source_api
        Passed to ``fetch_bars_multi_symbol_frame`` (restrict vendor when set).
    bar_ts_start, bar_ts_end
        Optional ISO bounds on ``bars.bar_ts`` to limit IO.
    anchor_session_start, anchor_session_end
        Optional inclusive range on **calendar dates** of month-end anchors to emit.
        If omitted, uses every month-end session between the first and last bar
        session in the loaded bar slice (per symbol), subject to label availability.
    forward_sessions
        Number of XNYS sessions to step forward from the anchor (default 21).
    require_forward_return
        If True (default), drop rows where the forward horizon is missing from
        bars. If False, still emit feature rows with ``forward_return`` set to
        NaN (for live prediction after month-end without future bars).
    """
    sym_clean = [x.strip().upper() for x in symbols if x and str(x).strip()]
    if not sym_clean or forward_sessions < 1:
        return pd.DataFrame()

    cal = _xnys()
    bars_all = fetch_bars_multi_symbol_frame(
        conn,
        symbols=sym_clean,
        bar_interval=bar_interval,
        source_api=bar_source_api,
        bar_ts_start=bar_ts_start,
        bar_ts_end=bar_ts_end,
    )
    if bars_all.empty:
        return pd.DataFrame()

    a0 = anchor_session_start
    a1 = anchor_session_end
    rows: list[dict[str, Any]] = []

    for symbol in sym_clean:
        b_sym = bars_all.loc[bars_all["symbol"].str.upper() == symbol.upper()].copy()
        if b_sym.empty:
            continue
        deduped = _dedupe_bars_by_session(b_sym)
        if deduped.empty:
            continue
        sess_min = _normalize_session_date(deduped["nyse_session"].min())
        sess_max = _normalize_session_date(deduped["nyse_session"].max())
        candidates = _month_end_nyse_sessions(cal, sess_min, sess_max)
        if a0 is not None:
            lo = _normalize_session_date(a0)
            candidates = [c for c in candidates if c >= lo]
        if a1 is not None:
            hi = _normalize_session_date(a1)
            candidates = [c for c in candidates if c <= hi]
        known_sessions = {_normalize_session_date(x) for x in deduped["nyse_session"]}
        candidates = [c for c in candidates if _normalize_session_date(c) in known_sessions]

        if not candidates:
            continue

        global_prior_first = min(_prior_calendar_month_bounds(c)[0] for c in candidates)
        global_prior_last = max(_prior_calendar_month_bounds(c)[1] for c in candidates)
        pub_start = pd.Timestamp(global_prior_first).tz_localize("UTC").isoformat()
        pub_end = _utc_end_of_day(global_prior_last).isoformat()
        scored = _load_scored_articles(conn, model_id=model_id, symbol=symbol, published_start=pub_start, published_end=pub_end)

        for anchor in candidates:
            rec = _row_for_anchor(
                cal,
                symbol=symbol,
                anchor_session=anchor,
                deduped_bars=deduped,
                scored_cache=scored,
                forward_sessions=forward_sessions,
                require_forward_return=require_forward_return,
            )
            if rec is not None:
                rows.append(rec)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["symbol", "anchor_nyse_session"]).reset_index(drop=True)
    return out
