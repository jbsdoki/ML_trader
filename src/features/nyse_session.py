"""
Map timestamps to NYSE (XNYS) session labels for alignment with daily OHLC bars.

Uses ``exchange_calendars`` session indices (timezone-naive UTC midnight labels).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from exchange_calendars.errors import RequestedSessionOutOfBounds


@lru_cache(maxsize=1)
def _xnys_calendar():
    return xcals.get_calendar("XNYS")


def _naive_session_label(session_label: Any) -> pd.Timestamp:
    """XNYS session index: timezone-naive midnight (calendar date in exchange_calendars)."""
    t = pd.Timestamp(session_label)
    if t.tzinfo is not None:
        t = pd.Timestamp(t.tz_convert("UTC").date())
    return pd.Timestamp(t.date()).normalize()


@lru_cache(maxsize=4096)
def nyse_sentiment_window_bounds_for_target_session(session_label: Any) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """
    Bounds on ``published_at`` (UTC) for features frozen at **NYSE open** of target session *T*.

    Include articles with ``session_close(T-1) <= published_at < session_open(T)``.
    If there is no prior session in the calendar, returns ``(None, upper)`` so only
    ``published_at < session_open(T)`` applies.

    Returns
    -------
    lower_inclusive, upper_exclusive
        Both timezone-aware UTC (from ``exchange_calendars``).
    """
    cal = _xnys_calendar()
    t = _naive_session_label(session_label)
    upper = cal.session_open(t)
    try:
        prev = cal.previous_session(t)
    except RequestedSessionOutOfBounds:
        return None, upper
    lower = cal.session_close(prev)
    return lower, upper


def nyse_session_label_for_instant(ts: Any) -> pd.Timestamp:
    """
    Map an instant (article ``published_at``, bar ``bar_ts``, etc.) to the XNYS
    session it belongs to for daily aggregation.

    Steps: normalize to UTC, convert to ``America/New_York``, take the local
    calendar date *D*. If *D* is a trading session, use that session (same
    calendar day as the NYSE regular session, including pre/post on that day).
    If *D* is a weekend or exchange holiday, use the next session.
    """
    cal = _xnys_calendar()
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    local = t.tz_convert("America/New_York")
    d = local.date()
    label = pd.Timestamp(d)
    if cal.is_session(label):
        return label
    return cal.date_to_session(label, direction="next")


def nyse_session_label_series(series: pd.Series) -> pd.Series:
    """Vectorized wrapper: one XNYS session label per row (same rules as above)."""
    return series.map(nyse_session_label_for_instant)
