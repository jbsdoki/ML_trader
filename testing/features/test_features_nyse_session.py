"""NYSE session labels and open-cutoff bounds (exchange_calendars)."""

from __future__ import annotations

import pandas as pd

from features.nyse_session import (
    nyse_sentiment_window_bounds_for_target_session,
    nyse_session_label_for_instant,
)


def test_nyse_session_label_for_instant_friday_evening_stays_friday() -> None:
    t = pd.Timestamp("2024-01-12T23:00:00+00:00")
    lab = nyse_session_label_for_instant(t)
    assert lab == pd.Timestamp("2024-01-12")


def test_sentiment_window_bounds_known_session() -> None:
    lower, upper = nyse_sentiment_window_bounds_for_target_session(pd.Timestamp("2024-01-16"))
    assert lower == pd.Timestamp("2024-01-12 21:00:00", tz="UTC")
    assert upper == pd.Timestamp("2024-01-16 14:30:00", tz="UTC")
