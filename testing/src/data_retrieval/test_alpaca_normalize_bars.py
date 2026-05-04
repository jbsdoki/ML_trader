"""Alpaca ingest: bar DataFrame normalization (no live API)."""

from __future__ import annotations

import pandas as pd

from data_retrieval import alpaca_ingest as alp


def test_normalize_bars_df_empty() -> None:
    out = alp._normalize_bars_df(pd.DataFrame(), "AAPL")
    assert out.empty


def test_normalize_bars_df_lowercase_and_symbol() -> None:
    ts = pd.Timestamp("2024-06-15T13:30:00", tz="UTC")
    raw = pd.DataFrame(
        {
            "timestamp": [ts],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1e6],
        }
    )
    out = alp._normalize_bars_df(raw, "msft")
    assert "open" in out.columns
    assert out["symbol"].iloc[0] == "MSFT"
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
