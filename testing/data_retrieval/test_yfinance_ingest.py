"""yfinance ingest: mock HTTP layer, assert normalized output shape."""

from __future__ import annotations

import pandas as pd

import data_retrieval.yfinance_ingest as yfi


def test_fetch_ohlcv_normalizes_and_tags_symbol(monkeypatch) -> None:
    class _FakeTicker:
        def history(self, **kwargs):
            idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")])
            return pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.5],
                    "Volume": [1e6],
                },
                index=idx,
            )

    monkeypatch.setattr(yfi.yf, "Ticker", lambda symbol: _FakeTicker())
    df = yfi.fetch_ohlcv(
        "AAPL",
        start="2024-01-01",
        end="2024-01-10",
        interval="1d",
    )
    assert not df.empty
    assert "open" in df.columns
    assert df["symbol"].iloc[0] == "AAPL"
    assert float(df["close"].iloc[0]) == 100.5


def test_normalize_ohlcv_df_lowercases_columns() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    raw = pd.DataFrame({"Open": [1.0], "HIGH": [2.0]}, index=idx)
    out = yfi._normalize_ohlcv_df(raw)
    assert list(out.columns) == ["open", "high"]
