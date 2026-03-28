"""Ingest pipeline: pure helpers and run_ingest_pipeline with mocks."""

from __future__ import annotations

import sqlite3

import pytest

from pipelines.ingest_pipeline import (
    IngestConfig,
    normalize_symbols,
    parse_sources_csv,
    run_ingest_pipeline,
)
from storage.schema import init_schema


def test_normalize_symbols_strips_and_uppercases() -> None:
    assert normalize_symbols(["  aapl  ", "MSFT", ""]) == ["AAPL", "MSFT"]


def test_parse_sources_csv() -> None:
    d = parse_sources_csv("finnhub,yfinance")
    assert d["finnhub"] is True and d["yfinance"] is True
    assert d["newsapi"] is False


def test_run_ingest_pipeline_no_symbols() -> None:
    cfg = IngestConfig(symbols=[], start="2024-01-01", end="2024-01-05")
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    summary = run_ingest_pipeline(cfg, conn=conn, init_db=False)
    assert any("No symbols" in e for e in summary.errors)


def test_run_ingest_pipeline_mocks_yfinance(monkeypatch) -> None:
    def _fake_yf(
        conn: sqlite3.Connection,
        symbols: list,
        start: str,
        end: str,
        interval: str,
        summary,
    ) -> None:
        summary.bars_yfinance_inserted += 2

    monkeypatch.setattr(
        "pipelines.ingest_pipeline._ingest_yfinance_bars",
        _fake_yf,
    )
    cfg = IngestConfig(
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-01-10",
        finnhub=False,
        newsapi=False,
        yfinance=True,
        alpaca=False,
    )
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    summary = run_ingest_pipeline(cfg, conn=conn, init_db=False)
    assert summary.bars_yfinance_inserted == 2
    assert summary.errors == []
