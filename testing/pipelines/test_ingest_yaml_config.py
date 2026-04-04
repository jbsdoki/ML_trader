"""Ingest pipeline: YAML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.ingest_pipeline import IngestConfig, load_ingest_config_yaml


def test_load_ingest_config_yaml_minimal(tmp_path: Path) -> None:
    p = tmp_path / "ingest.yaml"
    p.write_text(
        """
symbols:
  - AAPL
  - msft
start: "2024-01-01"
end: "2024-01-31"
bar_interval: "1d"
sources:
  - finnhub
  - yfinance
""",
        encoding="utf-8",
    )
    cfg = load_ingest_config_yaml(p)
    assert isinstance(cfg, IngestConfig)
    assert cfg.symbols == ["AAPL", "MSFT"]
    assert cfg.start == "2024-01-01"
    assert cfg.end == "2024-01-31"
    assert cfg.bar_interval == "1d"
    assert cfg.finnhub is True
    assert cfg.yfinance is True
    assert cfg.newsapi is False
    assert cfg.alpaca is False


def test_load_ingest_config_yaml_missing_required_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("symbols: [AAPL]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symbols, start, and end"):
        load_ingest_config_yaml(p)
