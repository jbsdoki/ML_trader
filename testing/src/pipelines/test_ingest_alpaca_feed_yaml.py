"""Ingest pipeline: YAML ``alpaca_feed`` maps into :class:`IngestConfig`."""

from __future__ import annotations

from pathlib import Path

from pipelines.ingest_pipeline import IngestConfig, load_ingest_config_yaml


def test_load_ingest_config_yaml_alpaca_feed_iex(tmp_path: Path) -> None:
    p = tmp_path / "ingest.yaml"
    p.write_text(
        """
symbols: [AAPL]
start: "2024-01-01"
end: "2024-01-31"
bar_interval: "1d"
sources: [alpaca]
alpaca_feed: "iex"
""",
        encoding="utf-8",
    )
    cfg = load_ingest_config_yaml(p)
    assert isinstance(cfg, IngestConfig)
    assert cfg.alpaca_feed == "iex"
    assert cfg.alpaca is True
