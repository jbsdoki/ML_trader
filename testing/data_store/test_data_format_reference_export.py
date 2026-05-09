"""
Write a human-readable data format reference (no golden-string assertions).

Output: ``logs/testing/data_formats/DATA_FORMATS.md`` (gitignored via ``logs/*``).

Run::

    python -m pytest testing/data_store/test_data_format_reference_export.py -v
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_DIR = _REPO_ROOT / "logs" / "testing" / "data_formats"
_EXPORT_PATH = _EXPORT_DIR / "DATA_FORMATS.md"


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [str(r[0]) for r in cur.fetchall()]


def _pragma_columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    rows: list[dict[str, Any]] = []
    for r in cur.fetchall():
        rows.append(
            {
                "cid": r[0],
                "name": r[1],
                "type": r[2],
                "notnull": bool(r[3]),
                "default": r[4],
                "pk": bool(r[5]),
            }
        )
    return rows


def _fmt_sqlite_section(conn: sqlite3.Connection) -> str:
    lines = ["## SQLite application tables", ""]
    for table in _user_tables(conn):
        lines.append(f"### `{table}`")
        lines.append("")
        lines.append("| column | sqlite type | pk | notnull |")
        lines.append("|--------|-------------|----|---------|")
        for c in _pragma_columns(conn, table):
            lines.append(
                f"| `{c['name']}` | {c['type']} | {c['pk']} | {c['notnull']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _df_columns_md(title: str, df: pd.DataFrame) -> str:
    lines = [f"### {title}", ""]
    if df.empty:
        lines.append("_(empty frame; columns are still defined for dtype)_")
        lines.append("")
    lines.append("| column | dtype |")
    lines.append("|--------|-------|")
    for col in df.columns:
        lines.append(f"| `{col}` | `{df[col].dtype}` |")
    lines.append("")
    return "\n".join(lines)


def _sample_alpaca_bars_df() -> pd.DataFrame:
    from data_retrieval import alpaca_ingest as alp

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
    return alp._normalize_bars_df(raw, "msft")


def _sample_finnhub_articles_df() -> pd.DataFrame:
    from data_retrieval import finnhub_ingest as fh

    records = [
        {
            "id": 99,
            "datetime": 1718448000,
            "headline": "H",
            "summary": "S",
            "source": "Finnhub",
            "url": "https://example.com/n",
            "category": None,
            "image": None,
            "related": None,
        }
    ]
    return fh._news_records_to_df(records, "aapl")


def _sample_newsapi_articles_df() -> pd.DataFrame:
    from data_retrieval import newsapi_ingest as n

    articles = [
        {
            "title": "T1",
            "description": "D1",
            "publishedAt": "2024-06-15T12:00:00Z",
            "url": "https://example.com/1",
            "source": {"name": "SRC"},
            "author": "A1",
            "content": None,
        }
    ]
    return n._articles_to_df(articles, "AAPL")


def _ingest_dataframes_md() -> str:
    parts = [
        "## Normalized ingest DataFrames (pandas, before SQLite)",
        "",
        "Shapes below come from the same helper paths used in production; "
        "values are fixtures, not live API data.",
        "",
    ]
    parts.append(_df_columns_md("Alpaca OHLCV (`_normalize_bars_df`)", _sample_alpaca_bars_df()))
    parts.append(_df_columns_md("Finnhub news (`_news_records_to_df`)", _sample_finnhub_articles_df()))
    parts.append(_df_columns_md("NewsAPI articles (`_articles_to_df`)", _sample_newsapi_articles_df()))
    return "\n".join(parts)


def _training_features_md() -> str:
    from features.training_frame import default_training_feature_columns

    cols = default_training_feature_columns()
    lines = [
        "## Training feature column order",
        "",
        "`features.training_frame.default_training_feature_columns()`:",
        "",
    ]
    for c in cols:
        lines.append(f"- `{c}`")
    lines.append("")
    return "\n".join(lines)


def test_export_data_format_reference_markdown(sqlite_conn: sqlite3.Connection) -> None:
    """
    Writes ``DATA_FORMATS.md`` for reading; only checks the file was created and non-empty.
    """
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# ML_Trader data format reference",
            "",
            "_Generated by `testing/data_store/test_data_format_reference_export.py`. "
            "Do not edit by hand; re-run pytest to refresh._",
            "",
            _fmt_sqlite_section(sqlite_conn),
            _ingest_dataframes_md(),
            _training_features_md(),
        ]
    )
    _EXPORT_PATH.write_text(body, encoding="utf-8")
    assert _EXPORT_PATH.is_file()
    assert _EXPORT_PATH.stat().st_size > 100
