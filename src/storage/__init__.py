"""
Local persistence (SQLite): database path, schema, article and bar repositories.
"""

from .database import connect, ensure_data_dir, get_data_dir, get_db_path
from .schema import init_schema, schema_version
from .articles_repo import article_dedupe_key, fetch_articles_frame, upsert_articles
from .bars_repo import fetch_bars_frame, upsert_bars

__all__ = [
    "article_dedupe_key",
    "connect",
    "ensure_data_dir",
    "fetch_articles_frame",
    "fetch_bars_frame",
    "get_data_dir",
    "get_db_path",
    "init_schema",
    "schema_version",
    "upsert_articles",
    "upsert_bars",
]
