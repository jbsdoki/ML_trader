"""
SQLite connection helpers and on-disk database path resolution.

This module does **not** define tables — see :mod:`storage.schema` for DDL.

Useful references
-----------------
- Python ``sqlite3`` standard library:
  https://docs.python.org/3/library/sqlite3.html
- SQLite file format / when to use WAL:
  https://www.sqlite.org/wal.html
- Environment-based config pattern (12-factor style):
  https://12factor.net/config
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)

# Default relative path when ML_TRADER_DATA_DIR is unset (local dev).
_DEFAULT_DATA_SUBDIR = "data_store"


def get_data_dir() -> Path:
    """
    Return the root directory where the SQLite file and future artifacts live.

    Reads ``ML_TRADER_DATA_DIR`` from the environment. If missing, uses
    ``./data_store`` under the current working directory so clones work out of
    the box. On a VPS, set the env var to an absolute path (e.g.
    ``/opt/ml_trader/data``).
    """
    raw = os.getenv("ML_TRADER_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / _DEFAULT_DATA_SUBDIR).resolve()


def get_db_path() -> Path:
    """
    Full path to the primary SQLite database file (``ml_trader.db``).

    The parent directory is **not** created here — call :func:`ensure_data_dir`
    before first write if you need mkdir behavior.
    """
    return get_data_dir() / "ml_trader.db"


def ensure_data_dir() -> Path:
    """
    Create :func:`get_data_dir` on disk if it does not exist.

    Safe to call before opening a new database for the first time.
    """
    d = get_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def connect(*, create_parent: bool = True, **kwargs: Any) -> sqlite3.Connection:
    """
    Open a SQLite connection to ``ml_trader.db`` with sensible defaults.

    Parameters
    ----------
    create_parent
        If True, ensures the data directory exists before connecting.
    **kwargs
        Forwarded to ``sqlite3.connect`` (e.g. ``timeout=30.0``).

    Returns
    -------
    sqlite3.Connection
        Connection with ``row_factory`` set to ``sqlite3.Row`` for dict-like rows.

    Notes
    -----
    ``check_same_thread=False`` allows passing the connection across threads in
    simple scripts; for production services prefer one connection per thread or
    a pool. See Python docs for trade-offs.
    """
    if create_parent:
        ensure_data_dir()
    path = get_db_path()
    data_dir = get_data_dir()
    try:
        conn = sqlite3.connect(str(path), check_same_thread=False, **kwargs)
    except (OSError, sqlite3.Error):
        logger.exception(
            "SQLite connection failed path=%s data_dir=%s",
            path,
            data_dir,
        )
        raise
    conn.row_factory = sqlite3.Row
    # Foreign keys are off by default in SQLite; enable if you add FK constraints later.
    conn.execute("PRAGMA foreign_keys = ON;")
    logger.info(
        "SQLite connection opened path=%s data_dir=%s",
        path,
        data_dir,
    )
    return conn
