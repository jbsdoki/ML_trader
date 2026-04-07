"""
Helpers so each ``data_retrieval`` ingest module can write its own log file.

When a module imports, it calls :func:`attach_module_file_logger` once. That creates
``logs/data_retrieval/<module>.log`` next to the project (repo root is one level above
this package).

We attach the file handler to the **module** logger (e.g. ``data_retrieval.alpaca_ingest``),
not the root logger, so ingest code does not take over logging for the whole program.

Logs still go to the **terminal** as usual: messages propagate up to the root logger, where
scripts like ``run_ingest.py`` attach stderr output via ``basicConfig``. Console and file
use the same line format.

**Order matters:** the root logger starts at WARNING. Until ``basicConfig`` runs (INFO or
DEBUG), ``logger.info`` is ignored everywhere—including the file. Configure logging in your
script **before** importing ingest modules if you want INFO lines.

**Files grow over time:** each run **appends**; old lines stay. Timestamps are on each line.
Use a rotating handler later if you want size- or date-based roll-over.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_ROOT = Path(__file__).resolve().parent.parent / "logs" / "data_retrieval"
_FORMAT = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# One FileHandler per logger name; avoids duplicate handlers if a module is re-imported.
_attached: set[str] = set()


def attach_module_file_logger(logger: logging.Logger) -> None:
    """
    Append a UTF-8 log file for this logger (one file per module basename).

    Does not disable propagation; root handlers (e.g. console from ``basicConfig``) still run.
    """
    name = logger.name
    if name in _attached:
        return
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)
    basename = name.rsplit(".", maxsplit=1)[-1] if name else "data_retrieval"
    path = _LOG_ROOT / f"{basename}.log"
    # mode "a" (default): new runs append; old lines are kept.
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(_FORMAT)
    logger.addHandler(fh)
    _attached.add(name)
