"""
File logging for :mod:`storage` modules under ``logs/storage/``.

Same pattern as :mod:`data_retrieval._file_log`: each module calls
:func:`attach_module_file_logger` once; handlers attach to that module's logger;
``propagate`` remains true so the root logger (e.g. stderr from ``basicConfig``) still receives records.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_ROOT = Path(__file__).resolve().parent.parent / "logs" / "storage"
_FORMAT = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_attached: set[str] = set()


def attach_module_file_logger(logger: logging.Logger) -> None:
    """Append ``logs/storage/<module_basename>.log`` for this logger name."""
    name = logger.name
    if name in _attached:
        return
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)
    basename = name.rsplit(".", maxsplit=1)[-1] if name else "storage"
    path = _LOG_ROOT / f"{basename}.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(_FORMAT)
    logger.addHandler(fh)
    _attached.add(name)
