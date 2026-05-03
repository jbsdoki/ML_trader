"""
Financial Modeling Prep (FMP) — fundamentals and related JSON over HTTPS.

Requires env var ``FMP_API_KEY`` (or pass ``api_key=`` to functions).

**How this module is structured**

1. :func:`_fmp_api_key` — read the secret from the environment (after your CLI loads ``.env``).
2. :func:`fmp_get_json` — one place that builds the URL, adds ``apikey=``, GETs JSON, and
   parses the response. Add new FMP features by passing the path segment their docs show
   (e.g. ``profile/AAPL``, ``income-statement/AAPL``) or use the small wrappers below.
3. Optional wrappers (e.g. :func:`fetch_company_profile`) — normalize to a DataFrame for pipelines.

Confirm paths and query parameters in the official docs (plans differ by endpoint):
https://site.financialmodelingprep.com/developer/docs

Stable vs legacy base URLs may differ by endpoint; ``FMP_API_BASE`` below targets the common
``/api/v3`` prefix. If the docs specify ``/stable/...``, set the env var
``FMP_API_BASE`` to that root (no trailing slash) or adjust the constant.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)

# Default root for many documented endpoints; override with env FMP_API_BASE if your plan uses another.
_DEFAULT_FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _fmp_api_base() -> str:
    return (os.getenv("FMP_API_BASE") or _DEFAULT_FMP_BASE).strip().rstrip("/")


def _fmp_api_key(api_key: str | None = None) -> str:
    """Return FMP API key from the argument or ``FMP_API_KEY`` env; strip whitespace."""
    key = (api_key or os.getenv("FMP_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "FMP_API_KEY is missing. Set FMP_API_KEY in the environment or pass api_key= explicitly."
        )
    return key


def _fmp_parse_payload( data: Any, path: str) -> Any:
    # Raise if FMP returned an error
    if isinstance(data, dict) and data.get("Error Message"):
        msg = data["Error Message"]
        logger.error("FMP API error fo path=%s: %s", path, msg)
        raise ValueError(f"FMP error: {msg}")
    return data

def fmp_get_json( path: str, *, api_key: str | None = None, timeout_s: float = 60.0) -> Any:
    """
    GET one FMP endpoint and return parsed JSON (usually ``list`` or ``dict``).

    Parameters
    ----------
    path
        Path after the API base, e.g. ``profile/AAPL`` or ``income-statement/AAPL?limit=5``.
        Do not include the leading slash; ``apikey`` is appended by this function (if the path
        already has query params, ``apikey`` is appended with ``&``).
    api_key
        Optional override; defaults to ``FMP_API_KEY``.
    timeout_s
        Socket timeout in seconds.

    Use the site docs for the exact ``path`` and optional query string your subscription allows.
    """
    key = _fmp_api_key(api_key)
    segment = path.strip().lstrip("/")
    if "?" in segment:
        url = f"{_fmp_api_base()}/{segment}&apikey={quote(key, safe='')}"
    else:
        url = f"{_fmp_api_base()}/{segment}?apikey={quote(key, safe='')}"

    req = Request(url, headers={"User-Agent": "ML_Trader/fmp_ingest"})

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        logger.exception("FMP HTTP error path=%s code=%s", path, e.code)
        raise
    except URLError:
        logger.exception("FMP network error path=%s", path)
        raise

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("FMP invalid JSON path=%s", path)
        raise

    return _fmp_parse_payload(data, path)


def fetch_company_profile(symbol: str, *, api_key: str | None = None) -> pd.DataFrame:
    """
    Company profile for one ticker (FMP ``profile/{symbol}``).

    Returns a single-row DataFrame when data exists, else empty columns compatible with a profile row.
    """
    sym = symbol.strip().upper()
    try:
        data = fmp_get_json(f"profile/{sym}", api_key=api_key)
    except Exception:
        logger.exception("FMP profile fetch failed symbol=%s", sym)
        raise

    if not data:
        logger.warning("FMP returned no profile for %s", sym)
        return pd.DataFrame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    logger.warning("FMP profile unexpected shape for %s: %s", sym, type(data))
    return pd.DataFrame()


class FMPIngestor:
    """Optional holder for a default API key; delegates to module functions."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def get_json(self, path: str, *, api_key: str | None = None, timeout_s: float = 60.0) -> Any:
        return fmp_get_json(
            path,
            api_key=self.api_key if api_key is None else api_key,
            timeout_s=timeout_s,
        )

    def company_profile(self, symbol: str) -> pd.DataFrame:
        return fetch_company_profile(symbol, api_key=self.api_key)
