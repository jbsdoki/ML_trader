"""
Targets aligned with daily bars (e.g. session open-to-close return).
"""

from __future__ import annotations

import pandas as pd


def add_open_to_close_return(df: pd.DataFrame, *, out_col: str = "target_return_oc") -> pd.DataFrame:
    """
    Intraday return for the same bar row: ``close / open - 1``.

    Sorts by ``symbol`` then ``bar_ts`` before assigning (stable per-symbol series).
    Rows with non-positive or missing ``open``/``close`` should be dropped by the caller.
    """
    if df.empty:
        out = df.copy()
        out[out_col] = pd.Series(dtype=float)
        return out

    ordered = df.sort_values(["symbol", "bar_ts"], kind="mergesort")
    out = ordered.copy()
    o = pd.to_numeric(out["open"], errors="coerce")
    c = pd.to_numeric(out["close"], errors="coerce")
    out[out_col] = c / o - 1.0
    return out
