"""
Persist OHLCV bar DataFrames (yfinance, Alpaca) with deduplication.

Bars are keyed by ``(source_api, symbol, bar_interval, bar_ts)`` so re-fetching
the same history does not duplicate rows. Uses ``INSERT OR IGNORE``.

Useful references
-----------------
- SQLite composite primary keys:
  https://www.sqlite.org/lang_createtable.html
- Alpaca bar object (fields like vwap, trade_count):
  https://docs.alpaca.markets/reference/stockbars
- yfinance ``history()`` output:
  https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ._file_log import attach_module_file_logger

logger = logging.getLogger(__name__)
attach_module_file_logger(logger)


def _utc_bar_ts_iso(ts: Any) -> str:
    """
    Normalize a bar timestamp to UTC ISO-8601 string for the primary key.

    Bar **open** time must be consistent across runs so the same candle always
    maps to the same ``bar_ts``.
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        raise ValueError("Bar timestamp is missing; cannot build dedupe key.")
    t = pd.Timestamp(ts)
    if t is pd.NaT:
        raise ValueError("Bar timestamp is NaT; cannot build dedupe key.")
    if t.tzinfo is None:
        t = t.tz_localize(timezone.utc)
    else:
        t = t.tz_convert(timezone.utc)
    return t.isoformat()


def _ingested_now_iso() -> str:
    """UTC time when the row was written (audit)."""
    return datetime.now(timezone.utc).isoformat()


def _bars_df_to_records(
    df: pd.DataFrame,
    source_api: str,
    bar_interval: str,
) -> list[dict[str, Any]]:
    """
    Convert an ingest DataFrame into flat dicts ready for SQL binding.

    Accepts either a ``timestamp`` column or a DatetimeIndex (resets index if needed).
    Required OHLCV columns: ``open``, ``high``, ``low``, ``close``, ``volume``
    (after lowercasing — matches ``data.yfinance_ingest`` / ``data.alpaca_ingest``).
    """
    if df is None or df.empty:
        return []

    work = df.copy()
    if isinstance(work.index, pd.DatetimeIndex):
        work = work.reset_index()
        # yfinance sets index name to 'timestamp'; reset_index creates column
        ts_col = work.columns[0] if "timestamp" not in work.columns else "timestamp"
        if ts_col != "timestamp" and work.columns[0] != "timestamp":
            work = work.rename(columns={work.columns[0]: "timestamp"})

    if "timestamp" not in work.columns:
        raise ValueError("bars DataFrame must have a 'timestamp' column or DatetimeIndex.")

    src = source_api.strip().lower()
    interval = bar_interval.strip().lower()
    records: list[dict[str, Any]] = []

    for _, row in work.iterrows():
        sym = row.get("symbol")
        if sym is None or (isinstance(sym, float) and pd.isna(sym)):
            raise ValueError("Each bar row must include a 'symbol' column.")
        sym_str = str(sym).upper()
        bar_ts = _utc_bar_ts_iso(row["timestamp"])

        def _f(col: str) -> float | None:
            """Safely read a float column from a row (missing or NaN -> None)."""
            if col not in row.index:
                return None
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return float(v)

        def _i(col: str) -> int | None:
            """Safely read an int column (e.g. Alpaca ``trade_count``)."""
            if col not in row.index:
                return None
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return int(v)

        # yfinance lowercases split column to ``stock splits`` (with space); Alpaca may differ.
        split_val = _f("stock splits")
        if split_val is None:
            split_val = _f("stock_splits")

        records.append(
            {
                "source_api": src,
                "symbol": sym_str,
                "bar_interval": interval,
                "bar_ts": bar_ts,
                "open": _f("open"),
                "high": _f("high"),
                "low": _f("low"),
                "close": _f("close"),
                "volume": _f("volume"),
                "vwap": _f("vwap"),
                "trade_count": _i("trade_count"),
                "dividends": _f("dividends"),
                "stock_splits": split_val,
            }
        )

    return records


def upsert_bars(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source_api: str,
    bar_interval: str,
) -> int:
    """
    Insert OHLCV rows; ignore rows that match an existing primary key.

    Parameters
    ----------
    conn
        SQLite connection with ``bars`` table initialized.
    df
        Output of ``fetch_ohlcv`` / ``fetch_stock_bars`` (lowercase OHLCV columns).
    source_api
        e.g. ``\"yfinance\"`` or ``\"alpaca\"``.
    bar_interval
        Must match the request (e.g. ``\"1d\"``, ``\"1h\"``) — not always inferable from df.

    Returns
    -------
    int
        Number of rows **newly inserted** (SQLite ``total_changes`` delta).
    """
    records = _bars_df_to_records(df, source_api, bar_interval)
    if not records:
        return 0

    before = conn.total_changes
    sql = """
        INSERT OR IGNORE INTO bars (
            source_api, symbol, bar_interval, bar_ts,
            open, high, low, close, volume,
            vwap, trade_count, dividends, stock_splits,
            ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    ingested = _ingested_now_iso()
    cur = conn.cursor()
    for r in records:
        cur.execute(
            sql,
            (
                r["source_api"],
                r["symbol"],
                r["bar_interval"],
                r["bar_ts"],
                r["open"],
                r["high"],
                r["low"],
                r["close"],
                r["volume"],
                r["vwap"],
                r["trade_count"],
                r["dividends"],
                r["stock_splits"],
                ingested,
            ),
        )

    conn.commit()
    inserted = conn.total_changes - before
    logger.info(
        "bars upsert: source=%s interval=%s rows_attempted=%s rows_inserted=%s",
        records[0]["source_api"],
        records[0]["bar_interval"],
        len(records),
        inserted,
    )
    return inserted


def fetch_bars_frame(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    bar_interval: str,
    source_api: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Read bars back for one symbol and interval (optional filter by ``source_api``).
    """
    q = """
        SELECT * FROM bars
        WHERE symbol = ? AND bar_interval = ?
    """
    params: list[Any] = [symbol.upper(), bar_interval.strip().lower()]
    if source_api:
        q += " AND source_api = ?"
        params.append(source_api.strip().lower())
    q += " ORDER BY bar_ts ASC"
    if limit is not None:
        q += f" LIMIT {int(limit)}"

    return pd.read_sql_query(q, conn, params=params)


def _bar_ts_bound_iso(value: Any) -> str:
    """Normalize a date bound to UTC ISO-8601 for ``bars.bar_ts`` comparisons."""
    t = pd.Timestamp(value)
    if t is pd.NaT:
        raise ValueError("Invalid bar_ts bound.")
    if t.tzinfo is None:
        t = t.tz_localize(timezone.utc)
    else:
        t = t.tz_convert(timezone.utc)
    return t.isoformat()


def fetch_bars_multi_symbol_frame(
    conn: sqlite3.Connection,
    *,
    symbols: list[str],
    bar_interval: str,
    source_api: str | None = None,
    bar_ts_start: str | None = None,
    bar_ts_end: str | None = None,
) -> pd.DataFrame:
    """
    Load bars for multiple symbols (same interval), optionally one vendor and time bounds.
    """
    cleaned = [x.strip().upper() for x in symbols if x and str(x).strip()]
    if not cleaned:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(cleaned))
    q = f"""
        SELECT * FROM bars
        WHERE symbol IN ({placeholders}) AND bar_interval = ?
    """
    params: list[Any] = list(cleaned)
    params.append(bar_interval.strip().lower())

    if source_api:
        q += " AND source_api = ?"
        params.append(source_api.strip().lower())

    if bar_ts_start:
        q += " AND bar_ts >= ?"
        params.append(_bar_ts_bound_iso(bar_ts_start))

    if bar_ts_end:
        q += " AND bar_ts <= ?"
        params.append(_bar_ts_bound_iso(bar_ts_end))

    q += " ORDER BY symbol ASC, bar_ts ASC"
    return pd.read_sql_query(q, conn, params=params)
