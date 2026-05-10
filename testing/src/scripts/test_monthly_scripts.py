"""Smoke tests for monthly XGBoost CLI scripts (file-backed SQLite + monkeypatched connect)."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from storage.bars_repo import upsert_bars
from storage.schema import init_schema

SYMBOL = "XXCLI"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_script_module(filename: str):
    root = _repo_root()
    path = root / "scripts" / filename
    name = filename.replace(".py", "_cli")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _xnys():
    return xcals.get_calendar("XNYS")


def _seed_walkforward_bars(conn: sqlite3.Connection) -> None:
    cal = _xnys()
    jan_anchor = pd.Timestamp("2025-01-31").normalize()
    assert cal.is_session(jan_anchor)
    last_bar_session = cal.next_session(jan_anchor)
    sessions = cal.sessions_in_range(pd.Timestamp("2024-04-01").normalize(), last_bar_session.normalize())
    close_map: dict[pd.Timestamp, float] = {}
    for i, s in enumerate(sessions):
        sn = pd.Timestamp(s).normalize()
        close_map[sn] = 100.0 + float(i) * 0.02
    rows: list[dict[str, Any]] = []
    for s in sessions:
        sn = pd.Timestamp(s).normalize()
        o = cal.session_open(sn)
        c = close_map[sn]
        rows.append(
            {
                "timestamp": o,
                "symbol": SYMBOL,
                "open": c - 0.1,
                "high": c + 0.2,
                "low": c - 0.2,
                "close": c,
                "volume": 1_000_000.0,
            }
        )
    upsert_bars(conn, pd.DataFrame(rows), "alpaca", "1d")


@pytest.fixture
def monthly_cli_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "monthly_cli.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    _seed_walkforward_bars(conn)
    conn.commit()
    conn.close()
    return db_path


def test_train_monthly_xgb_writes_model_and_meta(
    monthly_cli_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_connect(**kwargs: Any) -> sqlite3.Connection:
        c = sqlite3.connect(monthly_cli_db)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("storage.database.connect", fake_connect)
    model_path = tmp_path / "monthly_xgb.json"
    mod = _load_script_module("train_monthly_xgb.py")
    rc = mod.main(
        [
            "--symbols",
            SYMBOL,
            "--no-init-schema",
            "--bar-source",
            "alpaca",
            "--forward-sessions",
            "1",
            "--train-frac",
            "0.65",
            "--save-model",
            str(model_path),
            "--model-id",
            "finbert",
            "--bar-start",
            "2024-04-01",
            "--bar-end",
            "2025-12-31T23:59:59+00:00",
        ]
    )
    assert rc == 0
    assert model_path.is_file()
    meta = model_path.with_suffix(".meta.json")
    assert meta.is_file()


def test_predict_monthly_runs_after_train(
    monthly_cli_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_connect(**kwargs: Any) -> sqlite3.Connection:
        c = sqlite3.connect(monthly_cli_db)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("storage.database.connect", fake_connect)
    model_path = tmp_path / "m2.json"
    train = _load_script_module("train_monthly_xgb.py")
    assert (
        train.main(
            [
                "--symbols",
                SYMBOL,
                "--no-init-schema",
                "--bar-source",
                "alpaca",
                "--forward-sessions",
                "1",
                "--train-frac",
                "0.65",
                "--save-model",
                str(model_path),
                "--bar-start",
                "2024-04-01",
                "--bar-end",
                "2025-12-31T23:59:59+00:00",
            ]
        )
        == 0
    )

    pred = _load_script_module("predict_monthly.py")
    rc = pred.main(
        [
            "--model",
            str(model_path),
            "--symbols",
            SYMBOL,
            "--no-init-schema",
            "--bar-source",
            "alpaca",
            "--forward-sessions",
            "1",
            "--tau",
            "0.01",
            "--labeled-only",
            "--all-rows",
            "--bar-start",
            "2024-04-01",
            "--bar-end",
            "2025-12-31T23:59:59+00:00",
        ]
    )
    assert rc == 0


def test_backtest_monthly_tau_sweep(
    monthly_cli_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_connect(**kwargs: Any) -> sqlite3.Connection:
        c = sqlite3.connect(monthly_cli_db)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("storage.database.connect", fake_connect)
    bt = _load_script_module("backtest_monthly.py")
    rc = bt.main(
        [
            "--symbols",
            SYMBOL,
            "--no-init-schema",
            "--bar-source",
            "alpaca",
            "--forward-sessions",
            "1",
            "--min-train-rows",
            "2",
            "--max-train-rows",
            "12",
            "--taus",
            "0,0.01",
            "--bar-start",
            "2024-04-01",
            "--bar-end",
            "2025-12-31T23:59:59+00:00",
            "--max-history-days",
            "8000",
        ]
    )
    assert rc == 0


def test_train_monthly_xgb_no_rows_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "empty_bars.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.commit()
    conn.close()

    def fake_connect(**kwargs: Any) -> sqlite3.Connection:
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("storage.database.connect", fake_connect)
    mod = _load_script_module("train_monthly_xgb.py")
    rc = mod.main(
        [
            "--symbols",
            "NOSUCH",
            "--no-init-schema",
            "--bar-source",
            "alpaca",
            "--forward-sessions",
            "1",
        ]
    )
    assert rc == 1
