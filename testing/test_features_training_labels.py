"""Training label helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from features.training_labels import add_open_to_close_return


def test_add_open_to_close_return() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "bar_ts": ["2024-01-02T14:30:00+00:00", "2024-01-03T14:30:00+00:00"],
            "open": [100.0, 101.0],
            "close": [102.0, 99.0],
        }
    )
    out = add_open_to_close_return(df)
    assert out["target_return_oc"].iloc[0] == pytest.approx(0.02)
    assert out["target_return_oc"].iloc[1] == pytest.approx(-0.01980198019801982, rel=1e-6)
