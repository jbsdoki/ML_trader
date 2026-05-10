"""Tests for ``backtest.simulator``: net returns and equity compounding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.simulator import (
    equity_multiple,
    period_net_return,
    round_trip_cost_fraction,
    simulate_strategy_returns,
)


def test_round_trip_cost_fraction() -> None:
    assert round_trip_cost_fraction(10.0) == pytest.approx(0.001)
    assert round_trip_cost_fraction(0.0) == 0.0


def test_period_net_return_long_short_flat() -> None:
    assert period_net_return(1, 0.02, round_trip_cost_bps=10.0) == pytest.approx(0.02 - 0.001)
    assert period_net_return(-1, 0.02, round_trip_cost_bps=10.0) == pytest.approx(-0.02 - 0.001)
    assert period_net_return(0, 0.02, round_trip_cost_bps=10.0) == pytest.approx(0.0)


def test_period_net_return_zero_cost() -> None:
    assert period_net_return(1, 0.05, round_trip_cost_bps=0.0) == pytest.approx(0.05)


def test_period_net_return_coerces_invalid_action_to_flat() -> None:
    assert period_net_return(99, 0.03, round_trip_cost_bps=10.0) == pytest.approx(0.0)


def test_simulate_strategy_returns_vectorized() -> None:
    s = simulate_strategy_returns([1, -1, 0], [0.01, 0.01, 0.05], round_trip_cost_bps=10.0)
    assert len(s) == 3
    assert s.iloc[0] == pytest.approx(0.01 - 0.001)
    assert s.iloc[1] == pytest.approx(-0.01 - 0.001)
    assert s.iloc[2] == pytest.approx(0.0)


def test_simulate_strategy_returns_accepts_numpy_and_series() -> None:
    a = np.array([1.0, 0.0])
    r = pd.Series([0.02, 0.03])
    s = simulate_strategy_returns(a, r, round_trip_cost_bps=0.0)
    assert s.iloc[0] == pytest.approx(0.02)
    assert s.iloc[1] == pytest.approx(0.0)


def test_simulate_strategy_returns_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        simulate_strategy_returns([1], [0.01, 0.02])


def test_equity_multiple() -> None:
    m = equity_multiple([0.10, -0.05])
    assert m == pytest.approx(1.10 * 0.95)
