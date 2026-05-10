"""
Translate **actions** and **realized returns** into per-period PnL **fractions**.

The policy layer (see ``policy.threshold``) emits ``-1 / 0 / +1``. The label from
``features.monthly_forward_frame`` is a simple forward return *R* on the
underlying. A long earns ``+R``, a short earns ``-R``, flat earns ``0``.

**Costs:** when ``action != 0``, subtract a combined **round-trip** cost expressed
in basis points (default **10 bps** → 0.001 as a return fraction). Flat trades
pay no spread/commission in this v1 model.

Compounded equity is optional: ``equity_multiple`` multiplies ``(1 + r_i)``
across periods. Dollar PnL with fixed notional scales linearly with these
fractions (see ``policy.sizing``).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def round_trip_cost_fraction(round_trip_cost_bps: float) -> float:
    """Convert basis points to a **return fraction** (e.g. 10 → 0.001)."""
    return float(round_trip_cost_bps) / 10_000.0


def _coerce_action(a: int | float) -> int:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return 0
    i = int(a)
    if i not in (-1, 0, 1):
        return 0
    return i


def period_net_return(
    action: int | float,
    realized_return: float,
    *,
    round_trip_cost_bps: float = 10.0,
) -> float:
    """
    Net **simple** return for one period after round-trip cost (if non-flat).

    ``realized_return`` should match the horizon used to build labels (e.g.
    ``forward_return`` from the monthly frame).
    """
    act = _coerce_action(action)
    r = float(realized_return)
    gross = act * r
    if act == 0:
        return gross
    return gross - round_trip_cost_fraction(round_trip_cost_bps)


def simulate_strategy_returns(
    actions: Sequence[int | float] | pd.Series | np.ndarray,
    realized_returns: Sequence[float] | pd.Series | np.ndarray,
    *,
    round_trip_cost_bps: float = 10.0,
) -> pd.Series:
    """
    Vectorized net returns for a sequence of periods.

    Raises ``ValueError`` if lengths differ.
    """
    a = pd.Series(actions, dtype="float64").map(_coerce_action)
    r = pd.Series(realized_returns, dtype="float64").astype(float)
    if len(a) != len(r):
        raise ValueError("actions and realized_returns must have the same length.")
    cost = round_trip_cost_fraction(round_trip_cost_bps)
    gross = a.astype(float) * r
    paid = (a != 0).astype(float) * cost
    return gross - paid


def equity_multiple(period_returns: pd.Series | Sequence[float]) -> float:
    """Product of ``(1 + r)`` across periods (no cash yield)."""
    s = pd.Series(period_returns, dtype="float64")
    return float((1.0 + s).prod())
