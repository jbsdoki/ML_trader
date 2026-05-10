"""
Turn a **discrete action** from :mod:`policy.threshold` into a **dollar exposure** amount.

Why this exists
---------------
:func:`policy.threshold.pred_to_action` only answers *direction*: long (+1), short (-1),
or flat (0). It does **not** say how much capital is at risk. This module applies the
v1 rule: **one fixed notional per symbol** when the position is non-flat (default
**100 USD**), so backtests and (later) paper scripts can:

- convert **fractional** strategy returns (e.g. ``+0.02`` = +2% on the position) into
  **approximate dollar PnL** as ``signed_notional * strategy_return_fraction``, and
- keep **one place** to change stake size (``notional_usd``) without editing τ logic.

Signed convention
-----------------
- **Positive** dollars → **long** exposure (``action == +1``).
- **Negative** dollars → **short** exposure of the same **magnitude** (``action == -1``).
  The sign is bookkeeping for downstream code; it is **not** the model's predicted return.

This module does **not** compute share counts, margin, or broker rounding. Those belong
in execution or a separate sizing strategy later.
"""

from __future__ import annotations

from typing import Literal

Action = Literal[-1, 0, 1]


def signed_notional_from_action(
    action: int,
    *,
    notional_usd: float = 100.0,
) -> float:
    """
    Map ``action in {-1, 0, +1}`` to signed dollar notional (exposure magnitude).

    Mapping
    -------
    +1 (long)
        Return ``+notional_usd``: e.g. default ``100.0`` means about 100 USD notional
        on the long side for PnL scaling in backtests.
    -1 (short)
        Return ``-notional_usd``: same magnitude as long, negative sign indicates short.
    0 (flat)
        Return ``0.0``: no capital allocated to this leg.

    Relationship to predicted return
    ----------------------------------
    The regressor still outputs a **rate** (e.g. expected 21-day cumulative return as a
    fraction). ``notional_usd`` does **not** change that rate; it only scales **dollars**
    when you multiply rate × stake for reporting or order intent.

    Parameters
    ----------
    action
        Typically the return value of :func:`policy.threshold.pred_to_action`.
    notional_usd
        Absolute dollars per active side. Must be ``>= 0``. Default ``100.0`` matches
        the project v1 convention (100 USD per symbol per trade when non-flat).

    Returns
    -------
    float
        Signed notional in USD.

    Raises
    ------
    ValueError
        If ``action`` is not ``-1``, ``0``, or ``+1``, or ``notional_usd`` is negative.
    """
    if notional_usd < 0:
        raise ValueError(f"notional_usd must be >= 0, got {notional_usd!r}")
    if action not in (-1, 0, 1):
        raise ValueError(f"action must be -1, 0, or 1, got {action!r}")
    if action == 0:
        return 0.0
    if action == 1:
        return float(notional_usd)
    return -float(notional_usd)
