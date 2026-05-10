"""
Map a continuous return prediction to a discrete position {-1, 0, +1}.

Symmetric band: long if ``pred > +tau``, short if ``pred < -tau``, flat if
``-tau <= pred <= +tau`` (inclusive dead zone). ``tau`` is a **fraction** (e.g.
``0.01`` == 1%).

``tau == 0``: long if ``pred > 0``, short if ``pred < 0``, flat if ``pred == 0``.
Non-finite ``pred`` (NaN, inf) maps to **flat** (0).
"""

from __future__ import annotations

import math
from typing import Literal

Action = Literal[-1, 0, 1]


def pred_to_action(pred: float, tau: float) -> Action:
    """
    Convert predicted forward return ``pred`` to ``{-1, 0, +1}`` using half-band ``tau``.

    Parameters
    ----------
    pred
        Model-predicted return (same units as training label, typically a small fraction).
    tau
        Half-width of the no-trade zone; must be ``>= 0``. Same units as ``pred``.

    Returns
    -------
    - ``+1`` if ``pred > tau``
    - ``-1`` if ``pred < -tau``
    - ``0`` if ``-tau <= pred <= tau`` (inclusive), or if ``pred`` is not finite.
    """
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau!r}")
    if not math.isfinite(pred):
        return 0
    if pred > tau:
        return 1
    if pred < -tau:
        return -1
    return 0


def pred_to_action_many(preds: list[float], tau: float) -> list[Action]:
    """Apply :func:`pred_to_action` element-wise (convenience for batch / tests)."""
    return [pred_to_action(p, tau) for p in preds]
