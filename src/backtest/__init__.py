"""
Backtest utilities: walk-forward splitting and return simulation with costs.

- ``walkforward``: chronological monthly train/test folds with optional history
  cap and capped training length (6–12 style windows).
- ``simulator``: map discrete actions + realized returns to net PnL fractions
  including round-trip bps.
"""

from __future__ import annotations

from .simulator import (
    equity_multiple,
    period_net_return,
    round_trip_cost_fraction,
    simulate_strategy_returns,
)
from .walkforward import (
    fold_count,
    iter_monthly_walk_forward_splits,
    split_train_test_by_anchor,
)

__all__ = [
    "equity_multiple",
    "fold_count",
    "iter_monthly_walk_forward_splits",
    "period_net_return",
    "round_trip_cost_fraction",
    "simulate_strategy_returns",
    "split_train_test_by_anchor",
]
