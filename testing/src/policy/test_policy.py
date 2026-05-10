"""Tests for ``policy`` package: threshold actions and signed notional."""

from __future__ import annotations

import pytest

import policy
from policy import pred_to_action, pred_to_action_many, signed_notional_from_action
from policy.sizing import signed_notional_from_action as sizing_notional
from policy.threshold import pred_to_action as thr_action


def test_policy_package_exports() -> None:
    assert set(policy.__all__) == {
        "pred_to_action",
        "pred_to_action_many",
        "signed_notional_from_action",
    }
    assert policy.pred_to_action is thr_action
    assert policy.signed_notional_from_action is sizing_notional


def test_pred_to_action_symmetric_band() -> None:
    assert pred_to_action(0.02, 0.01) == 1
    assert pred_to_action(-0.02, 0.01) == -1
    assert pred_to_action(0.005, 0.01) == 0
    assert pred_to_action(-0.005, 0.01) == 0
    assert pred_to_action(0.01, 0.01) == 0
    assert pred_to_action(-0.01, 0.01) == 0


def test_pred_to_action_tau_zero() -> None:
    assert pred_to_action(0.001, 0.0) == 1
    assert pred_to_action(-0.001, 0.0) == -1
    assert pred_to_action(0.0, 0.0) == 0


def test_pred_to_action_non_finite_is_flat() -> None:
    assert pred_to_action(float("nan"), 0.01) == 0
    assert pred_to_action(float("inf"), 0.01) == 0
    assert pred_to_action(float("-inf"), 0.01) == 0


def test_pred_to_action_negative_tau_raises() -> None:
    with pytest.raises(ValueError, match="tau"):
        pred_to_action(0.1, -0.01)


def test_pred_to_action_many() -> None:
    out = pred_to_action_many([0.02, -0.02, 0.0, float("nan")], 0.01)
    assert out == [1, -1, 0, 0]


def test_signed_notional_mapping() -> None:
    assert signed_notional_from_action(1, notional_usd=100.0) == 100.0
    assert signed_notional_from_action(-1, notional_usd=100.0) == -100.0
    assert signed_notional_from_action(0, notional_usd=100.0) == 0.0


def test_signed_notional_invalid_action_raises() -> None:
    with pytest.raises(ValueError, match="action"):
        signed_notional_from_action(2, notional_usd=100.0)
    with pytest.raises(ValueError, match="action"):
        signed_notional_from_action(-2, notional_usd=100.0)


def test_signed_notional_negative_stake_raises() -> None:
    with pytest.raises(ValueError, match="notional"):
        signed_notional_from_action(1, notional_usd=-1.0)


def test_signed_notional_custom_notional() -> None:
    assert signed_notional_from_action(1, notional_usd=250.5) == pytest.approx(250.5)
    assert signed_notional_from_action(-1, notional_usd=250.5) == pytest.approx(-250.5)


def test_signed_notional_accepts_int_like() -> None:
    assert signed_notional_from_action(int(1), notional_usd=100.0) == 100.0
