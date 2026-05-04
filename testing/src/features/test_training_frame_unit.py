"""Training frame: column contract without full DB."""

from __future__ import annotations

from features.training_frame import default_training_feature_columns


def test_default_training_feature_columns_order() -> None:
    cols = default_training_feature_columns()
    assert cols == ["sentiment_mean", "sentiment_n", "sentiment_std"]
