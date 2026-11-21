"""Shared feature engineering must be causal."""

from __future__ import annotations

import numpy as np
import polars as pl

from common.features import add_calendar_features, add_target_history_features


def test_target_history_is_shifted() -> None:
    target = np.arange(1.0, 21.0)
    df = pl.DataFrame({"target": target})
    out = add_target_history_features(df, "target", windows=(5,))
    values = out["target_mean_5"].to_numpy()
    # Row 5 must use targets[0:5], not targets[1:6] or targets[5].
    assert np.isclose(values[5], np.mean(target[0:5]))
    assert np.isclose(values[6], np.mean(target[1:6]))
    # The first row (and first four) are null / nan.
    assert np.isnan(values[0])


def test_calendar_features() -> None:
    df = pl.DataFrame({"date": ["2025-01-02", "2025-12-31"]}).with_columns(
        pl.col("date").str.to_date()
    )
    out = add_calendar_features(df)
    assert out["cal_month"].to_list() == [1, 12]
    assert out["cal_quarter"].to_list() == [1, 4]
