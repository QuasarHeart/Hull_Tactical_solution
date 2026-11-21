from __future__ import annotations

import numpy as np
import polars as pl

from common.experiments import (
    causal_vol_estimates,
    position_turnover,
    score_positions,
    simulate_positions,
    simulate_vol_target,
)


def test_simulate_positions_matches_manual_policy():
    raw = np.array([0.001, -0.001, 0.0005])
    vol = np.array([0.1, 0.1, 0.1])
    multiplier = np.array([400.0, 400.0, 400.0])
    got = simulate_positions(raw, vol, multiplier)

    signal = np.clip(raw * multiplier, 0.0, 2.0)
    target = np.clip(signal / (vol * 1.2), 0.0, 2.0)
    expected = []
    last = 0.0
    for value in target:
        last = 0.75 * value + 0.25 * last
        last *= 1.0 - 3e-5
        expected.append(last)
    np.testing.assert_allclose(got, expected)


def test_vol_target_clips_and_smooths():
    got = simulate_vol_target(np.array([0.1, 10.0]), 0.2)
    # First target is clipped to 2.0, second to 0.02; smoothing keeps the second
    # above the raw target.
    assert got[0] > 1.4
    assert 0.02 < got[1] < 0.5


def test_cash_score_is_zero():
    result = score_positions(np.zeros(5), np.full(5, 0.01), np.full(5, 0.001))
    assert result["cash"] is True
    assert result["score"] == 0.0
    assert result["sharpe"] == 0.0


def test_position_turnover_starts_from_zero():
    assert position_turnover(np.array([0.5, 0.5])) == 0.25
    assert position_turnover(np.array([0.0, 0.0])) == 0.0


def test_causal_vol_estimates_are_causal():
    panel = pl.DataFrame(
        {
            "date_id": list(range(5)),
            "target": [0.01, -0.02, 0.03, -0.01, 0.02],
            "V1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    # For date_id=1 only target[0] is prior; changing target[4] must not change
    # the estimate for any earlier date.
    base = causal_vol_estimates(panel, [1, 2, 3])
    changed = panel.with_columns(
        pl.when(pl.col("date_id") == 4).then(999.0).otherwise(pl.col("target")).alias("target")
    )
    after = causal_vol_estimates(changed, [1, 2, 3])
    np.testing.assert_allclose(base, after)
