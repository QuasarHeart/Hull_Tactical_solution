from __future__ import annotations

import numpy as np
import pytest

from common.validation import (
    block_permute,
    circular_shift,
    exposure_expanding_mean,
    exposure_lag1,
    exposure_vol_matched,
    moving_block_indices,
    position_stats,
    quantile_returns,
    rank_ic,
    regime_labels,
    regime_thresholds,
    score_with_turnover_cost,
    trailing_volatility,
)
from common.metrics import official_score


def test_block_permute_preserves_values_and_block_order():
    values = np.arange(10)
    out = block_permute(values, 3, np.random.default_rng(7))
    assert np.array_equal(np.sort(out), np.sort(values))
    # Reconstruct the original blocks and check each appears contiguously in the
    # same internal order.
    blocks = [list(values[i : i + 3]) for i in range(0, 10, 3)]
    out_list = list(out)
    for block in blocks:
        found = any(out_list[i : i + len(block)] == block for i in range(len(out_list)))
        assert found


def test_circular_shift_preserves_multiset_and_rotates():
    values = np.arange(6)
    out = circular_shift(values, 2)
    assert np.array_equal(out, np.array([4, 5, 0, 1, 2, 3]))


def test_moving_block_indices_are_valid_and_fixed_length():
    rng = np.random.default_rng(3)
    idx = moving_block_indices(25, 5, rng)
    assert len(idx) == 25
    assert idx.min() >= 0
    assert idx.max() < 25


def test_quantile_returns_are_monotone_for_monotone_signal():
    pred = np.arange(20, dtype=float)
    target = pred.copy()
    table = quantile_returns(pred, target, 5)
    means = [row["mean_target"] for row in table]
    assert means == sorted(means)
    assert table[-1]["mean_target"] > table[0]["mean_target"]


def test_rank_ic_sign():
    pred = np.arange(20, dtype=float)
    assert rank_ic(pred, pred) == pytest.approx(1.0)
    assert rank_ic(pred, -pred) == pytest.approx(-1.0)


def test_trailing_volatility_does_not_use_future_values():
    values = np.arange(10, dtype=float)
    base = trailing_volatility(values, 3)
    changed = values.copy()
    changed[-1] = 999.0
    after = trailing_volatility(changed, 3)
    assert np.allclose(base[:-1], after[:-1], equal_nan=True)


def test_regime_labels_use_fixed_thresholds():
    labels = regime_labels([0.1, 0.5, 0.9], [0.3, 0.7])
    assert labels.tolist() == [0, 1, 2]


def test_exposure_baselines_are_causal():
    pos = np.array([0.2, 0.4, 0.6, 0.8])
    lag = exposure_lag1(pos)
    expanding = exposure_expanding_mean(pos)
    assert lag.tolist() == [0.0, 0.2, 0.4, 0.6]
    assert np.allclose(expanding, [0.0, 0.2, 0.3, 0.4])
    changed = pos.copy()
    changed[-1] = 1.9
    assert np.array_equal(exposure_lag1(changed)[:-1], lag[:-1])
    assert np.allclose(exposure_expanding_mean(changed)[:-1], expanding[:-1])


def test_exposure_vol_matched_clips():
    pos = np.array([0.5, 0.5, 0.5])
    vol = np.array([0.1, 0.1, 0.1])
    out = exposure_vol_matched(pos, vol, 0.1)
    assert np.all(out <= 2.0)
    assert np.all(out >= 0.0)


def test_position_stats_bounds():
    stats = position_stats(np.array([0.0, 0.5, 2.0]))
    assert stats["zero_share"] == pytest.approx(1 / 3)
    assert stats["near_zero_share"] == pytest.approx(1 / 3)
    assert stats["two_share"] == pytest.approx(1 / 3)
    assert stats["near_two_share"] == pytest.approx(1 / 3)
    assert stats["max"] == 2.0


def test_turnover_cost_matches_official_at_zero():
    rng = np.random.default_rng(4)
    pos = np.clip(rng.normal(0.6, 0.4, 60), 0, 2)
    fwd = rng.normal(0.0005, 0.01, 60)
    rf = np.full(60, 0.0001)
    base = official_score(pos, fwd, rf)
    zero_cost = score_with_turnover_cost(pos, fwd, rf, 0.0)
    assert zero_cost["score"] == pytest.approx(base.score)
    assert zero_cost["sharpe"] == pytest.approx(base.sharpe)
    expensive = score_with_turnover_cost(pos, fwd, rf, 100.0)
    assert expensive["score"] < zero_cost["score"]
