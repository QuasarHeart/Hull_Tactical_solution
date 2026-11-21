from __future__ import annotations

import numpy as np

from common.evaluation import (
    holm_adjust,
    linear_cost_returns,
    max_drawdown,
    paired_block_bootstrap,
    strategy_equity,
    summarize_positions,
    summarize_turnover_adjusted,
)


def test_max_drawdown_simple():
    equity = np.array([1.0, 1.2, 0.9, 1.1, 1.3])
    assert max_drawdown(equity) == -0.25


def test_cumulative_return_and_turnover():
    pos = np.array([1.0, 1.0])
    fwd = np.array([0.10, -0.05])
    rf = np.zeros(2)
    result = summarize_positions(pos, fwd, rf)
    expected = (1.10 * 0.95) - 1.0
    assert abs(result["cumulative_return"] - expected) < 1e-12
    assert abs(result["turnover"] - 0.5) < 1e-12


def test_turnover_adjusted_cost_is_zero_at_zero_rate():
    pos = np.array([1.0, 0.5, 1.5])
    fwd = np.array([0.02, -0.01, 0.03])
    rf = np.zeros(3)
    result = summarize_turnover_adjusted(pos, fwd, rf, cost_bps=0.0)
    assert abs(result["gross_sharpe"] - result["net_sharpe"]) < 1e-12
    assert abs(result["total_cost"]) < 1e-12


def test_paired_block_bootstrap_identical_paths():
    positions = {
        "2015": np.array([0.2, 0.4, 0.6, 0.5]),
        "2017": np.array([0.1, 0.3, 0.2]),
    }
    fwd = {key: np.linspace(0.01, 0.04, len(value)) for key, value in positions.items()}
    rf = {key: np.zeros(len(value)) for key, value in positions.items()}
    result = paired_block_bootstrap(
        positions, positions, fwd, rf, block_length=2, n_boot=100, seed=7
    )
    assert abs(result["actual_fold_mean_diff"]) < 1e-12
    assert abs(result["fold_ci_low"]) < 1e-12
    assert abs(result["fold_ci_high"]) < 1e-12
    assert result["fold_p_two_sided"] == 1.0
    assert result["actual_concat_diff"] == 0.0


def test_holm_adjust_monotone():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(value >= 0.0 for value in adjusted)
