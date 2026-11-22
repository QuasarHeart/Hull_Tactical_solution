from __future__ import annotations

import numpy as np

from scripts.run_phase6_signal_robustness import hac_mean, signal_bootstrap


def test_hac_mean_matches_iid_variance_with_zero_lags() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
    result = hac_mean(values, max_lags=0)
    expected_se = float(np.std(values, ddof=0) / np.sqrt(len(values)))
    assert abs(result["mean"] - values.mean()) < 1e-12
    assert abs(result["se"] - expected_se) < 1e-12
    assert result["p_positive"] < 0.5


def test_signal_bootstrap_detects_monotonic_signal() -> None:
    prediction = np.linspace(-1.0, 1.0, 60)
    target = prediction * 0.01
    result = signal_bootstrap(
        prediction,
        target,
        block_length=5,
        reps=100,
        seed=11,
    )
    assert result["actual_rank_ic"] > 0.99
    assert result["actual_q5_minus_q1"] > 0.0
    assert result["n_boot"] == 100
