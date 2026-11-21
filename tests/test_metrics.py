"""Metric parity and diagnostic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.metrics import (
    bootstrap_score,
    ic_spearman,
    official_score,
    proxy_score,
    r2_weighted,
    score,
)


def _sample(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    allocation = rng.uniform(0.0, 2.0, size=n)
    forward = rng.normal(0.0005, 0.01, size=n)
    risk_free = np.full(n, 0.0001)
    return allocation, forward, risk_free


def test_official_score_matches_reference() -> None:
    allocation, forward, risk_free = _sample()
    expected = score(
        pd.DataFrame({"forward_returns": forward, "risk_free_rate": risk_free}),
        pd.DataFrame({"prediction": allocation}),
        "row_id",
    )
    actual = official_score(allocation, forward, risk_free).score
    assert np.isclose(actual, expected, rtol=0.0, atol=1e-10)


def test_proxy_score_with_rf_reconstructs_official() -> None:
    allocation, forward, risk_free = _sample(seed=1)
    target_metric = forward - risk_free
    official = official_score(allocation, forward, risk_free)
    proxy = proxy_score(allocation, target_metric, risk_free)
    assert np.isclose(proxy.score, official.score, rtol=0.0, atol=1e-12)


def test_diagnostics() -> None:
    assert np.isclose(r2_weighted(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 1.0)
    assert ic_spearman(np.arange(10), np.arange(10)) > 0.99
    mean, low, high = bootstrap_score(*_sample(seed=2), n_boot=50, block=10)
    assert np.isfinite(mean)
    assert low <= high
