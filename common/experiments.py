"""Position simulation primitives for baselines and prediction ablations.

The functions here are deliberately independent of the model family: they take
an already-computed prediction (or a volatility estimate) and reproduce the
notebook's sizing rules. This keeps the experiments apples-to-apples with the
saved CV OOF allocations.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from .metrics import official_score

DEFAULT_SMOOTHING = 0.25
DEFAULT_TRANSACTION_COST = 3e-5
DEFAULT_VOL_SCALING = 1.2
MAX_POSITION = 2.0


def _as_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf")
    return arr


def _smooth_targets(
    target: np.ndarray,
    smoothing: float,
    transaction_cost: float,
) -> np.ndarray:
    positions = np.empty_like(target, dtype=float)
    last = 0.0
    for i, value in enumerate(target):
        last = (1.0 - smoothing) * float(value) + smoothing * last
        last *= 1.0 - transaction_cost
        positions[i] = last
    return positions


def simulate_positions(
    raw_pred: Any,
    vol: Any,
    multiplier: Any,
    smoothing: float = DEFAULT_SMOOTHING,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    vol_scaling: float = DEFAULT_VOL_SCALING,
    max_position: float = MAX_POSITION,
) -> np.ndarray:
    """Reproduce the notebook policy from raw predictions and causal vol inputs."""
    raw = _as_1d(raw_pred, "raw_pred")
    vol = _as_1d(vol, "vol")
    multiplier = _as_1d(multiplier, "multiplier")
    if not (len(raw) == len(vol) == len(multiplier)):
        raise ValueError("raw_pred/vol/multiplier lengths differ")
    signal = np.clip(raw * multiplier, 0.0, max_position)
    target = np.clip(signal / (np.maximum(vol, 1e-12) * vol_scaling), 0.0, max_position)
    return _smooth_targets(target, smoothing, transaction_cost)


def simulate_vol_target(
    vol: Any,
    target_vol: Any,
    smoothing: float = DEFAULT_SMOOTHING,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    max_position: float = MAX_POSITION,
) -> np.ndarray:
    """Volatility-only sizing: target position = target_vol / current vol."""
    vol = _as_1d(vol, "vol")
    target_vol = np.asarray(target_vol, dtype=float)
    if target_vol.ndim == 0:
        target_vol = np.full_like(vol, float(target_vol))
    else:
        target_vol = _as_1d(target_vol, "target_vol")
        if len(target_vol) != len(vol):
            raise ValueError("target_vol/vol lengths differ")
    target = np.clip(target_vol / np.maximum(vol, 1e-12), 0.0, max_position)
    return _smooth_targets(target, smoothing, transaction_cost)


def position_turnover(positions: Any) -> float:
    pos = _as_1d(positions, "positions")
    if len(pos) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.r_[0.0, pos]))))


def score_positions(
    positions: Any,
    forward_returns: Any,
    risk_free_rate: Any,
) -> dict[str, Any]:
    """Score a position path with the official metric.

    A zero position has zero excess return and zero volatility; the official
    formula is undefined (0/0), so it gets the explicit cash convention
    score=sharpe=0.0. This is used only for the cash control.
    """
    pos = _as_1d(positions, "positions")
    fwd = _as_1d(forward_returns, "forward_returns")
    rf = _as_1d(risk_free_rate, "risk_free_rate")
    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("positions/forward_returns/risk_free_rate lengths differ")
    if np.allclose(pos, 0.0):
        return {
            "score": 0.0,
            "sharpe": 0.0,
            "strategy_vol": 0.0,
            "market_vol": 0.0,
            "vol_penalty": 1.0,
            "return_penalty": 1.0,
            "strategy_mean_excess": 0.0,
            "market_mean_excess": 0.0,
            "n_days": int(len(pos)),
            "mean_allocation": 0.0,
            "turnover": 0.0,
            "cash": True,
        }
    report = official_score(pos, fwd, rf)
    result = report.as_dict()
    result["mean_allocation"] = float(pos.mean())
    result["turnover"] = position_turnover(pos)
    result["cash"] = False
    return result


def causal_vol_estimates(
    panel: pl.DataFrame,
    dates: Sequence[Any],
    vol_window: int = 20,
    v1_column: str = "V1",
) -> np.ndarray:
    """Estimate the notebook's GARCH-like vol using only prior targets.

    For a date at index i, the estimate uses targets[i-window:i] (not the
    current target) and V1[i], matching the policy's causal history.
    """
    frame = panel.sort("date_id")
    target = frame["target"].to_numpy()
    v1 = frame[v1_column].to_numpy()
    index = {int(date_id): i for i, date_id in enumerate(frame["date_id"].to_list())}
    estimates: list[float] = []
    for date_id in dates:
        i = index[int(date_id)]
        recent = np.asarray(target[max(0, i - vol_window) : i], dtype=float)
        recent = recent[np.isfinite(recent)]
        current_v1 = float(v1[i]) if np.isfinite(v1[i]) else 0.01
        if len(recent) > 1:
            estimate = np.sqrt(0.3 * np.var(recent) + 0.7 * current_v1**2)
        else:
            estimate = current_v1
        estimates.append(max(float(estimate), 0.01))
    return np.asarray(estimates, dtype=float)
