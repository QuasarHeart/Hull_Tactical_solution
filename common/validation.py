"""Causal validation utilities: temporal nulls, block bootstrap, IC/regimes.

The functions here only reorder or resample already-realized test-period values;
they never change the model, the training window, or the official prediction
files.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _finite_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if len(a) != len(b):
        raise ValueError("IC inputs have different lengths")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def rank_ic(prediction: Any, target: Any) -> float:
    a, b = _finite_pair(prediction, target)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def pearson_ic(prediction: Any, target: Any) -> float:
    a, b = _finite_pair(prediction, target)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def block_permute(values: Any, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Permute contiguous blocks, preserving order inside each block."""
    arr = np.asarray(values).copy()
    n = len(arr)
    if n == 0:
        return arr
    length = max(1, min(int(block_length), n))
    blocks = [arr[start : start + length] for start in range(0, n, length)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])[:n]


def circular_shift(values: Any, shift: int) -> np.ndarray:
    """Rotate the sequence; a non-zero shift destroys date alignment."""
    return np.roll(np.asarray(values), int(shift))


def moving_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving-block bootstrap indices, concatenated and trimmed to length n."""
    if n <= 0:
        return np.empty(0, dtype=int)
    length = max(1, min(int(block_length), n))
    if length == n:
        return np.arange(n)
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=n_blocks)
    indices = np.concatenate([np.arange(start, start + length) for start in starts])
    return indices[:n].astype(int)


def quantile_returns(
    prediction: Any,
    target: Any,
    n_quantiles: int = 5,
) -> list[dict[str, float | int]]:
    """Equal-count prediction quantiles and their mean realized target."""
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    if len(pred) != len(y):
        raise ValueError("prediction/target lengths differ")
    mask = np.isfinite(pred) & np.isfinite(y)
    pred = pred[mask]
    y = y[mask]
    if len(pred) < n_quantiles:
        return []
    order = np.argsort(pred, kind="mergesort")
    groups = np.array_split(order, n_quantiles)
    return [
        {
            "quantile": i + 1,
            "count": int(len(group)),
            "mean_target": float(np.mean(y[group])),
            "median_target": float(np.median(y[group])),
        }
        for i, group in enumerate(groups)
    ]


def position_stats(positions: Any) -> dict[str, float]:
    pos = np.asarray(positions, dtype=float).reshape(-1)
    if len(pos) == 0:
        return {}
    turnover = (
        float(np.mean(np.abs(np.diff(np.r_[0.0, pos])))) if len(pos) > 1 else 0.0
    )
    return {
        "mean": float(np.mean(pos)),
        "median": float(np.median(pos)),
        "std": float(np.std(pos, ddof=1)) if len(pos) > 1 else 0.0,
        "q05": float(np.quantile(pos, 0.05)),
        "q25": float(np.quantile(pos, 0.25)),
        "q75": float(np.quantile(pos, 0.75)),
        "q95": float(np.quantile(pos, 0.95)),
        "min": float(np.min(pos)),
        "max": float(np.max(pos)),
        "zero_share": float(np.mean(pos <= 1e-9)),
        "near_zero_share": float(np.mean(pos <= 0.01)),
        "two_share": float(np.mean(pos >= 2.0 - 1e-9)),
        "near_two_share": float(np.mean(pos >= 1.99)),
        "turnover": turnover,
    }


def trailing_volatility(values: Any, window: int = 20) -> np.ndarray:
    """Strictly prior-window volatility; the first `window` values are NaN."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(int(window), len(arr)):
        recent = arr[i - int(window) : i]
        recent = recent[np.isfinite(recent)]
        if len(recent) > 1:
            out[i] = float(np.std(recent, ddof=1))
    return out


def regime_thresholds(train_values: Any, n_regimes: int = 3) -> np.ndarray:
    values = np.asarray(train_values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite training values for regime thresholds")
    quantiles = np.linspace(0, 1, int(n_regimes) + 1)[1:-1]
    return np.asarray(np.quantile(values, quantiles), dtype=float)


def regime_labels(values: Any, thresholds: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    labels = np.full(len(arr), -1, dtype=int)
    finite = np.isfinite(arr)
    labels[finite] = np.digitize(arr[finite], thresholds)
    return labels


def exposure_lag1(positions: Any) -> np.ndarray:
    """Yesterday's allocation, with 0 for the first day. Strictly causal."""
    pos = np.asarray(positions, dtype=float).reshape(-1)
    out = np.zeros_like(pos)
    if len(pos) > 1:
        out[1:] = pos[:-1]
    return out


def exposure_expanding_mean(positions: Any) -> np.ndarray:
    """Mean allocation observed strictly before each day; first day is 0."""
    pos = np.asarray(positions, dtype=float).reshape(-1)
    if len(pos) == 0:
        return pos.copy()
    out = np.zeros_like(pos)
    cumulative = np.cumsum(pos)
    for i in range(1, len(pos)):
        out[i] = cumulative[i - 1] / i
    return out


def exposure_vol_matched(
    positions: Any,
    vol: Any,
    vol_ref: float,
    max_position: float = 2.0,
) -> np.ndarray:
    """Match the prior average exposure while keeping volatility timing.

    pos_t = clip(c_t * mean(pos[:t]) * vol_ref / vol_t, 0, max_position)

    c_t is computed from data strictly before t so the *past* average of the
    resulting target matches the past average of the model's allocation. This
    keeps the baseline causal while making its average exposure comparable to
    the model's.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1)
    vol_arr = np.asarray(vol, dtype=float).reshape(-1)
    if len(pos) != len(vol_arr):
        raise ValueError("positions/vol lengths differ")
    exposure = exposure_expanding_mean(pos)
    raw_target = exposure * float(vol_ref) / np.maximum(vol_arr, 1e-12)
    out = np.zeros_like(raw_target)
    cumulative_pos = 0.0
    cumulative_out = 0.0
    for t in range(len(raw_target)):
        if t > 0 and cumulative_out > 0:
            scale = (cumulative_pos / t) / (cumulative_out / t)
        else:
            scale = 1.0
        out[t] = float(np.clip(scale * raw_target[t], 0.0, max_position))
        cumulative_pos += pos[t]
        cumulative_out += out[t]
    return out


def score_with_turnover_cost(
    positions: Any,
    forward_returns: Any,
    risk_free_rate: Any,
    cost_bps: float,
) -> dict[str, float]:
    """Official-style score with a linear cost on |position change|.

    The cost is charged once per unit of turnover (not the multiplicative
    haircut used inside the notebook policy). This is a diagnostic, not a
    replacement for the official score.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1)
    fwd = np.asarray(forward_returns, dtype=float).reshape(-1)
    rf = np.asarray(risk_free_rate, dtype=float).reshape(-1)
    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("positions/returns lengths differ")
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    strategy = rf * (1.0 - pos) + pos * fwd - (float(cost_bps) / 1e4) * turnover
    excess = strategy - rf
    if np.any(1.0 + excess <= 0.0):
        return {"score": float("nan"), "sharpe": float("nan")}
    strategy_mean = float(np.prod(1.0 + excess) ** (1.0 / len(excess)) - 1.0)
    market_excess = fwd - rf
    market_mean = float(np.prod(1.0 + market_excess) ** (1.0 / len(market_excess)) - 1.0)
    strategy_std = float(np.std(strategy, ddof=1))
    market_std = float(np.std(fwd, ddof=1))
    if strategy_std == 0.0 or market_std == 0.0:
        return {"score": float("nan"), "sharpe": float("nan")}
    sharpe = strategy_mean / strategy_std * np.sqrt(252.0)
    strategy_vol = strategy_std * np.sqrt(252.0) * 100.0
    market_vol = market_std * np.sqrt(252.0) * 100.0
    vol_penalty = 1.0 + max(0.0, strategy_vol / market_vol - 1.2)
    gap = max(0.0, (market_mean - strategy_mean) * 100.0 * 252.0)
    return_penalty = 1.0 + gap**2 / 100.0
    return {
        "score": float(sharpe / (vol_penalty * return_penalty)),
        "sharpe": float(sharpe),
    }
