"""Evaluation metrics.

The headline metric is the competition's volatility-adjusted Sharpe ratio.
`official_score` is a vectorised, array-based version of the reference
`score` function. `proxy_score` is used for live rows where `forward_returns`
and `risk_free_rate` are not available yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
TRADING_DAYS_PER_YEAR = 252


class ParticipantVisibleError(Exception):
    pass


@dataclass
class ScoreReport:
    score: float
    sharpe: float
    strategy_vol: float
    market_vol: float
    vol_penalty: float
    return_penalty: float
    strategy_mean_excess: float
    market_mean_excess: float
    n_days: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf")
    return arr


def official_score(
    allocation: Any,
    forward_returns: Any,
    risk_free_rate: Any,
) -> ScoreReport:
    """Vectorised competition score.

    Mirrors `common.metrics.score` (pandas, ddof=1) to floating-point tolerance.
    """
    pos = _as_1d(allocation, "allocation")
    fwd = _as_1d(forward_returns, "forward_returns")
    rf = _as_1d(risk_free_rate, "risk_free_rate")
    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("allocation/forward_returns/risk_free_rate lengths differ")
    if pos.max() > MAX_INVESTMENT + 1e-12:
        raise ParticipantVisibleError(
            f"Position of {pos.max()} exceeds maximum of {MAX_INVESTMENT}"
        )
    if pos.min() < MIN_INVESTMENT - 1e-12:
        raise ParticipantVisibleError(
            f"Position of {pos.min()} below minimum of {MIN_INVESTMENT}"
        )

    strategy_returns = rf * (1.0 - pos) + pos * fwd
    excess = strategy_returns - rf
    market_excess = fwd - rf

    if np.any(1.0 + excess <= 0.0) or np.any(1.0 + market_excess <= 0.0):
        raise ParticipantVisibleError("Non-positive cumulative return in score window")

    strategy_mean_excess = float(np.prod(1.0 + excess) ** (1.0 / len(excess)) - 1.0)
    market_mean_excess = float(np.prod(1.0 + market_excess) ** (1.0 / len(market_excess)) - 1.0)

    strategy_std = float(np.std(strategy_returns, ddof=1))
    market_std = float(np.std(fwd, ddof=1))
    if strategy_std == 0.0:
        raise ParticipantVisibleError("Division by zero, strategy std is zero")
    if market_std == 0.0:
        raise ParticipantVisibleError("Division by zero, market std is zero")

    sharpe = strategy_mean_excess / strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_vol = strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    market_vol = market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0

    vol_penalty = 1.0 + max(0.0, strategy_vol / market_vol - 1.2)
    return_gap = max(0.0, (market_mean_excess - strategy_mean_excess) * 100.0 * TRADING_DAYS_PER_YEAR)
    return_penalty = 1.0 + (return_gap**2) / 100.0

    adjusted = sharpe / (vol_penalty * return_penalty)
    return ScoreReport(
        score=min(float(adjusted), 1_000_000.0),
        sharpe=float(sharpe),
        strategy_vol=float(strategy_vol),
        market_vol=float(market_vol),
        vol_penalty=float(vol_penalty),
        return_penalty=float(return_penalty),
        strategy_mean_excess=strategy_mean_excess,
        market_mean_excess=market_mean_excess,
        n_days=len(pos),
    )


def proxy_score(
    allocation: Any,
    target_metric: Any,
    risk_free_rate: Any | None = None,
) -> ScoreReport:
    """Score using the metric-aligned target.

    If rf is unknown, assume rf=0; the error is O(1-pos)*rf, which is tiny for
    daily data. If rf is known, reconstruct fwd = target_metric + rf exactly.
    """
    target = _as_1d(target_metric, "target_metric")
    if risk_free_rate is None:
        return official_score(allocation, target, np.zeros_like(target))
    rf = _as_1d(risk_free_rate, "risk_free_rate")
    return official_score(allocation, target + rf, rf)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Reference competition score function (kept for parity tests)."""
    if not pd.api.types.is_numeric_dtype(submission["prediction"]):
        raise ParticipantVisibleError("Predictions must be numeric")

    solution = solution
    solution["position"] = submission["prediction"]

    if solution["position"].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(
            f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}'
        )
    if solution["position"].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(
            f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}'
        )

    solution["strategy_returns"] = (
        solution["risk_free_rate"] * (1 - solution["position"])
        + solution["position"] * solution["forward_returns"]
    )

    strategy_excess_returns = solution["strategy_returns"] - solution["risk_free_rate"]
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution["strategy_returns"].std()
    if strategy_std == 0:
        raise ParticipantVisibleError("Division by zero, strategy std is zero")
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_volatility = float(strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

    market_excess_returns = solution["forward_returns"] - solution["risk_free_rate"]
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution["forward_returns"].std()
    market_volatility = float(market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
    if market_volatility == 0:
        raise ParticipantVisibleError("Division by zero, market std is zero")

    excess_vol = max(0, strategy_volatility / market_volatility - 1.2)
    vol_penalty = 1 + excess_vol
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * TRADING_DAYS_PER_YEAR,
    )
    return_penalty = 1 + (return_gap**2) / 100
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)


def r2_weighted(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
    y_true = _as_1d(y_true, "y_true")
    y_pred = _as_1d(y_pred, "y_pred")
    if sample_weight is None:
        w = np.ones_like(y_true)
    else:
        w = _as_1d(sample_weight, "sample_weight")
    numerator = np.sum(w * (y_pred - y_true) ** 2)
    denominator = np.sum(w * y_true**2) + 1e-38
    return float(1.0 - numerator / denominator)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def ic_spearman(y_true: Any, y_pred: Any) -> float:
    y_true = _as_1d(y_true, "y_true")
    y_pred = _as_1d(y_pred, "y_pred")
    if len(y_true) < 3:
        return float("nan")
    return float(np.corrcoef(_rank(y_true), _rank(y_pred))[0, 1])


def sharpe(returns: Any, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    returns = _as_1d(returns, "returns")
    std = np.std(returns, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def max_drawdown(equity: Any) -> float:
    equity = _as_1d(equity, "equity")
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def turnover(allocation: Any) -> float:
    pos = _as_1d(allocation, "allocation")
    if len(pos) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(pos))))


def bootstrap_score(
    allocation: Any,
    forward_returns: Any,
    risk_free_rate: Any,
    block: int = 20,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Circular block bootstrap of the whole score. Returns (mean, p2.5, p97.5)."""
    pos = _as_1d(allocation, "allocation")
    fwd = _as_1d(forward_returns, "forward_returns")
    rf = _as_1d(risk_free_rate, "risk_free_rate")
    n = len(pos)
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_boot):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            idx.extend((start + np.arange(block)) % n)
        take = np.asarray(idx[:n], dtype=int)
        try:
            scores.append(official_score(pos[take], fwd[take], rf[take]).score)
        except ParticipantVisibleError:
            continue
    if not scores:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(scores)
    return (float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
