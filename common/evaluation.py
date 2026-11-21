"""Shared position-path evaluation (score, Sharpe, return, drawdown, turnover)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .experiments import position_turnover, score_positions
from .metrics import ParticipantVisibleError, official_score, sharpe as annualized_sharpe
from .validation import moving_block_indices


def strategy_equity(
    positions: Any,
    forward_returns: Any,
    risk_free_rate: Any,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=float).reshape(-1)
    fwd = np.asarray(forward_returns, dtype=float).reshape(-1)
    rf = np.asarray(risk_free_rate, dtype=float).reshape(-1)
    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("positions/returns lengths differ")
    returns = rf * (1.0 - pos) + pos * fwd
    equity = np.cumprod(1.0 + returns)
    return equity, returns


def max_drawdown(equity: Any) -> float:
    values = np.asarray(equity, dtype=float).reshape(-1)
    if len(values) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    return float(np.min(drawdowns))


def summarize_positions(
    positions: Any,
    forward_returns: Any,
    risk_free_rate: Any,
) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=float).reshape(-1)
    result = score_positions(pos, forward_returns, risk_free_rate)
    if result.get("cash"):
        result.update(
            {
                "cumulative_return": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "mean_allocation": 0.0,
                "n_days": int(len(pos)),
            }
        )
        return result
    equity, _ = strategy_equity(pos, forward_returns, risk_free_rate)
    result.update(
        {
            "cumulative_return": float(equity[-1] - 1.0),
            "max_drawdown": max_drawdown(equity),
            "turnover": position_turnover(pos),
            "mean_allocation": float(pos.mean()),
            "n_days": int(len(pos)),
        }
    )
    return result


def linear_cost_returns(
    positions: Any,
    forward_returns: Any,
    risk_free_rate: Any,
    cost_bps: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gross strategy returns minus a linear cost on |position change|.

    The notebook policy already applies a multiplicative cost. This function is
    an additional, explicit stress diagnostic so configurations can be compared
    on a common turnover-cost basis.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1)
    fwd = np.asarray(forward_returns, dtype=float).reshape(-1)
    rf = np.asarray(risk_free_rate, dtype=float).reshape(-1)
    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("positions/returns lengths differ")
    gross = rf * (1.0 - pos) + pos * fwd
    changes = np.abs(np.diff(np.r_[0.0, pos]))
    costs = (float(cost_bps) / 1e4) * changes
    return gross - costs, costs


def summarize_turnover_adjusted(
    positions: Any,
    forward_returns: Any,
    risk_free_rate: Any,
    cost_bps: float = 10.0,
) -> dict[str, float]:
    """Gross and net-of-linear-turnover-cost summary metrics."""
    pos = np.asarray(positions, dtype=float).reshape(-1)
    gross, costs = linear_cost_returns(pos, forward_returns, risk_free_rate, cost_bps)
    gross_equity = np.cumprod(1.0 + gross)
    net_equity = np.cumprod(1.0 + gross - costs)
    return {
        "cost_bps": float(cost_bps),
        "gross_sharpe": float(annualized_sharpe(gross)),
        "net_sharpe": float(annualized_sharpe(gross - costs)),
        "gross_cumulative_return": float(gross_equity[-1] - 1.0),
        "net_cumulative_return": float(net_equity[-1] - 1.0),
        "gross_max_drawdown": max_drawdown(gross_equity),
        "net_max_drawdown": max_drawdown(net_equity),
        "total_cost": float(np.sum(costs)),
    }


def holm_adjust(p_values: Any) -> list[float]:
    """Holm step-down adjustment; preserves input order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (len(values) - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _score_for_bootstrap(
    positions: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
) -> float:
    if np.allclose(positions, 0.0):
        return 0.0
    try:
        return float(official_score(positions, forward_returns, risk_free_rate).score)
    except ParticipantVisibleError:
        # A resampled block can hit an undefined/degenerate score (zero
        # volatility or a non-positive cumulative path). Keep it out of the
        # finite bootstrap distribution rather than aborting the experiment.
        return float("nan")


def paired_block_bootstrap(
    positions_a: Mapping[str, Any],
    positions_b: Mapping[str, Any],
    forward_returns: Mapping[str, Any],
    risk_free_rate: Mapping[str, Any],
    block_length: int = 20,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired moving-block bootstrap for non-contiguous test folds.

    The same within-fold block indices are applied to A and B. The primary
    statistic is the mean of the six fold-level score differences. A secondary
    synthetic statistic concatenates the resampled fold segments in fold order;
    it is useful for comparison but is not a continuous investable track record.

    This is the common uncertainty interface for GBDT and later GRU/TCN models.
    """
    folds = list(positions_a)
    if not folds:
        raise ValueError("paired_block_bootstrap requires at least one fold")
    if set(folds) != set(positions_b):
        raise ValueError("positions_a/positions_b folds differ")
    if set(folds) != set(forward_returns) or set(folds) != set(risk_free_rate):
        raise ValueError("forward_returns/risk_free_rate folds differ")

    pos_a = {fold: np.asarray(positions_a[fold], dtype=float).reshape(-1) for fold in folds}
    pos_b = {fold: np.asarray(positions_b[fold], dtype=float).reshape(-1) for fold in folds}
    fwd = {fold: np.asarray(forward_returns[fold], dtype=float).reshape(-1) for fold in folds}
    rf = {fold: np.asarray(risk_free_rate[fold], dtype=float).reshape(-1) for fold in folds}
    for fold in folds:
        if not (len(pos_a[fold]) == len(pos_b[fold]) == len(fwd[fold]) == len(rf[fold])):
            raise ValueError(f"Fold {fold}: mismatched series lengths")

    actual_a = np.asarray(
        [_score_for_bootstrap(pos_a[fold], fwd[fold], rf[fold]) for fold in folds]
    )
    actual_b = np.asarray(
        [_score_for_bootstrap(pos_b[fold], fwd[fold], rf[fold]) for fold in folds]
    )
    fold_diffs = actual_a - actual_b
    actual_fold_mean = float(np.mean(fold_diffs))
    actual_concat_a = _score_for_bootstrap(
        np.concatenate([pos_a[fold] for fold in folds]),
        np.concatenate([fwd[fold] for fold in folds]),
        np.concatenate([rf[fold] for fold in folds]),
    )
    actual_concat_b = _score_for_bootstrap(
        np.concatenate([pos_b[fold] for fold in folds]),
        np.concatenate([fwd[fold] for fold in folds]),
        np.concatenate([rf[fold] for fold in folds]),
    )
    actual_concat = float(actual_concat_a - actual_concat_b)

    rng = np.random.default_rng(int(seed))
    boot_fold = np.empty(int(n_boot), dtype=float)
    boot_concat = np.empty(int(n_boot), dtype=float)
    for rep in range(int(n_boot)):
        diffs: list[float] = []
        parts_a: list[np.ndarray] = []
        parts_b: list[np.ndarray] = []
        parts_fwd: list[np.ndarray] = []
        parts_rf: list[np.ndarray] = []
        for fold in folds:
            indices = moving_block_indices(len(pos_a[fold]), int(block_length), rng)
            a = pos_a[fold][indices]
            b = pos_b[fold][indices]
            y = fwd[fold][indices]
            r = rf[fold][indices]
            diffs.append(
                _score_for_bootstrap(a, y, r) - _score_for_bootstrap(b, y, r)
            )
            parts_a.append(a)
            parts_b.append(b)
            parts_fwd.append(y)
            parts_rf.append(r)
        boot_fold[rep] = float(np.nanmean(diffs))
        boot_concat[rep] = _score_for_bootstrap(
            np.concatenate(parts_a),
            np.concatenate(parts_fwd),
            np.concatenate(parts_rf),
        ) - _score_for_bootstrap(
            np.concatenate(parts_b),
            np.concatenate(parts_fwd),
            np.concatenate(parts_rf),
        )

    def _summary(actual: float, values: np.ndarray) -> dict[str, Any]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return {
                "actual_diff": float(actual),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_two_sided": float("nan"),
                "significant_5pct": False,
            }
        low = float(np.percentile(finite, 2.5))
        high = float(np.percentile(finite, 97.5))
        p_two = min(
            1.0,
            2.0
            * min(
                (1.0 + float(np.sum(finite <= 0.0))) / (len(finite) + 1.0),
                (1.0 + float(np.sum(finite >= 0.0))) / (len(finite) + 1.0),
            ),
        )
        return {
            "actual_diff": float(actual),
            "ci_low": low,
            "ci_high": high,
            "p_two_sided": float(p_two),
            "significant_5pct": bool(low > 0.0 or high < 0.0),
        }

    fold_summary = _summary(actual_fold_mean, boot_fold)
    concat_summary = _summary(actual_concat, boot_concat)
    return {
        "folds": folds,
        "n_boot": int(n_boot),
        "block_length": int(block_length),
        "seed": int(seed),
        "actual_a": actual_a.tolist(),
        "actual_b": actual_b.tolist(),
        "fold_diffs": fold_diffs.tolist(),
        "actual_fold_mean_diff": actual_fold_mean,
        **{f"fold_{key}": value for key, value in fold_summary.items()},
        "actual_concat_diff": actual_concat,
        **{f"concat_{key}": value for key, value in concat_summary.items()},
        "method": (
            "paired moving-block bootstrap within each non-contiguous fold; "
            "fold_mean is primary; concatenated is a synthetic diagnostic"
        ),
    }

