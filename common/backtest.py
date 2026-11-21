"""Backtest utilities for allocation series."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def equity_curve(
    allocation: Any,
    forward_returns: Any,
    risk_free_rate: Any,
    dates: Any | None = None,
) -> pl.DataFrame:
    pos = np.asarray(allocation, dtype=float).reshape(-1)
    fwd = np.asarray(forward_returns, dtype=float).reshape(-1)
    rf = np.asarray(risk_free_rate, dtype=float).reshape(-1)
    if not (len(pos) == len(fwd) == len(rf)):
        raise ValueError("allocation/forward_returns/risk_free_rate lengths differ")

    strategy = rf * (1.0 - pos) + pos * fwd
    strategy_equity = np.cumprod(1.0 + strategy)
    market_equity = np.cumprod(1.0 + fwd)
    running_peak = np.maximum.accumulate(strategy_equity)
    drawdown = strategy_equity / running_peak - 1.0

    data: dict[str, Any] = {
        "allocation": pos,
        "strategy_return": strategy,
        "market_return": fwd,
        "strategy_equity": strategy_equity,
        "market_equity": market_equity,
        "drawdown": drawdown,
    }
    if dates is not None:
        data = {"date": list(dates), **data}
    return pl.DataFrame(data)


def summary(curve: pl.DataFrame) -> dict[str, float]:
    strat = curve["strategy_return"].to_numpy()
    mkt = curve["market_return"].to_numpy()
    std = np.std(strat, ddof=1) if len(strat) > 1 else 0.0
    mstd = np.std(mkt, ddof=1) if len(mkt) > 1 else 0.0
    return {
        "total_return": float(curve["strategy_equity"][-1] - 1.0),
        "market_total_return": float(curve["market_equity"][-1] - 1.0),
        "annual_return": float(np.mean(strat) * 252),
        "annual_vol": float(std * np.sqrt(252)),
        "market_annual_vol": float(mstd * np.sqrt(252)),
        "sharpe": float(np.mean(strat) / std * np.sqrt(252)) if std > 0 else float("nan"),
        "max_drawdown": float(curve["drawdown"].min()),
        "avg_allocation": float(np.mean(curve["allocation"])),
    }
