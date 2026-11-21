"""Prediction -> allocation.

The model's `raw_pred` is an expected excess return. The allocation policy is a
separate, small object fitted on the tune segment (or fixed, for notebook
parity). Keeping it separate lets GBDT and DL share exactly the same sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class AllocationContext:
    date: Any = None
    v1: float | None = None
    v1_median: float | None = None
    vol: float | None = None
    multiplier: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class AllocationPolicy(Protocol):
    def fit(self, train_batch: Any) -> None:
        ...

    def reset(self) -> None:
        ...

    def transform(self, raw_pred: float, context: AllocationContext) -> float:
        ...

    def update(self, realized_target: float) -> None:
        ...


class NotebookAllocationPolicy:
    """Exact reproduction of the notebook's sizing logic.

    vol = sqrt(0.3 * var(last 20 targets) + 0.7 * V1^2), min 0.01
    signal = clip(raw_pred * (600 if V1 < median else 400), 0, 2)
    allocation = clip(signal / (vol * 1.2), 0, 2)
    allocation = (0.75 * allocation + 0.25 * last) * (1 - 0.00003)
    """

    def __init__(
        self,
        vol_window: int = 20,
        multiplier_low: float = 600.0,
        multiplier_high: float = 400.0,
        min_signal: float = 0.0,
        max_signal: float = 2.0,
        vol_scaling: float = 1.2,
        smoothing: float = 0.25,
        transaction_cost: float = 0.00003,
    ) -> None:
        self.vol_window = int(vol_window)
        self.multiplier_low = float(multiplier_low)
        self.multiplier_high = float(multiplier_high)
        self.min_signal = float(min_signal)
        self.max_signal = float(max_signal)
        self.vol_scaling = float(vol_scaling)
        self.smoothing = float(smoothing)
        self.transaction_cost = float(transaction_cost)
        self.history: list[float] = []
        self.last_allocation = 0.0

    def fit(self, train_batch: Any) -> "NotebookAllocationPolicy":
        y = getattr(train_batch, "y", None)
        if y is None:
            raise ValueError("NotebookAllocationPolicy.fit requires train_batch.y")
        self.history = [float(v) for v in np.asarray(y, dtype=float)[-self.vol_window :]]
        self.last_allocation = 0.0
        return self

    def reset(self, initial_allocation: float = 0.0) -> None:
        self.last_allocation = float(initial_allocation)

    def update(self, realized_target: float) -> None:
        self.history.append(float(realized_target))
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def _estimate_vol(self, v1: float | None) -> float:
        recent = np.asarray(self.history[-self.vol_window :], dtype=float)
        if v1 is None:
            v1 = float(np.std(recent, ddof=0)) if len(recent) > 1 else 0.01
        v1 = float(v1)
        if len(recent) > 1:
            garch_vol = np.sqrt(0.3 * np.var(recent) + 0.7 * v1**2)
            return max(float(garch_vol), 0.01)
        return max(v1, 0.01)

    def transform(self, raw_pred: float, context: AllocationContext) -> float:
        v1 = context.v1 if context.v1 is not None else 0.0
        v1_median = context.v1_median if context.v1_median is not None else 0.0
        vol = self._estimate_vol(v1)
        multiplier = self.multiplier_low if v1 < v1_median else self.multiplier_high

        signal = float(np.clip(float(raw_pred) * multiplier, self.min_signal, self.max_signal))
        allocation = min(self.max_signal, max(self.min_signal, signal / (vol * self.vol_scaling)))
        allocation = (1.0 - self.smoothing) * allocation + self.smoothing * self.last_allocation
        allocation *= 1.0 - self.transaction_cost

        self.last_allocation = float(allocation)
        context.vol = vol
        context.multiplier = multiplier
        return float(allocation)

    def state_dict(self) -> dict[str, Any]:
        return {
            "vol_window": self.vol_window,
            "multiplier_low": self.multiplier_low,
            "multiplier_high": self.multiplier_high,
            "min_signal": self.min_signal,
            "max_signal": self.max_signal,
            "vol_scaling": self.vol_scaling,
            "smoothing": self.smoothing,
            "transaction_cost": self.transaction_cost,
            "last_allocation": self.last_allocation,
            "history": list(self.history),
        }


class LinearAllocationPolicy:
    """Two-parameter calibrated policy: pos = clip(a + b * pred / vol, 0, 2).

    Placeholder for the DL/tuning phase; `fit_predictions` is a hook the
    pipeline calls when tune predictions are available.
    """

    def __init__(
        self,
        a: float = 0.8,
        b: float = 1.0,
        min_position: float = 0.0,
        max_position: float = 2.0,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.min_position = float(min_position)
        self.max_position = float(max_position)

    def fit(self, train_batch: Any) -> "LinearAllocationPolicy":
        return self

    def reset(self) -> None:
        return None

    def update(self, realized_target: float) -> None:
        return None

    def transform(self, raw_pred: float, context: AllocationContext) -> float:
        vol = context.vol if context.vol is not None else 0.01
        pos = self.a + self.b * float(raw_pred) / max(float(vol), 1e-4)
        return float(np.clip(pos, self.min_position, self.max_position))
