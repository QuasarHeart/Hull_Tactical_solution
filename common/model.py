"""Shared model contracts.

A model family implements `Predictor`; the common pipeline owns the date loop,
the online-update protocol, the allocation policy and the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Batch:
    """A model-ready batch.

    GBDT: X is (N, F). DL: X is (N, L, F). y is the metric-aligned target
    (forward excess return), not the competition's `market_forward_excess_returns`.
    """

    X: np.ndarray
    y: np.ndarray | None = None
    aux: np.ndarray | None = None
    w: np.ndarray | None = None
    dates: list[Any] | np.ndarray | None = None
    features: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim not in (2, 3):
            raise ValueError(f"Batch.X must be 2D or 3D, got shape {self.X.shape}")
        n = len(self.X)
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=float).reshape(-1)
            if len(self.y) != n:
                raise ValueError(f"Batch.y length {len(self.y)} != X length {n}")
        if self.aux is not None:
            self.aux = np.asarray(self.aux, dtype=float)
            if len(self.aux) != n:
                raise ValueError(f"Batch.aux length {len(self.aux)} != X length {n}")
        if self.w is not None:
            self.w = np.asarray(self.w, dtype=float).reshape(-1)
            if len(self.w) != n:
                raise ValueError(f"Batch.w length {len(self.w)} != X length {n}")
        if self.dates is None:
            self.dates = np.arange(n)
        else:
            self.dates = np.asarray(self.dates, dtype=object)

    def __len__(self) -> int:
        return len(self.X)


@runtime_checkable
class Predictor(Protocol):
    """Minimal interface a model family must implement."""

    name: str
    lookback: int

    def fit(self, train: Batch, valid: Batch | None = None) -> Any:
        ...

    def predict(self, batch: Batch) -> np.ndarray:
        ...

    def update(self, batch: Batch) -> Any:
        ...

    def save(self, path: str | Path) -> Any:
        ...


@runtime_checkable
class DatasetBuilder(Protocol):
    """Turns a canonical panel into model-ready batches.

    Implementations decide whether a batch is tabular (GBDT) or a sequence (DL).
    """

    features: list[str]
    lookback: int

    def fit(self, panel: Any, dates: list[Any]) -> Any:
        ...

    def build(self, panel: Any, dates: list[Any], with_target: bool) -> Batch:
        ...

    def context(self, panel: Any, date: Any) -> Any:
        ...
