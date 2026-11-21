"""Preprocessing transforms with fit-on-train semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass
class Standardizer:
    """Per-feature standardization, optionally clipped after scaling.

    Supports 2D (N, F) and 3D (N, L, F) arrays; statistics are always computed
    per feature across all samples/steps in the training set.
    """

    clip_value: float | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _reshape(self, X: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        arr = np.asarray(X, dtype=float)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Standardizer expects 2D or 3D input, got {arr.shape}")
        original = arr.shape
        flat = arr.reshape(-1, original[-1])
        return flat, original

    def fit(self, X: np.ndarray) -> "Standardizer":
        flat, _ = self._reshape(X)
        if not np.isfinite(flat).all():
            raise ValueError("Standardizer.fit received NaN/inf")
        self.mean_ = flat.mean(axis=0)
        scale = flat.std(axis=0, ddof=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        flat, original = self._reshape(X)
        out = (flat - self.mean_) / self.scale_
        if self.clip_value is not None:
            out = np.clip(out, -float(self.clip_value), float(self.clip_value))
        return out.reshape(original)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Standardizer, got {type(obj)}")
        return obj
