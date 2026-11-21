"""GBDT configuration (dataclass mirror of configs/gdbt/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class GBDTConfig:
    # Notebook reproduction
    max_train_rows: int = 800
    min_date_id: int = 37
    missing_threshold: float = 0.5
    online_update: bool = False
    retrain_freq: int = 1

    # Estimators
    enet_alpha: float = 0.01
    enet_l1_ratio: float = 0.5
    xgb_n_estimators: int = 350
    xgb_max_depth: int = 10
    xgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 10
    lgb_learning_rate: float = 0.05
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {"enet": 0.45, "xgb": 0.55, "lgb": 0.45}
    )
    normalize_weights: bool = False

    # Allocation
    vol_window: int = 20
    signal_multiplier_low_vol: float = 600.0
    signal_multiplier_high_vol: float = 400.0
    min_signal: float = 0.0
    max_signal: float = 2.0
    vol_scaling: float = 1.2
    smoothing: float = 0.25
    transaction_cost: float = 0.00003

    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GBDTConfig":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def weights(self) -> dict[str, float]:
        weights = {key: float(value) for key, value in self.ensemble_weights.items()}
        if self.normalize_weights:
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("ensemble_weights must sum to a positive value")
            weights = {key: value / total for key, value in weights.items()}
        return weights
