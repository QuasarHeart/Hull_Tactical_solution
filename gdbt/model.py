"""Sklearn GBDT ensemble implementing the shared Predictor protocol."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import ElasticNet

from common.model import Batch
from common.transforms import Standardizer

from .config import GBDTConfig


class SklearnEnsemble:
    """ElasticNet + XGBoost + LightGBM, weighted per the notebook config."""

    name = "gdbt"
    lookback = 1

    def __init__(self, config: GBDTConfig) -> None:
        self.config = config
        self.scaler = Standardizer()
        self.enet = ElasticNet(
            alpha=config.enet_alpha,
            l1_ratio=config.enet_l1_ratio,
            max_iter=1_000_000,
        )
        self.xgb = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.seed,
            verbosity=0,
        )
        self.lgb = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=config.lgb_n_estimators,
            max_depth=config.lgb_max_depth,
            learning_rate=config.lgb_learning_rate,
            random_state=config.seed,
            verbosity=-1,
        )
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.features: list[str] = []
        self._fitted = False

    def fit(self, train: Batch, valid: Batch | None = None) -> "SklearnEnsemble":
        if train.y is None:
            raise ValueError("SklearnEnsemble.fit requires train.y")
        X = np.asarray(train.X, dtype=float)
        y = np.asarray(train.y, dtype=float).reshape(-1)
        if not np.isfinite(X).all():
            raise ValueError("Training X contains NaN/inf")
        if not np.isfinite(y).all():
            raise ValueError("Training y contains NaN/inf")
        self._X = X.copy()
        self._y = y.copy()
        self.features = list(train.features)
        self._refit()
        return self

    def _refit(self) -> None:
        if self._X is None or self._y is None:
            raise RuntimeError("No training data stored")
        X_scaled = self.scaler.fit_transform(self._X)
        self.enet.fit(X_scaled, self._y)
        self.xgb.fit(X_scaled, self._y)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            self.lgb.fit(X_scaled, self._y)
        self._fitted = True

    def _components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("SklearnEnsemble is not fitted")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=float))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            lgb_pred = self.lgb.predict(X_scaled)
        return {
            "enet": np.asarray(self.enet.predict(X_scaled), dtype=float).reshape(-1),
            "xgb": np.asarray(self.xgb.predict(X_scaled), dtype=float).reshape(-1),
            "lgb": np.asarray(lgb_pred, dtype=float).reshape(-1),
        }

    def _combine(self, components: dict[str, np.ndarray]) -> np.ndarray:
        weights = self.config.weights()
        raw = (
            weights.get("enet", 0.0) * components["enet"]
            + weights.get("xgb", 0.0) * components["xgb"]
            + weights.get("lgb", 0.0) * components["lgb"]
        )
        if not self.config.normalize_weights:
            # Notebook parity: weights sum to 1.45. `weights()` only divides
            # when normalize_weights=True, so this branch is the raw sum.
            pass
        return np.asarray(raw, dtype=float)

    def predict(self, batch: Batch) -> np.ndarray:
        return self._combine(self._components(batch.X))

    def predict_components(self, batch: Batch) -> dict[str, np.ndarray]:
        components = self._components(batch.X)
        components["ensemble"] = self._combine(components)
        return components

    def update(self, batch: Batch) -> None:
        if batch.y is None:
            return
        if self._X is None or self._y is None:
            raise RuntimeError("Cannot update an unfitted model")
        self._X = np.vstack([self._X, np.asarray(batch.X, dtype=float)])
        self._y = np.concatenate([self._y, np.asarray(batch.y, dtype=float).reshape(-1)])
        if len(self._y) > self.config.max_train_rows:
            self._X = self._X[-self.config.max_train_rows :]
            self._y = self._y[-self.config.max_train_rows :]
        self._refit()

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnEnsemble":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SklearnEnsemble, got {type(obj)}")
        return obj

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": list(self.features),
            "weights": self.config.weights(),
            "normalize_weights": self.config.normalize_weights,
        }
