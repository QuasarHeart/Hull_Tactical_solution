"""Tabular dataset builder: exact transliteration of the submitted notebook.

The notebook's feature engineering is row-wise except for `I2_MA66`, which is
computed but not used, and the training-only lags, which are computed then
dropped. We reproduce that behaviour so the local replay matches the notebook.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from common.allocation import AllocationContext
from common.config import COL_DATE_ID, COL_TARGET
from common.model import Batch
from common.schema import FEATURE_PREFIXES

from .config import GBDTConfig

DERIVED_FEATURES = ["U1", "U2", "V1_S1", "M11_V1", "I9_S1"]
TRAIN_ONLY_FEATURES = [
    "S1_lag1",
    "P1_lag1",
    "I9_lag1",
    "V1_roll_mean_5",
    "target_roll_std_5",
]


class NotebookFeatureBuilder:
    """Reproduces load_trainset() + create_features() from the notebook."""

    lookback = 1

    def __init__(self, config: GBDTConfig) -> None:
        self.config = config
        self.features: list[str] = []
        self.raw_feature_columns: list[str] = []
        self.base_features: list[str] = []
        self.median_values: dict[str, float] = {}
        self.v1_median: float = 0.0
        self.train_frame: pl.DataFrame | None = None
        self._train_dates: set[Any] | None = None
        self._train_batch: Batch | None = None

    def fit(self, panel: pl.DataFrame, dates: Sequence[Any]) -> "NotebookFeatureBuilder":
        dates = list(dates)
        raw = panel.filter(pl.col(COL_DATE_ID).is_in(dates))
        if raw.height == 0:
            raise ValueError("NotebookFeatureBuilder.fit received no rows")
        # Live panels have pending targets at the tail; they must not enter the
        # training batch. Competition data has no null targets, so parity is
        # unchanged.
        raw = raw.filter(pl.col(COL_TARGET).is_not_null())
        raw = raw.filter(pl.col(COL_DATE_ID) >= self.config.min_date_id)
        raw = raw.tail(self.config.max_train_rows)

        missing = {col: raw[col].is_null().mean() for col in raw.columns}
        feature_cols = [
            col
            for col, miss_rate in missing.items()
            if miss_rate <= self.config.missing_threshold
            and col not in (COL_DATE_ID, COL_TARGET)
        ]
        if len(feature_cols) != len(set(feature_cols)):
            raise ValueError("Duplicate feature columns after missing-rate filter")

        frame = raw.select([COL_DATE_ID, COL_TARGET] + feature_cols)
        frame = self._create_features(frame, is_train=True, median_values=None)

        self.features = [
            col
            for col in frame.columns
            if col not in [COL_DATE_ID, COL_TARGET, *TRAIN_ONLY_FEATURES]
        ]
        self.base_features = [
            col
            for col in frame.columns
            if col.startswith(FEATURE_PREFIXES) and col not in DERIVED_FEATURES
        ]
        self.raw_feature_columns = list(self.base_features)
        self.median_values = {}
        for col in self.features:
            median = frame[col].median()
            self.median_values[col] = float(median) if median is not None else 0.0
        self.v1_median = float(frame["V1"].median()) if "V1" in frame.columns else 0.0
        self.train_frame = frame
        self._train_dates = set(frame[COL_DATE_ID].to_list())
        self._train_batch = self._make_batch(frame, with_target=True)
        return self

    def _create_features(
        self,
        df: pl.DataFrame,
        is_train: bool,
        median_values: dict[str, float] | None,
    ) -> pl.DataFrame:
        base_features = [col for col in df.columns if col.startswith(FEATURE_PREFIXES)]

        required = ["I1", "I2", "I7", "I9", "M11"]
        if all(col in base_features for col in required):
            df = df.with_columns(
                (pl.col("I2") - pl.col("I1")).alias("U1"),
                (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3)).alias("U2"),
                pl.col("I2").rolling_mean(window_size=66).alias("I2_MA66"),
            )
        else:
            df = df.with_columns(
                pl.lit(0.0).alias("U1"),
                pl.lit(0.0).alias("U2"),
            )

        if "V1" in base_features and "S1" in base_features:
            df = df.with_columns((pl.col("V1") * pl.col("S1")).alias("V1_S1"))
        if "M11" in base_features and "V1" in base_features:
            df = df.with_columns((pl.col("M11") * pl.col("V1")).alias("M11_V1"))
        if "I9" in base_features and "S1" in base_features:
            df = df.with_columns((pl.col("I9") * pl.col("S1")).alias("I9_S1"))

        # Robustness for early/short windows where an interaction input is
        # absent: the notebook only runs on the last 800 rows, so this branch is
        # never hit in parity, but it keeps the builder usable for CV/tests.
        for name in DERIVED_FEATURES:
            if name not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(name))

        if is_train:
            if "S1" in base_features:
                df = df.with_columns(pl.col("S1").shift(1).alias("S1_lag1"))
            if "P1" in base_features:
                df = df.with_columns(pl.col("P1").shift(1).alias("P1_lag1"))
            if "I9" in base_features:
                df = df.with_columns(pl.col("I9").shift(1).alias("I9_lag1"))
            if "V1" in base_features:
                df = df.with_columns(
                    pl.col("V1").rolling_mean(window_size=5).alias("V1_roll_mean_5")
                )
            if COL_TARGET in df.columns:
                df = df.with_columns(
                    pl.col(COL_TARGET).rolling_std(window_size=5).alias("target_roll_std_5")
                )

        # Notebook uses forward_fill -> backward_fill for I columns. This is
        # kept for parity; the strict/causal builder should drop backward_fill.
        for col in base_features:
            if col.startswith("I"):
                df = df.with_columns(
                    pl.col(col)
                    .fill_null(pl.col(col).forward_fill())
                    .fill_null(pl.col(col).backward_fill())
                )
            if median_values:
                median = median_values.get(col, df[col].median())
            else:
                median = df[col].median()
            df = df.with_columns(
                pl.col(col).fill_null(median if median is not None else 0.0)
            )

        additional_features = TRAIN_ONLY_FEATURES if is_train else []
        for col in DERIVED_FEATURES + additional_features:
            if col in df.columns:
                if median_values:
                    median = median_values.get(col, df[col].median())
                else:
                    median = df[col].median()
                df = df.with_columns(
                    pl.col(col).fill_null(median if median is not None else 0.0)
                )

        features = base_features + DERIVED_FEATURES
        select_cols = [COL_DATE_ID] + features + ([COL_TARGET] if is_train else [])
        return df.select(select_cols)

    def _make_batch(self, frame: pl.DataFrame, with_target: bool) -> Batch:
        X = frame.select(self.features).to_numpy()
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values in the training features")
        y = None
        if with_target and COL_TARGET in frame.columns:
            y = frame[COL_TARGET].to_numpy()
            if not np.isfinite(y).all():
                raise ValueError("Non-finite values in the training target")
        return Batch(
            X=X,
            y=y,
            dates=frame[COL_DATE_ID].to_numpy(),
            features=list(self.features),
        )

    def build(
        self,
        panel: pl.DataFrame,
        dates: Sequence[Any],
        with_target: bool,
    ) -> Batch:
        dates = list(dates)
        if self._train_dates is not None and set(dates) == self._train_dates:
            if self._train_batch is None:
                raise RuntimeError("Training batch was not built")
            return self._train_batch

        if not self.features:
            raise RuntimeError("NotebookFeatureBuilder.fit must be called before build")

        columns = [COL_DATE_ID] + self.raw_feature_columns
        if with_target:
            columns.append(COL_TARGET)
        raw = panel.filter(pl.col(COL_DATE_ID).is_in(dates)).select(columns)
        if raw.height != len(dates):
            found = set(raw[COL_DATE_ID].to_list())
            missing = [date for date in dates if date not in found]
            raise ValueError(f"Missing panel rows for dates: {missing[:5]}")

        frame = self._create_features(raw, is_train=False, median_values=self.median_values)
        frame = frame.sort(COL_DATE_ID)
        X = frame.select(self.features).to_numpy()
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite values in features for dates {dates[:3]}")

        y = None
        if with_target:
            labels = raw.sort(COL_DATE_ID)[COL_TARGET].to_numpy()
            y = labels
            if not np.isfinite(y).all():
                raise ValueError(f"Non-finite target for dates {dates[:3]}")

        return Batch(
            X=X,
            y=y,
            dates=frame[COL_DATE_ID].to_numpy(),
            features=list(self.features),
        )

    def context(self, panel: pl.DataFrame, date: Any) -> AllocationContext:
        row = panel.filter(pl.col(COL_DATE_ID) == date)
        if row.height != 1:
            raise ValueError(f"Expected one panel row for {date}, found {row.height}")
        v1 = row["V1"][0] if "V1" in row.columns else None
        return AllocationContext(
            date=date,
            v1=float(v1) if v1 is not None else None,
            v1_median=self.v1_median,
        )
