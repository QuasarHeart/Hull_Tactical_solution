"""GBDT inference for the live panel (daily job)."""

from __future__ import annotations

from typing import Any

import polars as pl

from common.allocation import NotebookAllocationPolicy
from common.config import COL_DATE, COL_DATE_ID, COL_TARGET
from common.pipeline import FullPipeline

from .config import GBDTConfig
from .features import NotebookFeatureBuilder
from .model import SklearnEnsemble


def build_pipeline(model_config: GBDTConfig) -> FullPipeline:
    policy = NotebookAllocationPolicy(
        vol_window=model_config.vol_window,
        multiplier_low=model_config.signal_multiplier_low_vol,
        multiplier_high=model_config.signal_multiplier_high_vol,
        min_signal=model_config.min_signal,
        max_signal=model_config.max_signal,
        vol_scaling=model_config.vol_scaling,
        smoothing=model_config.smoothing,
        transaction_cost=model_config.transaction_cost,
    )
    return FullPipeline(
        predictor=SklearnEnsemble(model_config),
        builder=NotebookFeatureBuilder(model_config),
        policy=policy,
        online_update=False,
    )


def predict_latest(
    panel: pl.DataFrame,
    model_config: GBDTConfig,
    initial_allocation: float = 0.0,
    latest_date_id: int | None = None,
) -> dict[str, Any]:
    """Fit on the last N realized rows and produce one allocation for the latest row."""
    realized = panel.filter(pl.col(COL_TARGET).is_not_null())
    if realized.height < model_config.max_train_rows:
        raise ValueError(
            f"Only {realized.height} realized rows; need {model_config.max_train_rows}"
        )
    train_dates = realized[COL_DATE_ID].to_list()[-model_config.max_train_rows :]

    latest_id = int(latest_date_id) if latest_date_id is not None else int(panel[COL_DATE_ID].max())
    latest_rows = panel.filter(pl.col(COL_DATE_ID) == latest_id)
    if latest_rows.height != 1:
        raise ValueError(f"Expected one panel row for date_id={latest_id}, found {latest_rows.height}")
    latest_date = latest_rows[COL_DATE][0]

    pipeline = build_pipeline(model_config)
    pipeline.fit(panel, train_dates)
    predictions = pipeline.predict_dates(
        panel,
        [latest_id],
        initial_allocation=float(initial_allocation),
    )
    return {
        "predictions": predictions,
        "latest_id": latest_id,
        "latest_date": latest_date,
        "train_start": int(train_dates[0]),
        "train_end": int(train_dates[-1]),
        "n_train": len(train_dates),
        "features": list(pipeline.builder.features),
    }
