"""The local replay must match the submitted notebook's allocations."""

from __future__ import annotations

import numpy as np
import polars as pl

from common.config import Config
from common.data import load_panel
from gdbt.config import GBDTConfig
from scripts.notebook_reference import run_reference
from scripts.run_replay import build_pipeline


def test_notebook_allocations_match() -> None:
    config = Config.load("configs/gdbt/notebook.yaml")
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    reference = run_reference()

    pipeline = build_pipeline(model_config)
    train_dates = panel["date_id"].to_list()[-model_config.max_train_rows :]
    pipeline.fit(panel, train_dates)
    predictions = pipeline.predict_dates(panel, reference["date_id"].to_list())

    joined = predictions.join(reference, on="date_id", suffix="_ref")
    diff = (joined["allocation"] - joined["allocation_ref"]).abs().max()
    assert joined.height == reference.height == 10
    assert float(diff) < 1e-6
