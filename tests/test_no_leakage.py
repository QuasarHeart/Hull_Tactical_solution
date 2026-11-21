"""A test row's features must not change when future panel rows are present."""

from __future__ import annotations

import numpy as np
import polars as pl

from common.config import Config
from common.data import load_panel
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder


def test_single_row_features_are_point_in_time() -> None:
    config = Config.load("configs/gdbt/notebook.yaml")
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    train_dates = panel["date_id"].to_list()[7000:7300]

    builder = NotebookFeatureBuilder(model_config)
    builder.fit(panel, train_dates)

    date = 7300
    full = builder.build(panel, [date], with_target=False).X
    truncated = builder.build(
        panel.filter(pl.col("date_id") <= date), [date], with_target=False
    ).X
    assert np.allclose(full, truncated, rtol=0.0, atol=0.0)
