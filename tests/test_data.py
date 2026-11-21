"""Data loading and schema tests."""

from __future__ import annotations

import numpy as np

from common.config import COL_TARGET_METRIC, Config
from common.data import load_competition_test, load_date_map, load_panel
from common.schema import N_FEATURES, feature_columns


def test_competition_panel_shape() -> None:
    config = Config.load("configs/gdbt/notebook.yaml")
    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    assert panel.height == 9048
    assert len(feature_columns(panel)) == N_FEATURES
    assert panel["date_id"].min() == 0
    assert panel["date_id"].max() == 9047
    # target_metric is fwd - rf, not the competition's target.
    diff = panel[COL_TARGET_METRIC] - (panel["forward_returns"] - panel["risk_free_rate"])
    assert float(np.abs(diff.to_numpy()).max()) < 1e-12


def test_test_frame() -> None:
    config = Config.load("configs/gdbt/notebook.yaml")
    test = load_competition_test(config.paths.test_csv)
    assert test.height == 10
    assert test["date_id"].to_list() == list(range(8980, 8990))
    assert int(test["is_scored"].sum()) == 9


def test_date_map() -> None:
    config = Config.load("configs/gdbt/notebook.yaml")
    dates = load_date_map(config.paths.date_map_csv)
    assert dates.height == 9291
    assert dates["date_id"].min() == 0
    assert dates["date_id"].max() == 9290
