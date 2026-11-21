"""Pipeline smoke test: fit -> predict -> save/load on a small slice."""

from __future__ import annotations

import numpy as np

from common.allocation import NotebookAllocationPolicy
from common.config import Config
from common.data import load_panel
from common.pipeline import FullPipeline
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder
from gdbt.model import SklearnEnsemble


def _config() -> GBDTConfig:
    return GBDTConfig(
        max_train_rows=300,
        xgb_n_estimators=10,
        lgb_n_estimators=10,
        seed=0,
    )


def test_pipeline_smoke(tmp_path) -> None:
    config = Config.load("configs/gdbt/notebook.yaml")
    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    model_config = _config()
    train_dates = panel["date_id"].to_list()[7000:7300]
    test_dates = panel["date_id"].to_list()[7300:7305]

    policy = NotebookAllocationPolicy(
        vol_window=model_config.vol_window,
        multiplier_low=model_config.signal_multiplier_low_vol,
        multiplier_high=model_config.signal_multiplier_high_vol,
        vol_scaling=model_config.vol_scaling,
        smoothing=model_config.smoothing,
        transaction_cost=model_config.transaction_cost,
    )
    pipeline = FullPipeline(
        predictor=SklearnEnsemble(model_config),
        builder=NotebookFeatureBuilder(model_config),
        policy=policy,
    )
    pipeline.fit(panel, train_dates)
    predictions = pipeline.predict_dates(panel, test_dates)

    assert predictions.height == len(test_dates)
    allocation = predictions["allocation"].to_numpy()
    assert np.isfinite(allocation).all()
    assert allocation.min() >= 0.0
    assert allocation.max() <= 2.0

    model_path = tmp_path / "model.joblib"
    pipeline.predictor.save(model_path)
    loaded = SklearnEnsemble.load(model_path)
    batch = pipeline.builder.build(panel, [test_dates[0]], with_target=False)
    assert np.allclose(
        loaded.predict(batch), pipeline.predictor.predict(batch), rtol=0.0, atol=0.0
    )
