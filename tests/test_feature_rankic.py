from __future__ import annotations

import numpy as np

from scripts.run_phase8_feature_rankic import (
    bh_adjust,
    derived_feature_values,
    flag_feature,
    resolve_oof_path,
    stability_label,
)


def test_bh_adjust_known_values() -> None:
    adjusted = bh_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_stability_label_requires_four_folds() -> None:
    assert stability_label([0.1, 0.08, 0.06], 3) == "insufficient"
    assert stability_label([0.1, 0.08, 0.06, 0.04, 0.02, 0.01], 6) == "stable_positive"
    assert stability_label([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01], 6) == "stable_negative"
    assert stability_label([0.3, -0.01, 0.01, -0.01, 0.01, -0.01], 6) == "single_period"
    assert stability_label([0.2, -0.1, 0.1, -0.2, 0.1, 0.0], 6) == "mixed"


def test_derived_features_are_rowwise() -> None:
    raw_features = ["I1", "I2", "I7", "I9", "M11", "V1", "S1"]
    raw = np.array(
        [
            [1.0, 3.0, 2.0, 4.0, 6.0, 2.0, 5.0],
            [2.0, 5.0, 2.0, 2.0, 3.0, -1.0, 2.0],
        ]
    )
    derived = derived_feature_values(raw, raw_features)
    assert np.allclose(derived[:, 0], [2.0, 3.0])
    assert np.allclose(derived[:, 1], [2.0, 1.0])
    assert np.allclose(derived[:, 2], [10.0, -2.0])
    assert np.allclose(derived[:, 3], [12.0, -3.0])
    assert np.allclose(derived[:, 4], [20.0, 4.0])


def test_flag_feature_marks_constant_and_outliers() -> None:
    constant = flag_feature(np.ones(20))
    assert constant["constant"] is True
    values = np.r_[np.arange(20), 1000.0]
    flagged = flag_feature(values)
    assert flagged["outlier_count"] >= 1
    assert flagged["max_abs_robust_z"] > 8.0


def test_resolve_oof_path_prefers_explicit_and_configured(tmp_path) -> None:
    import os

    models = tmp_path / "models"
    old = models / "20260101_gdbt_notebook_cv_offline_old" / "oof_predictions.csv"
    new = models / "20260102_gdbt_notebook_cv_offline_new" / "oof_predictions.csv"
    explicit = tmp_path / "explicit.csv"
    for path in (old, new, explicit):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("date_id,raw_pred,allocation\n1,0.0,0.0\n")
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))

    assert resolve_oof_path(explicit, None, models) == explicit
    assert resolve_oof_path(None, new, models) == new
    assert resolve_oof_path(None, None, models) == new


def test_resolve_oof_path_reports_missing_oof(tmp_path) -> None:
    try:
        resolve_oof_path(None, None, tmp_path / "models")
    except FileNotFoundError as exc:
        assert "run_cv.py --offline" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
