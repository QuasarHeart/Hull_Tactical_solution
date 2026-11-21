from __future__ import annotations

import csv
import random
from datetime import date, timedelta

import polars as pl
import pytest

from common.config import (
    COL_DATE,
    COL_DATE_ID,
    COL_TARGET,
    COL_TARGET_METRIC,
    TARGET_OFFSET,
    Config,
)
from common.ingest import (
    IngestResult,
    expected_feature_columns,
    load_panel_live,
    load_state,
    read_daily_csv,
    save_state,
    upsert_panel,
    upsert_prediction_history,
    validate_daily,
    write_health,
)

FEATURES = expected_feature_columns(Config.load("configs/base.yaml").paths.train_csv)


def _write_daily(
    path,
    n: int = 1200,
    null_tail: int = 3,
    middle_null: bool = False,
) -> None:
    rng = random.Random(7)
    header = [""] + ["target"] + FEATURES
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(n):
            day = date(2020, 1, 1) + timedelta(days=i)
            if i >= n - null_tail or (middle_null and i == n // 2):
                target = "NA"
            else:
                target = f"{rng.gauss(0, 0.005):.8f}"
            features = [f"{rng.gauss(0, 1):.6f}" for _ in FEATURES]
            writer.writerow([day.isoformat(), target, *features])


@pytest.fixture()
def daily_csv(tmp_path):
    path = tmp_path / "20261120_data.csv"
    _write_daily(path)
    return path


def test_read_daily_csv_and_validate(daily_csv):
    df = read_daily_csv(daily_csv, FEATURES)
    assert df.height == 1200
    assert df.width == 100
    assert df[COL_DATE_ID].to_list() == list(range(1200))
    assert df[COL_TARGET].null_count() == 3
    assert df[COL_TARGET_METRIC].null_count() == 3
    first = df.row(0, named=True)
    assert first[COL_TARGET_METRIC] == pytest.approx(first[COL_TARGET] + TARGET_OFFSET)

    warnings = validate_daily(
        df, FEATURES, require_alignment=False, min_rows=100, min_targets=100
    )
    assert warnings == []


def test_validate_rejects_middle_null(tmp_path):
    path = tmp_path / "bad.csv"
    _write_daily(path, middle_null=True)
    df = read_daily_csv(path, FEATURES)
    with pytest.raises(ValueError, match="not confined"):
        validate_daily(
            df, FEATURES, require_alignment=False, min_rows=100, min_targets=100
        )


def test_validate_rejects_feature_mismatch(daily_csv):
    df = read_daily_csv(daily_csv, FEATURES)
    broken = df.rename({"D1": "D1_broken"})
    with pytest.raises(ValueError, match="do not match"):
        validate_daily(
            broken, FEATURES, require_alignment=False, min_rows=100, min_targets=100
        )


def test_upsert_panel_is_idempotent(daily_csv, tmp_path):
    df = read_daily_csv(daily_csv, FEATURES)
    panel_path = tmp_path / "panel.parquet"
    upsert_panel(panel_path, df)
    upsert_panel(panel_path, df)
    loaded = load_panel_live(panel_path)
    assert loaded.height == 1200
    assert loaded[COL_DATE].is_sorted()
    assert loaded[COL_DATE_ID].to_list() == list(range(1200))


def test_upsert_refuses_to_shrink(daily_csv, tmp_path):
    df = read_daily_csv(daily_csv, FEATURES)
    panel_path = tmp_path / "panel.parquet"
    upsert_panel(panel_path, df)
    with pytest.raises(ValueError, match="shrink"):
        upsert_panel(panel_path, df.head(1000))


def test_state_and_health(daily_csv, tmp_path):
    df = read_daily_csv(daily_csv, FEATURES)
    state_path = tmp_path / "state.json"
    state = {
        "last_data_date": "2026-11-20",
        "last_ingested_at": "2026-11-21T00:00:00+00:00",
        "warnings": ["E14 missing"],
    }
    save_state(state_path, state)
    assert load_state(state_path)["last_data_date"] == "2026-11-20"

    health = write_health(tmp_path / "health.json", df, state)
    assert health["rows"] == 1200
    assert health["target_rows"] == 1197
    assert health["pending_targets"] == 3
    assert health["warnings"] == ["E14 missing"]


def test_ingest_result_dict():
    result = IngestResult(
        status="checked",
        data_date=date(2026, 11, 20),
        url="https://example.invalid/data.csv",
        warnings=["E14 missing"],
    )
    payload = result.as_dict()
    assert payload["status"] == "checked"
    assert payload["data_date"] == "2026-11-20"
    assert payload["warnings"] == ["E14 missing"]


def test_prediction_history_is_idempotent(tmp_path):
    path = tmp_path / "predictions.parquet"
    base = {
        "date_id": 1,
        "raw_pred": 0.1,
        "vol": 0.2,
        "v1": 0.3,
        "multiplier": 600.0,
        "train_start": 1,
        "train_end": 800,
        "n_train": 800,
        "n_features": 99,
        "initial_allocation": 0.0,
        "created_at": "2026-11-19T00:00:00+00:00",
    }
    upsert_prediction_history(
        path, {**base, "data_date": "2026-11-19", "allocation": 1.0}
    )
    upsert_prediction_history(
        path, {**base, "data_date": "2026-11-20", "allocation": 1.5}
    )
    upsert_prediction_history(
        path, {**base, "data_date": "2026-11-20", "allocation": 1.6}
    )
    frame = pl.read_parquet(path).sort("data_date")
    assert frame.height == 2
    assert frame["allocation"].to_list() == [1.0, 1.6]
