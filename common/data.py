"""Data loading and panel construction.

Competition CSV -> canonical panel:
    date_id, target, target_metric, forward_returns, risk_free_rate, 94 features

Live CSV (daily-updated) will be added to this module later; the GBDT replay
only needs the competition files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from .config import (
    COL_DATE,
    COL_DATE_ID,
    COL_FWD,
    COL_IS_SCORED,
    COL_RF,
    COL_TARGET,
    COL_TARGET_METRIC,
)
from .schema import feature_columns, validate_competition_test, validate_competition_train


def load_competition_train(path: str | Path) -> pl.DataFrame:
    """Load train.csv, rename the label, and add the metric-aligned target."""
    df = pl.read_csv(path)
    validate_competition_train(df)
    df = df.with_columns(pl.exclude(COL_DATE_ID).cast(pl.Float64, strict=False))
    df = df.rename({"market_forward_excess_returns": COL_TARGET})
    df = df.with_columns((pl.col(COL_FWD) - pl.col(COL_RF)).alias(COL_TARGET_METRIC))
    return df.sort(COL_DATE_ID)


def load_competition_test(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    if df[COL_IS_SCORED].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col(COL_IS_SCORED).str.to_lowercase().eq("true").alias(COL_IS_SCORED)
        )
    validate_competition_test(df)
    df = df.with_columns(
        pl.exclude([COL_DATE_ID, COL_IS_SCORED]).cast(pl.Float64, strict=False)
    )
    return df.sort(COL_DATE_ID)


def load_panel(train_path: str | Path, test_path: str | Path | None = None) -> pl.DataFrame:
    """Build the canonical competition panel.

    The local test rows are historical copies of train rows, so the panel is
    just the train frame. `test_path` is accepted for symmetry and future live
    appends; the returned panel always contains the full history.
    """
    panel = load_competition_train(train_path)
    if test_path is not None:
        # Validate the test file early; its rows are already inside the panel.
        load_competition_test(test_path)
    return panel


def load_date_map(path: str | Path) -> pl.DataFrame:
    """Load a date_id -> date mapping (generated from the daily Hull CSV)."""
    df = pl.read_csv(path, try_parse_dates=True)
    if COL_DATE_ID not in df.columns or COL_DATE not in df.columns:
        raise ValueError("date map must contain date_id and date columns")
    if df[COL_DATE].dtype == pl.Utf8:
        df = df.with_columns(pl.col(COL_DATE).str.to_date())
    return df.with_columns(pl.col(COL_DATE_ID).cast(pl.Int64)).sort(COL_DATE_ID)


def date_ids_for_years(date_map: pl.DataFrame, years: Iterable[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for year in years:
        subset = date_map.filter(pl.col(COL_DATE).dt.year() == int(year))
        out[int(year)] = subset[COL_DATE_ID].to_list()
    return out


def date_ids_between(
    date_map: pl.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> list[int]:
    df = date_map
    if start is not None:
        df = df.filter(pl.col(COL_DATE) >= pl.lit(start).str.to_date())
    if end is not None:
        df = df.filter(pl.col(COL_DATE) <= pl.lit(end).str.to_date())
    return df[COL_DATE_ID].to_list()


def slice_dates(
    panel: pl.DataFrame,
    dates: Sequence,
    date_col: str = COL_DATE_ID,
) -> pl.DataFrame:
    return panel.filter(pl.col(date_col).is_in(list(dates)))


def attach_labels(
    panel: pl.DataFrame,
    predictions: pl.DataFrame,
    date_col: str = COL_DATE_ID,
) -> pl.DataFrame:
    """Join predictions to realized labels for scoring."""
    labels = panel.select(
        [date_col, COL_TARGET, COL_TARGET_METRIC, COL_FWD, COL_RF]
    )
    return predictions.join(labels, on=date_col, how="left")


def feature_frame(panel: pl.DataFrame) -> pl.DataFrame:
    """Select the 94 raw feature columns."""
    return panel.select(feature_columns(panel))
