"""Schema and lightweight validation for the competition and live panels."""

from __future__ import annotations

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

FEATURE_PREFIXES = ("D", "E", "I", "M", "P", "S", "V")
N_FEATURES = 94

NON_FEATURE_COLUMNS = {
    COL_DATE,
    COL_DATE_ID,
    COL_TARGET,
    COL_TARGET_METRIC,
    COL_FWD,
    COL_RF,
    COL_IS_SCORED,
    "market_forward_excess_returns",
    "lagged_market_forward_excess_returns",
    "lagged_forward_returns",
    "lagged_risk_free_rate",
}


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Return the 94 anonymised feature columns in frame order."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def validate_competition_train(df: pl.DataFrame) -> None:
    required = {COL_DATE_ID, "market_forward_excess_returns", COL_FWD, COL_RF}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    if df.height == 0:
        raise ValueError("train.csv is empty")
    if df[COL_DATE_ID].n_unique() != df.height:
        raise ValueError("train.csv has duplicate date_id values")
    if not df[COL_DATE_ID].is_sorted():
        raise ValueError("train.csv is not sorted by date_id")
    features = feature_columns(df)
    if len(features) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} feature columns, found {len(features)}")


def validate_competition_test(df: pl.DataFrame) -> None:
    required = {COL_DATE_ID, COL_IS_SCORED}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test.csv is missing columns: {sorted(missing)}")
    features = feature_columns(df)
    if len(features) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} feature columns, found {len(features)}")


def validate_panel(df: pl.DataFrame, date_col: str = COL_DATE_ID) -> None:
    if date_col not in df.columns:
        raise ValueError(f"panel is missing date column '{date_col}'")
    if df[date_col].n_unique() != df.height:
        raise ValueError(f"panel has duplicate {date_col} values")
    if not df[date_col].is_sorted():
        raise ValueError(f"panel is not sorted by {date_col}")
    if COL_TARGET not in df.columns:
        raise ValueError("panel is missing the target column")
