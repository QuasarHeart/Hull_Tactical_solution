"""Shared, causal feature engineering for both model families.

The notebook-parity GBDT builder lives in `gdbt/features.py` and deliberately
reproduces the submitted notebook, including its quirks. This module is the
clean, model-agnostic feature layer that the DL sequence builder (and future
GBDT variants) should use:

- target-derived features are shifted by one day before rolling, so day t never
  sees the return from t to t+1;
- no backward fill, no full-sample statistics;
- calendar features are derived from the date column.
"""

from __future__ import annotations

import polars as pl

from .config import COL_DATE


def add_calendar_features(df: pl.DataFrame, date_col: str = COL_DATE) -> pl.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if df.schema[date_col] == pl.Utf8:
        df = df.with_columns(pl.col(date_col).str.to_date())
    return df.with_columns(
        pl.col(date_col).dt.weekday().cast(pl.Int8).alias("cal_weekday"),
        pl.col(date_col).dt.month().cast(pl.Int8).alias("cal_month"),
        pl.col(date_col).dt.quarter().cast(pl.Int8).alias("cal_quarter"),
        (pl.col(date_col).dt.day() >= 25).cast(pl.Int8).alias("cal_month_end"),
    )


def add_target_history_features(
    df: pl.DataFrame,
    target_col: str,
    windows: tuple[int, ...] = (5, 20, 60),
) -> pl.DataFrame:
    """Rolling stats of *past* realized targets, shifted by one day.

    At row t the returned features use targets through t-1 only. This is the
    feature family that drives volatility targeting and must never include the
    current target.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    expressions = []
    for window in windows:
        shifted = pl.col(target_col).shift(1)
        expressions.extend(
            [
                shifted.rolling_std(window).alias(f"target_vol_{window}"),
                shifted.rolling_mean(window).alias(f"target_mean_{window}"),
                shifted.rolling_min(window).alias(f"target_min_{window}"),
                shifted.rolling_max(window).alias(f"target_max_{window}"),
            ]
        )
    return df.with_columns(expressions)


def add_rolling_features(
    df: pl.DataFrame,
    columns: list[str],
    windows: tuple[int, ...] = (5, 20, 60),
) -> pl.DataFrame:
    """Causal rolling mean/std for observed features (no shift needed:
    the current feature value is known at the close of day t)."""
    expressions = []
    for column in columns:
        if column not in df.columns:
            continue
        for window in windows:
            expressions.extend(
                [
                    pl.col(column).rolling_mean(window).alias(f"{column}_mean_{window}"),
                    pl.col(column).rolling_std(window).alias(f"{column}_std_{window}"),
                ]
            )
    return df.with_columns(expressions) if expressions else df


def add_missing_indicators(
    df: pl.DataFrame,
    columns: list[str],
    prefix: str = "isna_",
) -> pl.DataFrame:
    expressions = [
        pl.col(column).is_null().cast(pl.Int8).alias(f"{prefix}{column}")
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(expressions) if expressions else df


def add_volatility_regime(
    df: pl.DataFrame,
    vol_column: str = "V1",
    window: int = 252,
) -> pl.DataFrame:
    """A causal percentile of the vol proxy over a trailing window.

    Uses a rolling rank (average rank of the current value within the trailing
    window), which is causal by construction.
    """
    if vol_column not in df.columns:
        return df
    return df.with_columns(
        (
            pl.col(vol_column).rolling_rank(window_size=window, method="average")
            / float(window)
        ).alias(f"{vol_column}_pct_{window}")
    )
