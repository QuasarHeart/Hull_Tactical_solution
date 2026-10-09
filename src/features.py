import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
from model import params
#本地调试用，解决模块找不到的问题
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

# ============ PATHS ============
DATA_PATH = Path('../data')

# ============ DATA LOADING AND PREPROCESSING ============
def load_trainset() -> pl.DataFrame:
    df = (
        pl.read_csv(DATA_PATH / "train.csv")
        .rename({'market_forward_excess_returns': 'target'})
        .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
        .filter(pl.col('date_id') >= 37)
        .tail(params.max_train_rows)
    )
    missing_counts = {col: df[col].is_null().mean() for col in df.columns}
    feature_cols = [
        col for col, miss_rate in missing_counts.items() 
        if miss_rate <= params.missing_threshold and col not in ['date_id', 'target']
    ]
    keep_cols = ['date_id', 'target'] + feature_cols
    if len(keep_cols) != len(set(keep_cols)):
        raise ValueError(f"Duplicate columns in keep_cols: {keep_cols}")
    return df.select(keep_cols)

def load_testset() -> pl.DataFrame:
    df = (
        pl.read_csv(DATA_PATH / "test.csv")
        .with_columns(pl.exclude('date_id', 'is_scored').cast(pl.Float64, strict=False))
    )
    train_cols = load_trainset().columns
    feature_cols = [col for col in train_cols if col not in ['date_id', 'target']]
    return df.select(['date_id', 'is_scored', 'lagged_market_forward_excess_returns'] + feature_cols)

def create_features(df: pl.DataFrame, is_train: bool = False, median_values: dict = None) -> pl.DataFrame:
    feature_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V']
    base_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]
    
    # Derived features
    required_cols = ['I1', 'I2', 'I7', 'I9', 'M11']
    if all(col in base_features for col in required_cols):
        df = df.with_columns(
            (pl.col("I2") - pl.col("I1")).alias("U1"),
            (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3)).alias("U2"),
            (pl.col("I2").rolling_mean(window_size=66).alias("I2_MA66"))
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("U1"),
            pl.lit(0.0).alias("U2")
        )
    
    # Interaction features
    if 'V1' in base_features and 'S1' in base_features:
        df = df.with_columns((pl.col("V1") * pl.col("S1")).alias("V1_S1"))
    if 'M11' in base_features and 'V1' in base_features:
        df = df.with_columns((pl.col("M11") * pl.col("V1")).alias("M11_V1"))
    if 'I9' in base_features and 'S1' in base_features:
        df = df.with_columns((pl.col("I9") * pl.col("S1")).alias("I9_S1"))
    
    # Training-only features
    if is_train:
        if 'S1' in base_features:
            df = df.with_columns(pl.col("S1").shift(1).alias("S1_lag1"))
        if 'P1' in base_features:
            df = df.with_columns(pl.col("P1").shift(1).alias("P1_lag1"))
        if 'I9' in base_features:
            df = df.with_columns(pl.col("I9").shift(1).alias("I9_lag1"))
        if 'V1' in base_features:
            df = df.with_columns(pl.col("V1").rolling_mean(window_size=5).alias("V1_roll_mean_5"))
        if 'target' in df.columns:
            df = df.with_columns(pl.col("target").rolling_std(window_size=5).alias("target_roll_std_5"))
    
    # Impute missing values
    for col in base_features:
        if col.startswith('I'):
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).forward_fill()).fill_null(pl.col(col).backward_fill()))
        median = median_values.get(col, df[col].median()) if median_values else df[col].median()
        df = df.with_columns(pl.col(col).fill_null(median if median is not None else 0.0))
    
    # Impute derived and additional features
    derived_features = ["U1", "U2", "V1_S1", "M11_V1", "I9_S1"]
    additional_features = ["S1_lag1", "P1_lag1", "I9_lag1", "V1_roll_mean_5", "target_roll_std_5"] if is_train else []
    for col in derived_features + additional_features:
        if col in df.columns:
            median = median_values.get(col, df[col].median()) if median_values else df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median if median is not None else 0.0))
    
    # Feature list (exclude training-only features)
    features = base_features + derived_features
    select_cols = ["date_id"] + features + (["target"] if is_train else [])
    return df.select(select_cols)