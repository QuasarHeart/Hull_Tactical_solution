import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
import time
from features import load_testset,create_features,load_trainset
from model import params

start_time = time.time()
train = load_trainset()
train = create_features(train, is_train=True)
features = [col for col in train.columns if col not in ['date_id', 'target', 'S1_lag1', 'P1_lag1', 'I9_lag1', 'V1_roll_mean_5', 'target_roll_std_5']]

# Cache median values for imputation (only for features used in model)
median_values = {col: train[col].median() if col in train.columns and train[col].is_null().mean() < 1.0 else 0.0 for col in features}

# Check for NaNs
X_train = train.select(features).to_pandas()
if X_train.isna().any().any():
    raise ValueError(f"NaNs found in X_train for columns: {X_train.columns[X_train.isna().any()].tolist()}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = train['target'].to_pandas()

# Train ElasticNet
enet_model = ElasticNet(alpha=params.enet_alpha, l1_ratio=params.enet_l1_ratio, max_iter=1000000)
enet_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=params.xgb_n_estimators,
    max_depth=params.xgb_max_depth,
    learning_rate=params.xgb_learning_rate,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=params.lgb_n_estimators,
    max_depth=params.lgb_max_depth,
    learning_rate=params.lgb_learning_rate,
    random_state=42
)
lgb_model.fit(X_train, y_train)

# Check startup time
if time.time() - start_time > 900:
    raise RuntimeError("Startup time exceeded 900 seconds")

# State for online learning
previous_lagged = None
test_row_count = 0
last_allocation = 0.0
v1_median = train['V1'].median() if 'V1' in train.columns else 0.0