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

def predict(test: pl.DataFrame) -> float:
    global previous_lagged, train, enet_model, xgb_model, lgb_model, scaler, test_row_count, last_allocation, v1_median
    
    # Online learning: Update training data
    if previous_lagged is not None and 'lagged_market_forward_excess_returns' in previous_lagged.columns:
        append_row = previous_lagged.with_columns(
            pl.col('lagged_market_forward_excess_returns').alias('target')
        )
        append_row = create_features(append_row, is_train=False, median_values=median_values)
        if append_row.height > 0:
            append_row = append_row.with_columns(pl.lit(None).cast(pl.Float64).alias('target'))
            train = train.vstack(append_row.select(['date_id', 'target'] + features))
            if train.height > params.max_train_rows:
                train = train.tail(params.max_train_rows)
        
        # Retrain every `retrain_freq` rows
        if test_row_count % params.retrain_freq == 0:
            X_train = scaler.fit_transform(train.select(features).to_pandas())
            y_train = train['target'].to_pandas()
            if y_train.isna().any():
                raise ValueError("NaNs found in y_train during retraining")
            enet_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            lgb_model.fit(X_train, y_train)
    
    # Preprocess test data
    test = create_features(test, is_train=False, median_values=median_values)
    
    # Ensure no NaNs in test data
    X_test = test.select(features).to_pandas()
    if X_test.isna().any().any():
        raise ValueError(f"NaNs found in X_test for columns: {X_test.columns[X_test.isna().any()].tolist()}")
    
    X_test = scaler.transform(X_test)
    
    # Ensemble prediction
    enet_pred = enet_model.predict(X_test)[0]
    xgb_pred = xgb_model.predict(X_test)[0]
    lgb_pred = lgb_model.predict(X_test)[0]
    raw_pred = (params.ensemble_weights['enet'] * enet_pred +
                params.ensemble_weights['xgb'] * xgb_pred +
                params.ensemble_weights['lgb'] * lgb_pred)
    
    # Estimate volatility and dynamic signal multiplier
    vol = estimate_volatility(test, train)
    signal_multiplier = params.signal_multiplier_low_vol if ('V1' in test.columns and test['V1'][0] < v1_median) else params.signal_multiplier_high_vol
    
    # Convert to signal
    signal = np.clip(
        raw_pred * signal_multiplier,
        params.min_signal,
        params.max_signal
    )
    
    # Volatility-adjusted allocation
    allocation = min(params.max_signal, max(params.min_signal, signal / (vol * params.vol_scaling)))
    
    # Smooth allocation
    transaction_cost = 0.00003  # Reduced to 0.003%
    allocation = (0.75 * allocation + 0.25 * last_allocation) * (1 - transaction_cost)  # Adjusted smoothing
    last_allocation = allocation
    
    # Update state
    previous_lagged = test
    test_row_count += 1
    
    return float(allocation)