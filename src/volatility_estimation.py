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
from model import params


def estimate_volatility(test: pl.DataFrame, train: pl.DataFrame) -> float:
    vol = test['V1'][0] if 'V1' in test.columns else (train['target'].tail(params.vol_window).std() or 0.01)
    recent_returns = train['target'].tail(params.vol_window).to_numpy()
    if len(recent_returns) > 1:
        garch_vol = np.sqrt(0.3 * np.var(recent_returns) + 0.7 * vol**2)
        return max(garch_vol, 0.01)
    return max(vol, 0.01)