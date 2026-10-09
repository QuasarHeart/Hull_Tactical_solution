import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field


# ============ MODEL CONFIGS ============
@dataclass
class ModelParameters:
    enet_alpha: float = 0.01
    enet_l1_ratio: float = 0.5
    xgb_n_estimators: int = 350
    xgb_max_depth: int = 10
    xgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 10
    lgb_learning_rate: float = 0.05
    ensemble_weights: dict = field(default_factory=lambda: {'enet': 0.45, 'xgb': 0.55, 'lgb': 0.45})
    vol_window: int = 20
    signal_multiplier_low_vol: float = 600.0  # For low V1
    signal_multiplier_high_vol: float = 400.0  # For high V1
    min_signal: float = 0.0
    max_signal: float = 2.0
    vol_scaling: float = 1.2
    retrain_freq: int = 1  # Retrain every row
    missing_threshold: float = 0.5
    max_train_rows: int = 800  # Reduced

# Initialize parameters
params = ModelParameters()


