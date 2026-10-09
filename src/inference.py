import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

import kaggle_evaluation.default_inference_server
from predict_function import predict
from features import DATA_PATH

# ============ LAUNCH SERVER ============
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway((str(DATA_PATH),))