"""Configuration file."""

import os
from pathlib import Path
from dataclasses import dataclass, field
#判断代码是在哪个环境下运行
KAGGLE = 'KAGGLE_URL_BASE' in os.environ
LOCAL = not KAGGLE
#两种环境下的项目根目录 
base_paths = {
    "KAGGLE": Path("/kaggle/input"),
    "LOCAL" : Path("/home/yys/kaggle_projects/Hull_Tactical_solution")
}
#判断不同环境下的路径
if KAGGLE:
    base_path = base_paths["KAGGLE"]
    base_path_data = base_paths["KAGGLE"]
    PATH_DATA = base_path_data / "hull-tactical-market-prediction"
    PATH_MODELS = base_path / ""
    PATH_CODE = base_path / ""

elif LOCAL:
    base_path = base_paths["LOCAL"]
    base_path_data = base_paths["LOCAL"]
    PATH_DATA = base_path_data / "raw_data"
    PATH_MODELS = base_path / "hulltactical/models"
    PATH_CODE = base_path / ""

else:
    raise ValueError("Unknown Environment")

#训练集和测试集的路径
PATHS_DATA = {
    "train" : PATH_DATA / "train.csv",
}

WANDB_PROJECT = "Hull_Tactical"#项目名称

KAGGLE_USERNAME = "QuarsarHeart"#kaggle用户名，可以用来推送代码，具体怎么做以后再查

RANDOM_SEED = [42,250,88]#随机种子，之后可以尝试更多的或者是随机的种子

COL_TARGET = "market_forward_excess_returns"#目标变量
COL_DATE = "date_id"#要设置的indexname

BASE_FEATURES = ['E19', 'V7', 'V9', 'M1', 'V10', 'S8', 'V13', 'S2', 'P5', 'P6', 'P8', 'S3', 'M10', 'E2', 'P10', 'M4', 'P11', 'M11', 'S5', 'M13']
TO_LAGGED_FEATURES = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
LAGGED_FEATURES = [f"lagged_{col}" for col in TO_LAGGED_FEATURES]
FEATURES_TO_CREAT_WINDOWS = ['M1, M10, M11, M4, M13,V7, V9, V10, V13, S8, S2, S3, S5']
T = [5,22]
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
    signal_multiplier_low_vol: float = 600.0  # For low V1，这两个参数的确定还有待商榷
    signal_multiplier_high_vol: float = 400.0  # For high V1
    min_signal: float = 0.0
    max_signal: float = 2.0
    vol_scaling: float = 1.2
    retrain_freq: int = 1  # Retrain every row
    missing_threshold: float = 0.5
    max_train_rows: int = 800  # Reduced

# Initialize parameters
params = ModelParameters()