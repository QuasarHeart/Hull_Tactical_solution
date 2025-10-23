"""Configuration file."""

import os
from pathlib import Path
import numpy
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
    "test" : PATH_DATA / "test.csv"
}

WANDB_PROJECT = "Hull_Tactical"#项目名称

KAGGLE_USERNAME = "QuarsarHeart"#kaggle用户名，可以用来推送代码，具体怎么做以后再查

RANDOM_SEED = 42#随机种子，之后可以尝试更多的或者是随机的种子

COL_TARGET = "market_forward_excess_returns"#目标变量
COL_DATE = "date_id"#要设置的indexname

SCALING_FACTORS_TO_TEST = [round(i, 3) for i in np.arange(0.0, 1.0, 0.01)]

