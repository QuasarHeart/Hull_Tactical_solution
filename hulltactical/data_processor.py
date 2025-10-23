"""Custom data processor for feature engineering and transformation.
"""

import os
import joblib
import polars as pl

from hulltactical.config import PATH_DATA,PATH_MODELS
from hulltactical.utils import create_folder

class Dataprocessor:
    "Process data"

    PATH = os.path.join(PATH_MODELS / "data_processor")
    COLS_FEATURES_INIT = [f"D{i}" for i in range(1,10)]
    COLS_FEATURES_INIT += [f"E{i}"for i in range(1,21)]
    COLS_FEATURES_INIT += [f"I{i}"for i in range(1,10)]
    COLS_FEATURES_INIT += [f"M{i}"for i in range(1,19)]
    COLS_FEATURES_INIT += [f"P{i}"for i in range(1,14)]
    COLS_FEATURES_INIT += [f"S{i}"for i in range(1,13)]
    COLS_FEATURES_INIT += [f"V{i}"for i in range(1,14)]
    

    COLS_FEATURES_CORR = ["M1", "V13", "V10", "S5", "V7", "E19", "V9", "S12", "D2", "D1"]
                          

    T = 252
    
    def __init__(self):
        
