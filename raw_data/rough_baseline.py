#import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
#import kaggle_evaluation.default_inference_server
import time
from Hull_Tactical_solution.hulltactical.metrics import score
import pandas as pd
import collections
# ============ PATHS ============
DATA_PATH = Path('/home/yys/kaggle_projects/Hull_Tactical_solution/raw_data')

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

# ============ DATA LOADING AND PREPROCESSING ============
def load_trainset() -> pl.DataFrame:
    """
    读取 train.csv，并选择缺失率低的列作为特征。
    注意：这里明确排除 forward_returns 和 risk_free_rate（以避免泄露）
    """
    df = (
        pl.read_csv(DATA_PATH / "train.csv")
        .rename({'market_forward_excess_returns': 'target'})
        .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
        .filter(pl.col('date_id') >= 37)
        .tail(params.max_train_rows)
    )

    # 强制列名字符串（防护）
    df.columns = [str(c) for c in df.columns]

    # 计算缺失率（polars 版本）
    n = df.height
    missing_counts = {col: df[col].is_null().sum() / n for col in df.columns}

    # 排除可能泄露未来的信息列
    banned = {'date_id', 'target', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'}
    feature_cols = [
        col for col, miss_rate in missing_counts.items()
        if (miss_rate <= params.missing_threshold) and (col not in banned)
    ]

    keep_cols = ['date_id', 'target'] + feature_cols
    if len(keep_cols) != len(set(keep_cols)):
        raise ValueError(f"Duplicate columns in keep_cols: {keep_cols}")

    return df.select(keep_cols)
def load_testset() -> pl.DataFrame:#加载测试集
    df = (
        pl.read_csv(DATA_PATH / "test.csv")#读取
        .with_columns(pl.exclude('date_id', 'is_scored').cast(pl.Float64, strict=False))#把特定列转换成浮点型
    )
    train_cols = load_trainset().columns#直接移植训练集加载加载函数返回的数据框的列名
    feature_cols = [col for col in train_cols if col not in ['date_id', 'target']]#提取特征列
    return df.select(['date_id', 'is_scored', 'lagged_market_forward_excess_returns'] + feature_cols)#返回含有特定列名的训练集

def create_features(df_in, is_train: bool = False, median_values: dict = None) -> pl.DataFrame:
    """
    Robust feature creation:
    - Accepts polars.DataFrame or pandas.DataFrame
    - Force column names to str
    - Safe prefix matching
    - Create derived / lag / rolling features (training-only features only created when is_train=True)
    - Fill missing values using median_values if provided
    """
    # Accept pandas or polars
    if isinstance(df_in, pd.DataFrame):
        df = pl.from_pandas(df_in.reset_index(drop=True))
    else:
        df = df_in

    # Enforce column names as strings
    df.columns = [str(c) for c in df.columns]

    # safe prefixes
    feature_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V']
    feature_prefixes = [str(p) for p in feature_prefixes]

    # safe base_features detection
    base_features = [
        col for col in df.columns
        if isinstance(col, str) and any(col.startswith(prefix) for prefix in feature_prefixes)
    ]

    # Derived features
    required_cols = ['I1', 'I2', 'I7', 'I9', 'M11']
    if all(col in base_features for col in required_cols):
        df = df.with_columns(
            (pl.col("I2") - pl.col("I1")).alias("U1"),
            (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3 + 1e-12)).alias("U2"),
            pl.col("I2").rolling_mean(window_size=66).alias("I2_MA66")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("U1"), pl.lit(0.0).alias("U2"))

    # Interaction features
    if 'V1' in base_features and 'S1' in base_features:
        df = df.with_columns((pl.col("V1") * pl.col("S1")).alias("V1_S1"))
    if 'M11' in base_features and 'V1' in base_features:
        df = df.with_columns((pl.col("M11") * pl.col("V1")).alias("M11_V1"))
    if 'I9' in base_features and 'S1' in base_features:
        df = df.with_columns((pl.col("I9") * pl.col("S1")).alias("I9_S1"))

    # Training-only features (lag / rolling) - only when creating train data
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

    # Impute missing values for I* with forward/backward fill, others with median (or provided median_values)
    for col in base_features:
        if col.startswith('I'):
            # forward then backward fill for interest features
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).forward_fill()).fill_null(pl.col(col).backward_fill()))
        # median fallback
        med = None
        if median_values and col in median_values:
            med = median_values[col]
        else:
            try:
                med = float(df[col].median())
            except Exception:
                med = 0.0
        df = df.with_columns(pl.col(col).fill_null(med if med is not None else 0.0))

    # Fill derived and additional features' NaNs
    derived_features = ["U1", "U2", "V1_S1", "M11_V1", "I9_S1"]
    additional_features = ["S1_lag1", "P1_lag1", "I9_lag1", "V1_roll_mean_5", "target_roll_std_5"] if is_train else []
    for col in derived_features + additional_features:
        if col in df.columns:
            med = median_values.get(col) if (median_values and col in median_values) else None
            if med is None:
                try:
                    med = float(df[col].median())
                except Exception:
                    med = 0.0
            df = df.with_columns(pl.col(col).fill_null(med if med is not None else 0.0))

    # Final feature list
    features = base_features + [c for c in derived_features if c in df.columns]
    count = collections.Counter(features)
    print([col for col,fre in count.items() if fre>1])
    select_cols = ["date_id"] + features + (["target"] if is_train and "target" in df.columns else [])
    # ensure columns exist
    select_cols = [c for c in select_cols if c in df.columns]
    count = collections.Counter(select_cols)
    print([col for col,fre in count.items() if fre>1])
    print(is_train)
    return df.select(select_cols)


# ============ MODEL TRAINING ============
start_time = time.time()#计算代码执行时间
train = load_trainset()#加载训练集
train = create_features(train, is_train=True)#创建训练集的特征
features = [col for col in train.columns if col not in ['date_id', 'target', 'S1_lag1', 'P1_lag1', 'I9_lag1', 'V1_roll_mean_5', 'target_roll_std_5']]#特征过滤


# Cache median values for imputation (only for features used in model)
median_values = {col: train[col].median() if col in train.columns and train[col].is_null().mean() < 1.0 else 0.0 for col in features}#如果这一列不全是空值，就用中位数填充，否则用0.0填充

# Check for NaNs
X_train = train.select(features).to_pandas()
if X_train.isna().any().any():
    raise ValueError(f"NaNs found in X_train for columns: {X_train.columns[X_train.isna().any()].tolist()}")

scaler = StandardScaler()#把训练数据的特征标准化为均值0标准差1的形式，并取出目标变量用于训练模型
X_train = scaler.fit_transform(X_train)
y_train = train['target'].to_pandas()

# Train ElasticNet，这里是使用了starter notebook上的模型
enet_model = ElasticNet(alpha=params.enet_alpha, l1_ratio=params.enet_l1_ratio, max_iter=1000000)
enet_model.fit(X_train, y_train)
#这里我想要使用catBoost试一下
#train catBoost

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

# Check startup time,确保时间不会超时
if time.time() - start_time > 900:
    raise RuntimeError("Startup time exceeded 900 seconds")

# State for online learning，在线学习
previous_lagged = None#记录上一次的滚动平均
test_row_count = 0#记录现在到了第几行
last_allocation = 0.0#记录上一次分配的仓位
v1_median = train['V1'].median() if 'V1' in train.columns else 0.0

# ============ VOLATILITY ESTIMATION ============
def estimate_volatility(test: pl.DataFrame, train: pl.DataFrame) -> float:
    try:
        if isinstance(test, pl.DataFrame):
            if 'V1' in test.columns and test.height > 0:
                v = float(test.select(pl.col('V1')).to_pandas().iloc[0,0])
                return max(v, 0.01)
        else:
            # fallback: use train recent std
            recent_returns = train['target'].tail(params.vol_window).to_numpy() if 'target' in train.columns else np.array([])
            if len(recent_returns) > 1:
                vol = np.std(recent_returns, ddof=1)
                garch_vol = np.sqrt(0.3 * np.var(recent_returns) + 0.7 * (vol**2))
                return max(garch_vol, 0.01)
    except Exception:
        pass
    # ultimate fallback
    return 0.01
"""这下面是AI写的一个可能可以替换上面计算波动率的更好的函数
import numpy as np
from arch import arch_model

def estimate_volatility(test: pl.DataFrame, train: pl.DataFrame) -> float:
    
    使用 GARCH(1,1) 估计当前时刻的波动率，返回值与原函数保持一致。
    逻辑为：
    1. 如果 test 数据中含有 'V1'（波动代理），则直接使用 test['V1'][0]。
    2. 否则从 train 中提取 target（收益率），拟合 GARCH(1,1)，预测下一期波动率。
    3. 若 GARCH 失败或数据太少，则返回一个安全值 0.01。
    
    # ✅ 1. 如果有测试集直接提供的波动代理（比如成交量、波动指标），优先使用
    if 'V1' in test.columns:
        try:
            vol = float(test['V1'][0])
            return max(vol, 0.01)
        except:
            pass  # 继续走 GARCH

    # ✅ 2. 从 train 中提取收益率
    if 'target' not in train.columns:
        return 0.01
    
    returns = train['target'].to_numpy()

    # 取最近的 window 数据（比如 150 或 params.vol_window）避免过拟合
    window_size = min(len(returns), 150)
    recent_returns = returns[-window_size:]

    # 如果有效数据太少，返回兜底值
    if len(recent_returns) < 20:
        return 0.01

    try:
        # ✅ 3. 用 GARCH 模型拟合波动率（学术界标准方法）
        model = arch_model(recent_returns, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')  # 关闭训练日志

        # 预测下一期的波动率，这里返回的是方差的平方根 → σ_t
        forecast = res.forecast(horizon=1)
        sigma = np.sqrt(forecast.variance.values[-1, 0])

        return float(max(sigma, 0.01))

    except Exception:  # 如果 GARCH 拟合失败
        return 0.01

"""
# ============ PREDICTION FUNCTION ============
def predict(test_df_input) -> pd.DataFrame:
    """
    Accepts either polars.DataFrame or pandas.DataFrame containing multiple rows (e.g. last 180 rows).
    Simulates online inference: for each row, create features, predict, compute allocation, smooth and update state.
    Returns a pandas.DataFrame with 'prediction' column with same length / index as input.
    """
    global previous_lagged, train, enet_model, xgb_model, lgb_model, scaler, test_row_count, last_allocation, v1_median, median_values

    # normalize input to pandas for easy row iteration, but keep polars for feature ops
    if isinstance(test_df_input, pl.DataFrame):
        test_pd = test_df_input.to_pandas()
    else:
        test_pd = test_df_input.copy()

    # ensure date_id exists for indexing
    if 'date_id' not in test_pd.columns:
        test_pd = test_pd.reset_index().rename(columns={'index':'date_id'})

    preds = []
    # iterate rows to simulate online per-row inference
    for i in range(len(test_pd)):
        row_pd = test_pd.iloc[[i]].reset_index(drop=True)  # shape (1, ncols)
        # convert to polars for feature creation
        row_pl = pl.from_pandas(row_pd)
        # create features for single row (is_train=False)
        row_feat = create_features(row_pl, is_train=False, median_values=median_values)

        # ensure features exist; if some train features are missing, add zeros
        # features is global list (from training)
        feat_list = [f for f in features if f in row_feat.columns]
        # if some features missing, create them as zeros
        missing = [f for f in features if f not in row_feat.columns]
        if missing:
            for m in missing:
                row_feat = row_feat.with_columns(pl.lit(0.0).alias(m))

        # order columns like training features
        row_X = row_feat.select(features).to_pandas()
        # fill NA with median_values fallback
        row_X = row_X.fillna({c: median_values.get(c, 0.0) for c in row_X.columns})

        # scale with scaler (scaler was fit on training)
        X_scaled = scaler.transform(row_X)

        # model predictions (scikit-learn predict returns ndarray)
        enet_pred = enet_model.predict(X_scaled)[0]
        xgb_pred = xgb_model.predict(X_scaled)[0]
        lgb_pred = lgb_model.predict(X_scaled)[0]
        raw_pred = (params.ensemble_weights['enet'] * enet_pred +
                    params.ensemble_weights['xgb'] * xgb_pred +
                    params.ensemble_weights['lgb'] * lgb_pred)

        # vol estimate (use row V1 if present, else fallback)
        row_pl_for_vol = row_feat
        vol = estimate_volatility(row_pl_for_vol, train)

        # choose multiplier per-row (use V1 if available)
        try:
            v1_val = float(row_feat.select(pl.col('V1')).to_pandas().iloc[0,0]) if 'V1' in row_feat.columns else v1_median
        except Exception:
            v1_val = v1_median
        signal_multiplier = params.signal_multiplier_low_vol if v1_val < v1_median else params.signal_multiplier_high_vol

        # raw signal -> clip within min/max
        signal = np.clip(raw_pred * signal_multiplier, params.min_signal, params.max_signal)

        # volatility adjusted allocation
        allocation = min(params.max_signal, max(params.min_signal, signal / (vol * params.vol_scaling)))

        # smoothing with previous allocation
        allocation = (0.75 * allocation + 0.25 * last_allocation) * (1 - 0.00003)  # transaction cost factor
        last_allocation = allocation

        # update train state with lagged info: here we set previous_lagged for next step to mimic online update
        # create previous_lagged as polars row with lagged_market_forward_excess_returns field if available in row_pd
        previous_lagged = pl.from_pandas(row_pd)

        test_row_count += 1
        preds.append(allocation)

    # return as pandas DataFrame aligned with input index
    out_df = pd.DataFrame({'prediction': preds}, index=test_pd.index)
    return out_df

test = load_trainset()
test = create_features(test,is_train=True)
test_pd = test.to_pandas().iloc[-180:]

submission = predict(test_pd)


print(score(test_pd,submission))



# ============ LAUNCH SERVER ============
#inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

#if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#    inference_server.serve()
#else:
#    inference_server.run_local_gateway((str(DATA_PATH),))