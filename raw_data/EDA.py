import xgboost as xgb
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance


# 1️⃣ 读取数据
dataset = pl.read_csv("train.csv")
df = dataset.to_pandas()  # XGBoost 接受 pandas DataFrame
df[df.columns] = df[df.columns].apply(pd.to_numeric,errors="coerce")
# 2️⃣ 划分特征和目标
X = df.drop(columns=['market_forward_excess_returns', 'date_id'])
y = df['market_forward_excess_returns']

# 3️⃣ 创建 XGBoost 回归器
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

# 5️⃣ 输出特征重要性
importance_dict = model.get_booster().get_score(importance_type='weight')  # 'weight', 'gain', 'cover'
importance = pd.Series(importance_dict).sort_values(ascending=False)

print("特征重要性排名：")
print(importance.to_dict())

