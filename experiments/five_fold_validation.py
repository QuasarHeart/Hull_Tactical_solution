import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# ============ 0. 初始化与环境配置 ============
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CV_OUTPUT_DIR = Path('./results')
CV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("正在启动量化机构级 5-Fold 零泄露时序验证与回测模拟...")
print(f"图像输出文件夹已锁定：{CV_OUTPUT_DIR.resolve()}")
print("="*60)

# 从原始 train 中提取未标准化的原始特征 (确保 scaler 严格在循环内 fit 达到零泄露)
X_raw = train.select(features).to_pandas().to_numpy()
y_raw = train['target'].to_pandas()

# 实例化 5 折时序分裂器
tscv = TimeSeriesSplit(n_splits=5)

# ============ 图 1. 绘制时序切分机制图 ============
print("\n[1/3] 正在生成时序切分机制可视化图...")
plt.figure(figsize=(12, 5))
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
    # 绘制训练集部分 (天蓝色)
    plt.barh(y=f"Fold {fold+1}", width=len(train_idx), left=0, 
             color='skyblue', edgecolor='black', alpha=0.9,
             label='Train' if fold == 0 else "")
    # 绘制验证集部分 (橙色)
    plt.barh(y=f"Fold {fold+1}", width=len(val_idx), left=len(train_idx), 
             color='orange', edgecolor='black', alpha=0.9,
             label='Validation (Out-of-Sample)' if fold == 0 else "")

plt.title("5-Fold TimeSeriesSplit (Walk-Forward) CV Strategy", fontsize=13, fontweight='bold')
plt.xlabel("Timeline Index (Trading Days)", fontsize=11)
plt.ylabel("Folds", fontsize=11)
plt.legend(loc='upper left', frameon=True)
plt.gca().invert_yaxis()  # 让第一折置于最上方
plt.tight_layout()
plt.savefig(CV_OUTPUT_DIR / '01_cv_splits_visualization.png', dpi=300)
plt.show()


# ============ 2. 执行零泄露 CV 并搜集样本外预测值 ============
# 搜集所有折的非重叠验证集，拼接成一条完整的样本外时间线（用于模拟回测）
oos_targets = []
oos_preds_enet = []
oos_preds_xgb = []
oos_preds_lgb = []
oos_preds_ensemble = []

# 记录各折的单模型方向准确率（用于生成折对比柱状图）
fold_accs = {m: [] for m in ['enet', 'xgb', 'lgb', 'ensemble']}

print("\n[2/3] 正在进行 5 折滚动拟合与样本外回测数据搜集...")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
    X_tr_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
    y_tr, y_val = y_raw.iloc[train_idx], y_raw.iloc[val_idx]
    
    # 严格在局部训练集上拟合 Scaler，并在验证集上应用，杜绝前瞻偏差
    fold_scaler = StandardScaler()
    X_tr = fold_scaler.fit_transform(X_tr_raw)
    X_val = fold_scaler.transform(X_val_raw)
    
    # ElasticNet 拟合
    fold_enet = ElasticNet(alpha=params.enet_alpha, l1_ratio=params.enet_l1_ratio, max_iter=1000000)
    fold_enet.fit(X_tr, y_tr)
    p_enet = fold_enet.predict(X_val)
    
    # XGBoost 拟合
    fold_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params.xgb_n_estimators,
        max_depth=params.xgb_max_depth,
        learning_rate=params.xgb_learning_rate,
        random_state=42
    )
    fold_xgb.fit(X_tr, y_tr)
    p_xgb = fold_xgb.predict(X_val)
    
    # LightGBM 拟合
    fold_lgb = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=params.lgb_n_estimators,
        max_depth=params.lgb_max_depth,
        learning_rate=params.lgb_learning_rate,
        random_state=42,
        verbosity=-1
    )
    fold_lgb.fit(X_tr, y_tr)
    p_lgb = fold_lgb.predict(X_val)
    
    # 加权集成
    p_ens = (params.ensemble_weights['enet'] * p_enet +
             params.ensemble_weights['xgb'] * p_xgb +
             params.ensemble_weights['lgb'] * p_lgb)
    
    # 方向准确率计算闭包
    def calc_dir_acc(y_true, y_pred):
        return np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # 搜集整体样本外回测序列
    oos_targets.extend(y_val.values)
    oos_preds_enet.extend(p_enet)
    oos_preds_xgb.extend(p_xgb)
    oos_preds_lgb.extend(p_lgb)
    oos_preds_ensemble.extend(p_ens)
    
    # 记录该折各模型表现
    fold_accs['enet'].append(calc_dir_acc(y_val, p_enet))
    fold_accs['xgb'].append(calc_dir_acc(y_val, p_xgb))
    fold_accs['lgb'].append(calc_dir_acc(y_val, p_lgb))
    fold_accs['ensemble'].append(calc_dir_acc(y_val, p_ens))
    
    print(f"  - Fold {fold+1} 完成。验证集样本量: {len(y_val)} 天。集成的方向准确率: {fold_accs['ensemble'][-1]:.2%}")


# ============ 图 2. 绘制分折性能对比柱状图 ============
print("\n[3/3] 正在绘制各折模型性能多维度对比图...")
fold_df = pd.DataFrame(fold_accs)
fold_df['Fold'] = [f"Fold {i+1}" for i in range(5)]
melted_fold_df = fold_df.melt(id_vars='Fold', var_name='Model', value_name='Directional Accuracy')

plt.figure(figsize=(12, 5.5))
sns.barplot(data=melted_fold_df, x='Fold', y='Directional Accuracy', hue='Model', palette='muted')
plt.axhline(0.50, color='red', linestyle='--', alpha=0.6, label='Random Guess (50%)')
plt.title("Out-of-Sample (OOS) Directional Accuracy per Fold", fontsize=13, fontweight='bold')
plt.ylabel("Directional Accuracy", fontsize=11)
plt.xlabel("Folds", fontsize=11)
plt.ylim(0.40, 0.60)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(CV_OUTPUT_DIR / '02_fold_metrics_comparison.png', dpi=300)
plt.show()


# ============ 图 3. 模拟样本外交易累计收益曲线 (Backtest) ============
print("\n[4/3] 正在模拟连续 30 年样本外 Long/Short 策略回测并绘制累计净值曲线...")
# 转化为 numpy 数组
oos_targets = np.array(oos_targets)
oos_preds_enet = np.array(oos_preds_enet)
oos_preds_xgb = np.array(oos_preds_xgb)
oos_preds_lgb = np.array(oos_preds_lgb)
oos_preds_ensemble = np.array(oos_preds_ensemble)

# 模拟策略收益率：
# 每日收益 = 标普500超额收益 * 预测符号 (sign)。当预测 > 0 做多，预测 < 0 做空。
ret_bh = oos_targets  # Benchmark: Buy & Hold
ret_enet = np.sign(oos_preds_enet) * oos_targets
ret_xgb = np.sign(oos_preds_xgb) * oos_targets
ret_lgb = np.sign(oos_preds_lgb) * oos_targets
ret_ens = np.sign(oos_preds_ensemble) * oos_targets

# 计算累计收益（净值曲线）
cum_bh = np.cumsum(ret_bh)
cum_enet = np.cumsum(ret_enet)
cum_xgb = np.cumsum(ret_xgb)
cum_lgb = np.cumsum(ret_lgb)
cum_ens = np.cumsum(ret_ens)

# 绘图
plt.figure(figsize=(14, 6.5))
timeline = np.arange(len(cum_bh))

plt.plot(timeline, cum_bh, label='Benchmark: Buy & Hold (S&P 500 Excess)', color='gray', alpha=0.5, linewidth=1.5)
plt.plot(timeline, cum_enet, label='Strategy: ElasticNet (ENET) - Linear', color='blue', linewidth=2)
plt.plot(timeline, cum_xgb, label='Strategy: XGBoost (XGB) - Non-linear', color='orange', alpha=0.6, linewidth=1.5)
plt.plot(timeline, cum_lgb, label='Strategy: LightGBM (LGB) - Non-linear', color='green', alpha=0.6, linewidth=1.5)
plt.plot(timeline, cum_ens, label='Strategy: Weighted Ensemble (Ours)', color='purple', linewidth=2.5)

plt.title("Simulated Out-of-Sample (OOS) Equity Curves (7,540 continuous trading days)", fontsize=13, fontweight='bold')
plt.xlabel("Sequential Out-of-Sample Trading Days", fontsize=11)
plt.ylabel("Cumulative Returns (Sum of Excess Returns)", fontsize=11)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(CV_OUTPUT_DIR / '03_simulated_equity_curve.png', dpi=300)
plt.show()

# 输出总体 CV 与回测指标
print("\n" + "="*60)
print("回测结果")
print("="*60)
for model_name, cum_ret, oos_p in [('Buy & Hold', cum_bh, None), 
                                   ('ElasticNet', cum_enet, oos_preds_enet), 
                                   ('XGBoost', cum_xgb, oos_preds_xgb), 
                                   ('LightGBM', cum_lgb, oos_preds_lgb), 
                                   ('Ensemble', cum_ens, oos_preds_ensemble)]:
    print(f"【{model_name}】")
    print(f"  - 30年累计超额收益: {cum_ret[-1]:.4f}")
    if oos_p is not None:
        acc = np.mean(np.sign(oos_targets) == np.sign(oos_p))
        print(f"  - 样本外总体方向准确率 (Directional Acc): {acc:.2%}")
    print("-"*40)

print("CV 与回测绘图流程全部顺利结束！三张诊断图已保存至 `./results/`。")
print("="*60)