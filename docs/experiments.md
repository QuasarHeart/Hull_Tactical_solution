# 基线、消融与预测置换实验

日期：2026-11-21  
脚本：`scripts/run_experiments.py --shuffles 1000`  
输入：已保存的 6 折 CV OOF

## 0. 全项目实验结果索引

本页既是 Phase 2 的基线/消融记录，也是全项目的实验索引。所有结果都区分
`in-sample replay`、六折 walk-forward、官方 live window 和 locked post-window；
不要把不同口径的分数直接横比。

| Phase | 问题 | 主要入口 | 记录 | 核心结论 |
| --- | --- | --- | --- | --- |
| replay / CV | notebook 能否复现？六折 OOF 是什么？ | `scripts/run_replay.py`, `scripts/run_cv.py` | `docs/repro_gdbt.md` | allocation 与 notebook 对齐 <1e-6；六折共 1,488 天 |
| Phase 2 | 模型是否优于 B&H、Constant、Vol-only？ | `scripts/run_experiments.py` | 本页 | 六折均值：B&H 1.498、Constant 1.25 1.411、Vol-only 1.267、Online 1.253、Daily 0.871、Offline 0.833 |
| Phase 2b | 时间 null、bootstrap、exposure matching | `scripts/run_validation.py` | `docs/gbdt_validation.md` | 没有检出可靠的增量；预测 IC 和时间置换都不显著 |
| Phase 3 | 严格部署口径能否复现 Online？E14 缺失怎么办？ | `scripts/run_daily_retrain.py`, `scripts/compare_deployment_modes.py`, `scripts/run_e14_stress.py` | `docs/deployment_eval.md` | Daily 0.871 < Online 1.253，也低于简单基准；E14 保持 median fill |
| Phase 4 | feature builder 漂移还是模型重训造成差距？ | `scripts/run_phase4_dynamic_fixed.py`, `scripts/run_feature_builder_attribution.py` | `docs/feature_builder_attribution.md` | 2×2 点估计指向 dynamic builder 负向、rolling 正向；所有 paired CI 跨 0 |
| Phase 5 | 官方 137 天窗口的收益来自哪里？ | `scripts/run_phase5_window_attribution.py`, `scripts/score_live_window.py` | `docs/phase5_competition_window_attribution.md`, `docs/live_window_20251216_20260706.md` | 官方公式复算 2.192；反转信号 0.055；multiplier=1 为 0.556；shuffle p≈0.052 |
| Phase 6 | 信号是否跨 regime、是否集中、能否在 locked window 复现？ | `scripts/run_phase6_signal_robustness.py`, `scripts/score_live_window.py` | `docs/phase6_signal_robustness.md`, `docs/phase6_preregistration.md` | 官方 2.192 但 HAC raw p=0.014、Holm p=0.085；locked 95 天 Daily 1.382、Offline 2.711，均不显著 |
| Phase 7 | 如何把证据、限制和复现材料整合成可审计报告？ | `scripts/make_phase7_assets.py`, `scripts/build_data_manifest.py` | `docs/final_research_report.md`, `data/manifest.json` | 冻结 GBDT，保留 evidence map；不把 holdout 结果用于选参 |
| Phase 8 | 单个特征与未来收益的时间序列 RankIC 是否跨折/regime 稳定？冗余和多重检验后还剩什么？ | `scripts/run_phase8_feature_rankic.py` | `docs/feature_rankic_analysis.md` | 99 个 model-input 特征中 5 个通过 BH；D1/D2 完全重复；regime 和交易成本仍有限制；不选特征 |

产物默认写入 `reports/<phase>/`，该目录被 `.gitignore` 忽略；需要长期保存的
结论、脚本、配置和 manifest 在 `docs/`、`scripts/`、`configs/` 与 `data/manifest.json`。

> 后续 GBDT 增量价值验证（temporal nulls、paired bootstrap、exposure matching、prediction
> quality、allocation attribution）见 `docs/gbdt_validation.md`；决策记录见
> `docs/research_decisions.md`。本页保留第一阶段基线/消融的结果。
>
> Phase 3 的 deployment-faithful daily retrain、统一比较和 E14 检查见
> `docs/deployment_eval.md`。Phase 4 的 feature-builder / retraining 2×2 归因见
> `docs/feature_builder_attribution.md`。Phase 5 的官方窗口归因见
> `docs/phase5_competition_window_attribution.md`；Phase 6 的 signal × regime、
> multiplier、concentration 和独立窗口检验见 `docs/phase6_signal_robustness.md`。

- offline：`models/20261121_170606_gdbt_notebook_cv_offline_ae2a75`
- online：`models/20261121_170851_gdbt_notebook_cv_online_0746df`

所有实验使用同一批测试折（2015/2017/2019/2021/2023/2025，共 1,488 个测试日）和
同一官方评分函数 `common.metrics.official_score`。完整记录在
`reports/experiments/`，图表在 `reports/experiments/plots/`。

## 1. 实验定义

| 实验 | 定义 |
|---|---|
| Buy & Hold | 每天仓位 = 1.0，即持有 S&P 500。 |
| Constant Allocation | 每天仓位 = 0.25/0.5/0.75/1.25/1.5/2.0；1.0 由 Buy & Hold 单独表示。 |
| Volatility-only | 不使用收益预测。目标仓位 = `scale * median(训练窗口因果 vol) / 当前 vol`，再做同样的 0.75/0.25 平滑和 3e-5 交易成本。`scale` 取 0.5/1.0/1.5，主结果用 1.0。 |
| Prediction Ablation | 去掉模型预测，把 `raw_pred` 替换成该折训练窗口的 target 均值（若为负则取 0），保留 600/400 乘数、波动率缩放、平滑和交易成本。 |
| Prediction Ablation (zero) | `raw_pred = 0`，仓位恒为 0，作为“只有平滑机制、没有信号”的现金对照。 |
| Prediction Shuffle | 在每个测试折内打乱 `raw_pred` 的时间顺序，保留当天的 `vol`、`multiplier`、平滑和交易成本，重复 1,000 次。 |

约定：全零仓位的官方 score 公式是 0/0，本文把现金记为 `score = sharpe = 0`。
Vol-only 的目标波动率来自训练窗口，不使用测试期信息。

## 2. 汇总结果

`mean score` 是 6 折官方 score 的算术平均；`continuous` 是把 6 折测试日拼接后算出的单条 OOF 曲线；`mean Sharpe` 是每折 raw Sharpe 的算术平均。

| 实验 | mean score | continuous score | mean raw Sharpe | mean alloc | turnover |
|---|---:|---:|---:|---:|---:|
| Buy & Hold 1.0 | **1.4975** | **1.2804** | 1.4975 | 1.000 | 0.0040 |
| Constant 0.25 | 0.4729 | 0.5205 | 1.5457 | 0.250 | 0.0010 |
| Constant 0.50 | 0.7410 | 0.7852 | 1.5296 | 0.500 | 0.0020 |
| Constant 0.75 | 1.1794 | 1.1144 | 1.5136 | 0.750 | 0.0030 |
| Constant 1.25 | 1.4109 | 1.2038 | 1.4815 | 1.250 | 0.0050 |
| Constant 1.50 | 1.1273 | 0.9596 | 1.4654 | 1.500 | 0.0061 |
| Constant 2.00 | 0.7965 | 0.6747 | 1.4333 | 2.000 | 0.0081 |
| Vol-only 0.5x | 1.0821 | 1.1897 | 1.5580 | 0.781 | 0.0648 |
| Vol-only 1.0x | 1.2665 | 1.1509 | 1.4857 | 1.200 | 0.0660 |
| Vol-only 1.5x | 0.9165 | 0.9757 | 1.4005 | 1.470 | 0.0462 |
| Model offline | 0.8332 | 0.4191 | 0.9922 | 0.495 | 0.3034 |
| Model online | 1.2530 | 1.2707 | 1.4354 | 0.652 | 0.5146 |
| Ablation train mean | 0.3906 | 0.3173 | 1.0322 | 0.336 | 0.0275 |
| Ablation zero / cash | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 |

注意：raw Sharpe 和官方 score 不是一回事。Constant 0.25 的 raw Sharpe 1.5457 高于 Buy & Hold 的 1.4975，但它的官方 score 只有 0.4729，因为低仓位在牛市里产生很大的 return penalty。不能只看 raw Sharpe。

## 3. 分折 score

| fold | Buy & Hold | Model offline | Model online | Vol-only 1.0x | Ablation train mean | Constant 1.25 |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 0.011 | 0.226 | 0.460 | 0.076 | 0.079 | -0.007 |
| 2017 | 2.784 | 0.316 | 0.521 | 2.942 | 0.740 | 2.643 |
| 2019 | 2.072 | 1.979 | 2.588 | 1.506 | 0.000* | 1.958 |
| 2021 | 2.105 | 2.360 | 1.205 | 1.917 | 0.486 | 1.989 |
| 2023 | 1.396 | 0.166 | 2.851 | 0.753 | 1.040 | 1.314 |
| 2025 | 0.618 | -0.049 | -0.107 | 0.405 | 0.000* | 0.569 |
| 均值 | **1.498** | 0.833 | 1.253 | 1.267 | 0.391 | 1.411 |

`*` 表示该折训练窗口的 target 均值为负，ablation 正确地选择空仓；现金按 0 计分。

## 4. 五个问题的直接回答

### 4.1 模型是否优于 Buy & Hold？

没有稳定优势。按 6 折均值，Buy & Hold 1.4975 > model online 1.2530 > model offline 0.8332；按连续 OOF，Buy & Hold 1.2804 与 model online 1.2707 基本打平，model offline 只有 0.4191。

但这不是“B&H 每年都赢”：

- 模型在 2015 明显更好（0.226/0.460 vs 0.011）；
- 2019/2021 模型与 B&H 接近，offline 在 2021 更高；
- 2017、2023、2025 B&H 大幅更好（尤其 2017 2.784 vs 0.316，2023 1.396 vs 0.166）。

原因是模型平均仓位只有 0.495/0.652，而 B&H 是 1.0；模型在强趋势年份被 600/400 乘数、波动率目标和 smoothing 压住了仓位，return penalty 很重。结论是：在这个评分口径下，模型没有证明自己优于简单持有；online 只是接近，不是稳定超过。

### 4.2 模型是否优于简单固定仓位？

也没有稳定优势。model online 1.2530 低于 Constant 1.25 的 1.4109 和 Buy & Hold 1.4975，只优于 0.25/0.5/2.0 等常数；model offline 0.8332 还低于 Constant 0.75/1.25/1.5。

固定 1.25 仓位的均值 1.4109，已经接近 B&H 的 1.4975；这说明“稍微加一点杠杆并一直持有”在这个样本里非常强。模型没有提供超过这种简单配置的稳定增量。

### 4.3 仅用波动率能否取得相似表现？

能，至少在这个 CV 上非常接近。Vol-only 1.0x 的均值 1.2665，与 model online 1.2530 基本相同；Vol-only 0.5x 的连续 score 1.1897，也高于 model offline 的 0.4191。Vol-only 1.0x 的连续 score 1.1509，低于 model online 的 1.2707，因此“完全不用收益预测”在单条曲线上略差，但差距远小于模型与 B&H 的差距。

这说明：当前模型的表现很大一部分来自波动率缩放/仓位路径，而不是收益预测的方向信息。Vol-only 是一个必须保留的强基线。

### 4.4 去掉模型预测、保留平滑和其他规则会怎样？

明显变差。Ablation train mean 的均值只有 0.3906，连续 score 0.3173；Model offline 0.8332 / online 1.2530。2019 和 2025 的训练窗口均值为负，ablation 空仓，score 记为 0；其余年份也普遍低于模型。

所以“保留平滑/波动率规则但去掉预测”不能复制模型表现，预测内容本身有贡献。但这个结论不能反过来证明预测已经足够好：它仍然没有超过 B&H，也没有稳定超过 Vol-only。

### 4.5 打乱预测时间顺序后，Sharpe 是否明显下降？

1,000 次折内置换的结果：

| mode | actual mean score | shuffle mean | p05 | p95 | p-value |
|---|---:|---:|---:|---:|---:|
| offline | 0.8332 | 0.5214 | 0.1409 | 0.9857 | **0.134** |
| online | 1.2530 | 0.7844 | 0.3365 | 1.2685 | **0.061** |

- offline 的实际均值高于置换均值，但置换分布很宽，p=0.134，不能在 5% 水平拒绝“时间顺序无关”。
- online 的 p=0.061，处于边缘；实际值接近 95 分位，但没有达到 5% 显著。
- 因此：有“预测时间顺序可能有用”的弱证据，但没有统计上稳健的强证据。结合 Vol-only 的相近表现，不能把当前收益归因于可靠的收益预测。

## 5. 仓位路径的含义

- Model offline 平均仓位 0.495，18.5% 的交易日空仓；Model online 平均 0.652，只有 1.0% 空仓。
- Buy & Hold 平均 1.0；Vol-only 1.0x 平均 1.20；Constant 1.25 固定 1.25；Ablation 平均 0.336。
- 模型在强趋势年份（2017/2023）仓位偏低，是输给 B&H 的主因；2017 offline 有 53.4% 的交易日空仓，2023 offline 有 42.0% 空仓。
- 因此下一步如果要提高竞赛 score，重点不是继续微调预测器，而是重新校准“平均暴露/波动率目标/return penalty”这一层；否则预测再好也可能被仓位规则压成低收益。

## 6. 记录与图表

- `reports/experiments/summary.csv`：汇总表
- `reports/experiments/fold_metrics.csv`：实验 × 折的全部指标
- `reports/experiments/shuffle_scores.csv`：1,000 × 6 × 2 条置换结果
- `reports/experiments/summary.json`、`experiment_manifest.json`：完整 JSON 记录
- `reports/experiments/README.md`：脚本生成的摘要
- `reports/experiments/plots/01_sharpe_by_fold.png`
- `reports/experiments/plots/02_score_summary.png`
- `reports/experiments/plots/03_constant_grid.png`
- `reports/experiments/plots/04_allocation_paths.png`
- `reports/experiments/plots/05_shuffle_distribution.png`
- `reports/experiments/plots/06_equity_curves.png`

![Mean score](../reports/experiments/plots/02_score_summary.png)

![Constant grid](../reports/experiments/plots/03_constant_grid.png)

![Allocation paths](../reports/experiments/plots/04_allocation_paths.png)

![Shuffle distribution](../reports/experiments/plots/05_shuffle_distribution.png)

![Equity curves](../reports/experiments/plots/06_equity_curves.png)

## 7. 注意事项

- 6 折是 2015/2017/2019/2021/2023/2025 的抽样年份，不是连续实盘；“每折均值”和“连续 OOF”会给出不同的排名，两个都要看。
- 官方 score 对收益不足有平方惩罚；raw Sharpe 可以通过低仓位变高，不能单独作为优化目标。
- Vol-only 的目标是训练窗口因果 vol 的中位数，scale 0.5/1.0/1.5 是固定的敏感性网格，没有用测试集选参。
- Ablation 的常数预测是训练窗口 target 均值；训练均值为负时空仓，现金按 0 分处理。
- Shuffle 只在每个折内打乱预测，保留实际 `vol`/`multiplier`；p-value 是单侧 `(1 + #shuffled >= actual) / (B + 1)`，没有做多重检验校正。
- 这些实验都复用已保存的 OOF，没有重新训练模型；部署版“每天重训最后 800 行”的精确 score 仍需单独的 daily-retrain CV。
