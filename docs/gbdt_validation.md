# GBDT Validation Report — Before Deep Learning

日期：2026-11-21  
输入：已保存的 6 折 OOF（不重新训练、不修改正式预测）  
代码：`scripts/run_validation.py`、`common/validation.py`  
配置：`configs/validation.yaml`  
结果：`reports/validation/`

## 0. 结论摘要

**当前不建议进入 Deep Learning 实现阶段。**

原因不是“GBDT 一定没有价值”，而是本阶段没有检出可靠的增量价值：

1. Temporal nulls 没有在多重检验后显著；Online 的 block permutation 原始 p 最低为 0.025
   （L=40），Holm 后为 0.125；circular shift p=0.050。Offline 更弱。
2. Paired block bootstrap 下，Online 与 Buy & Hold、Constant 1.25、Vol-only 的平均差异
   全部跨 0；只有 2023 折出现显著正差，2017 折出现方向相反的边缘负差。
3. 与严格 exposure-matched 的 `lag1` 相比，Online 的平均 score 高 0.690，但 95% CI
   `[-0.160, 1.218]`，p=0.111；与 `expanding_mean` 的差异 p=0.582。不能确认增量价值。
4. Online 的 rank IC 为 0.0116，95% CI `[-0.034, 0.064]`，p=0.526；quantile 收益不单调。
   Offline 有弱的正 IC（0.0637，CI `[0.024, 0.109]`，p=0.002），但这不是当前 Online 模型。
5. 归因显示 600/400 multiplier 是主导杠杆；smoothing 小幅为正；vol scaling 的符号在
   Online/Offline 之间翻转；Vol-only 1.0x 的 mean score 1.2665 略高于 Online 的 1.2530。

因此：先把仓位/暴露层和评估协议做对，再决定是否值得用 DL。否则 DL 很可能是在同一个
弱信号/错误暴露层上增加复杂度。

## 1. 实验设置与防泄漏

- 6 折：2015、2017、2019、2021、2023、2025；共 1,488 个测试日。
- Offline OOF：`models/20261121_170606_gdbt_notebook_cv_offline_ae2a75`
- Online OOF：`models/20261121_170851_gdbt_notebook_cv_online_0746df`
- 主 block length：20（约一个交易月）；敏感性：5/10/20/40/60。
- Temporal nulls：各 1,000 次。
- Paired moving-block bootstrap：2,000 次；fold-stratified。
- Exposure paths：500；quantiles：5；regime window：20；seed=42。
- 没有重新训练、没有调参、没有利用 OOF 结果反向修改模型；所有随机化只重排/重采样已实现
  的测试期预测或收益。

## 2. Temporal Dependence Tests

### 2.1 Block permutation

主 block length = 20：

| mode | actual mean score | null mean | null 5–95% | p | Holm p |
|---|---:|---:|---|---:|---:|
| offline | 0.8332 | 0.5064 | 0.141–0.962 | 0.103 | 0.425 |
| online | 1.2530 | 0.8035 | 0.359–1.267 | 0.057 | 0.200 |

敏感性：

```
offline: null mean 0.49–0.61; p 0.085–0.203; Holm p 0.425
online:  null mean 0.72–0.92; p 0.025–0.106; Holm p 0.125–0.200
```

Online 在 L=40 的原始 p=0.025，但 Holm 后 0.125；L=5/20 的原始 p 约 0.05–0.057，Holm 后
约 0.20。不同 block length 下结论不稳定。

### 2.2 Circular shift

| mode | actual | null mean | null 5–95% | p |
|---|---:|---:|---|---:|
| offline | 0.8332 | 0.5405 | 0.120–1.063 | 0.147 |
| online | 1.2530 | 0.7628 | 0.301–1.245 | 0.050 |

Circular shift 的结果与 block permutation 一致：Online 有边缘信号，Offline 更弱；但在
多重检验和 6 折宽度下，**结论标记为 inconclusive**。

![Block sensitivity](../reports/validation/plots/01_temporal_block_sensitivity.png)

![Temporal null distributions](../reports/validation/plots/02_temporal_null_distributions.png)

## 3. Paired Statistical Comparison

### 3.1 Aggregate：concatenated vs fold mean

`diff = Online - baseline`；CI 为 95% moving-block bootstrap；`Holm` 为 3 个主比较的校正。

| baseline | level | Online | baseline | diff | 95% CI | p(two) | Holm |
|---|---|---:|---:|---:|---|---:|---:|
| Buy & Hold | concatenated | 1.2707 | 1.2804 | -0.0097 | [-0.883, 0.657] | 0.853 | 1.000 |
| Buy & Hold | fold mean | 1.2530 | 1.4975 | -0.2445 | [-0.990, 0.509] | 0.606 | 1.000 |
| Constant 1.25 | concatenated | 1.2707 | 1.2038 | +0.0669 | [-0.789, 0.726] | 0.698 | 1.000 |
| Constant 1.25 | fold mean | 1.2530 | 1.4109 | -0.1579 | [-0.884, 0.544] | 0.733 | 1.000 |
| Vol-only 1.0x | concatenated | 1.2707 | 1.1509 | +0.1197 | [-0.638, 0.814] | 0.550 | 1.000 |
| Vol-only 1.0x | fold mean | 1.2530 | 1.2665 | -0.0135 | [-0.624, 0.682] | 0.880 | 1.000 |

**所有主比较的 CI 都跨 0。结论是 inconclusive；不能声称 Online 优于或劣于这些基准。**

### 3.2 Fold-level：明显的 regime dependence

逐折差异不是均匀的：

| fold | vs Buy & Hold | vs Constant 1.25 | vs Vol-only |
|---|---:|---:|---:|
| 2015 | +0.449 [-1.667, 2.248] | +0.467 [-1.569, 2.199] | +0.384 [-0.685, 2.218] |
| 2017 | -2.262 [-3.842, 0.364] | -2.122 [-3.637, 0.429] | -2.421 [-3.868, 0.015] |
| 2019 | +0.516 [-1.525, 1.811] | +0.630 [-1.356, 1.853] | +1.083 [-0.714, 2.292] |
| 2021 | -0.900 [-3.219, 1.263] | -0.784 [-3.036, 1.317] | -0.712 [-3.003, 1.426] |
| 2023 | **+1.455 [0.301, 2.804]** | **+1.537 [0.430, 2.830]** | **+2.098 [1.220, 3.005]** |
| 2025 | -0.725 [-2.487, 0.747] | -0.676 [-2.352, 0.392] | -0.512 [-1.832, 0.255] |

- 2023 的逐折 CI 在三个比较中都不跨 0；但这是 6 个 fold 中选出的一个，未做多重检验，
  只能算 exploratory。
- 2017 的方向相反：Online 明显落后，尤其对 Vol-only（p=0.051）。
- 因此“平均不显著”不是“每年都没有差异”；而是“差异方向随 regime 翻转，平均值无法确认”。

![Paired forest](../reports/validation/plots/07_paired_forest.png)

![Paired by fold](../reports/validation/plots/08_paired_by_fold.png)

## 4. Exposure-Matched Baselines

严格因果的 matched baseline 是 `lag1`：第 t 天用第 t-1 天的 Online allocation，第一天为 0。
它几乎完整保留了 Online 的仓位分布。

| baseline | mean | median | std | q05 | q95 | score mean | continuous |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online | 0.652 | 0.383 | 0.685 | 0.000 | 1.914 | 1.253 | 1.271 |
| lag1 | 0.651 | 0.383 | 0.685 | 0.000 | 1.914 | 0.786 | 0.580 |
| expanding_mean | 0.688 | 0.780 | 0.279 | 0.281 | 1.133 | 1.055 | 1.098 |
| vol_tilted* | 0.890 | 0.715 | 0.666 | 0.127 | 2.000 | 1.156 | 1.172 |
| past_bootstrap | 0.849 | 0.841 | 0.294 | — | — | 0.849 | — |

`*` `vol_tilted` 尝试用过去 exposure 和 vol ratio 匹配，但实际平均仓位 0.890 高于 Online 的
0.652；**它不是严格 matched baseline**，只作为额外诊断。

Paired bootstrap：

- Online vs lag1：`+0.690`，CI `[-0.160, 1.218]`，p=0.111；fold mean `+0.467`，p=0.228。
- Online vs expanding_mean：`+0.173`，CI `[-0.565, 0.761]`，p=0.582；fold mean `+0.198`，p=0.412。
- `past_bootstrap` 的均值 0.849 vs Online 1.253，p=0.092。

结论：在相同/相近 exposure 下，Online 的点估计更高，但 CI 跨 0；**没有检出可靠的增量价值**。
`lag1` 的分数较低说明“当天的预测对齐”可能有一点作用，但不足以在 6 折上显著。

![Exposure scores](../reports/validation/plots/03_exposure_matched_scores.png)

![Exposure distributions](../reports/validation/plots/04_exposure_distributions.png)

## 5. Prediction Quality

### 5.1 Rank IC / Pearson IC

| mode | scope | mean rank IC | median | positive folds | 95% CI | p | status |
|---|---|---:|---:|---:|---|---:|---|
| offline | all | 0.0637 | 0.0812 | 5/6 | [0.024, 0.109] | 0.002 | directional (weak) |
| offline | low vol | 0.0709 | 0.0832 | 5/6 | — | — | inconclusive |
| offline | mid vol | 0.0661 | 0.1215 | 3/5 | — | — | inconclusive |
| offline | high vol | -0.0349 | 0.0377 | 2/3 | — | — | inconclusive |
| online | all | 0.0116 | 0.0031 | 3/6 | [-0.034, 0.064] | 0.526 | inconclusive |
| online | low vol | -0.0386 | -0.0579 | 2/6 | — | — | inconclusive |
| online | mid vol | 0.0089 | -0.0086 | 2/5 | — | — | inconclusive |
| online | high vol | 0.0593 | 0.1032 | 2/3 | — | — | inconclusive |

Offline 的弱正 IC 不等于 Online 有信号：Online 的 CI 跨 0，且 3/6 fold 为正。高/低波动
regime 的样本小、符号混杂，全部标记 inconclusive。Pearson IC 的均值：Offline 0.0507，
Online 0.0334，方向相同但更弱。

### 5.2 Prediction quantiles

平均每折 quintile 的 `target_metric`（bps）：

| mode | Q1 | Q2 | Q3 | Q4 | Q5 | Q5-Q1 | 单调 fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline | 1.08 | 9.65 | 1.77 | 6.61 | 15.96 | +14.88 | 0/6 |
| online | 8.75 | 4.18 | 0.33 | 12.19 | 9.49 | +0.73 | 0/6 |

Offline 的 Q5 最高，但 Q2 > Q3，且没有任何一折是严格单调的；Online 的排序更差，Q5 甚至低于
Q4。结论：**预测排序没有稳定单调性**；不能把当前 signal 当作可靠的 cross-sectional 排序器。

![Quantile returns](../reports/validation/plots/05_prediction_quantiles.png)

### 5.3 Regime IC

| fold | offline low | offline mid | offline high | online low | online mid | online high |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 0.12 | 0.16 | 0.04 | -0.17 | 0.08 | 0.10 |
| 2017 | 0.09 | — | — | -0.06 | — | — |
| 2019 | 0.20 | -0.02 | 0.08 | -0.06 | -0.01 | 0.19 |
| 2021 | 0.07 | 0.12 | — | 0.01 | -0.09 | — |
| 2023 | -0.11 | 0.14 | — | 0.16 | 0.15 | — |
| 2025 | 0.06 | -0.07 | -0.22 | -0.12 | -0.09 | -0.11 |

Regime 划分使用训练窗口 trailing market vol 的三分位；测试期只做分类。表格里大量空白是
因为该 regime 的测试日少于 20 天。整体结论：**regime 内的 IC 不稳定，全部 inconclusive**。

![IC by regime](../reports/validation/plots/06_ic_by_regime.png)

## 6. Allocation Attribution

### 6.1 Policy ladder

Mean score（6 折平均）：

| variant | offline | online | 说明 |
|---|---:|---:|---|
| full | 0.833 | 1.253 | 完整规则 |
| no_smoothing | 0.775 | 1.212 | 去掉 0.75/0.25 平滑 |
| no_vol | 0.987 | 1.092 | 去掉 vol scaling |
| no_multiplier | 0.336 | 0.360 | 600/400 乘数改为 1 |
| signal_only | 0.927 | 1.052 | 只有 clipped signal |
| signal_vol | 0.775 | 1.212 | signal + vol，无平滑/成本 |
| signal_vol_smooth | 0.833 | 1.253 | 加回平滑，无成本 |
| vol_only | 1.266 | 1.266 | 只有波动率目标 |

解读：

- **600/400 multiplier 是主导杠杆**：去掉后平均仓位降到约 0.009，score 从 1.25 降到 0.36。
- **smoothing 小幅为正**：Online full 1.253 vs no_smoothing 1.212（+0.041）；Offline
  0.833 vs 0.775（+0.058）。
- **vol scaling 的符号不稳定**：Online 去掉 vol 后从 1.253 降到 1.092（vol scaling 有贡献），
  Offline 去掉后从 0.833 升到 0.987（vol scaling 反而拖累）。
- **Vol-only 1.0x 仍然略高于 full**（1.266 vs 1.253/0.833），再次说明当前预测没有稳定超过
  纯波动率仓位。
- `signal_only` 也随 mode 翻转（Offline 0.927 > full 0.833；Online 1.052 < full 1.253），
  说明“平滑/波动率/预测”的交互很强，不能简单归因为某单一组件。

### 6.2 Position bounds and clipping

| mode | pre-clip <0 | pre-clip >2 | final exact 0 | final near 0 (<=0.01) | final exact 2 | final near 2 (>=1.99) |
|---|---:|---:|---:|---:|---:|---:|
| offline | 64.9% | 17.0% | 18.4% | 42.5% | 0.0% | 5.7% |
| online | 49.9% | 22.1% | 1.0% | 13.9% | 0.0% | 2.2% |

Clipping 很频繁：约一半的 raw target 在 Online 中低于 0，22% 高于 2；smoothing 使最终几乎
没有精确等于 2 的仓位，但仍有 2.2% 接近 2。Offline 的零仓比例高得多（18.4% 精确为 0，
42.5% 接近 0），这解释了它对强趋势年份的欠暴露。

### 6.3 Turnover and transaction cost

- 平均每日仓位变化：Offline 0.303，Online 0.514；Online 的换手约高 70%。
- 规则内置的乘法成本（3e-5/日）几乎不改变 score：Online 1.2530 → 1.2525（25 倍成本下）。
- 真正的 turnover-proportional cost 影响明显：

| mode | 0 bps | 1 bp | 5 bps | 10 bps | 25 bps |
|---|---:|---:|---:|---:|---:|
| offline | 0.833 | 0.792 | 0.584 | 0.254 | -0.081 |
| online | 1.253 | 1.141 | 0.763 | 0.337 | -0.162 |

在 5 bps/单位换手下，Online 的 mean score 从 1.253 降到 0.763，已经低于 Buy & Hold 的
1.498；因此成本敏感性与换手是必须纳入下一阶段评估的。

![Attribution ladder](../reports/validation/plots/09_allocation_ladder.png)

![Bounds](../reports/validation/plots/11_position_bounds.png)

![Turnover cost](../reports/validation/plots/12_turnover_cost.png)

## 7. 是否值得进入 Deep Learning？

**当前结论：不值得直接进入 DL 实现。**

更准确地说：本阶段**没有证据证明 GBDT 的预测信号有稳定的增量价值**，而不是证明它一定没有
价值。支持这个判断的证据是：

- 主比较的 paired bootstrap CI 全部跨 0；
- Online 的 rank IC 不显著，quantile 不单调；
- exposure-matched 的 `lag1`/`expanding_mean` 差异不显著；
- Vol-only 与 Online 的 mean score 基本相同（1.2665 vs 1.2530）；
- 归因显示 multiplier、vol scaling、smoothing 对结果的影响大于当前可验证的预测质量。

DL 只有在以下条件满足后才值得投入：

1. 先补 **daily-retrain 6 折 CV**：当前 OOF 的 offline 是每折训练一次，online 是逐日 update；
   部署实际是每天全量重训最后 800 行，这三种口径不能混用。
2. 重新校准 exposure/position 层（平均仓位、return penalty、vol target、smoothing），让
   GBDT 在 paired CI 下稳定超过 Buy & Hold/Constant 1.25/Vol-only。
3. 建立 signal gate：rank IC、quantile monotonicity、regime stability 至少要有稳定的正向
   证据，而不是靠单折 2023。
4. 用更多 rolling folds（或 expanding window）降低 CI 宽度；6 折的统计功效不足。
5. 把 turnover-proportional cost 纳入正式评估；当前 3e-5 的乘法成本不是实际交易成本。

## 8. Still Open / Inconclusive

- **Inconclusive**：Temporal dependence（block/circular 在 Holm 后不显著）。
- **Inconclusive**：Online vs Buy & Hold / Constant 1.25 / Vol-only 的平均差异。
- **Inconclusive**：Exposure-matched `lag1`/`expanding_mean` 的增量价值（点估计为正，CI 跨 0）。
- **Inconclusive**：Online 的 rank IC 与 regime IC；Offline 的弱正 IC 不能外推到 Online。
- **Descriptive only**：Allocation attribution 的分数差；不是因果识别，因为组件间有交互。
- **Open**：2017 vs 2023 的符号翻转是 regime、训练窗口还是 online update 造成？
- **Open**：daily-retrain 部署口径下的真实 OOF score/Sharpe 是多少？
- **Open**：Buy & Hold 在 6 个抽样年份的 1.498 是否会被更多 folds 或不同起点改变？
- **Open**：Vol-only 的目标波动率选择（scale 0.5/1.0/1.5）是否应作为正式基线参数？
- **Open**：E14 从 2026-06-16 起缺失 111 行；是否应在 GBDT/DL 特征中丢弃或加 missing indicator？
- **Open**：标准 turnover cost 下，Online 的优势是否在 1–5 bps 内就消失？
- **Open**：当前官方 score 对负 Sharpe 的惩罚方向异常；是否应同时报告 raw Sharpe 和连续 OOF 曲线？

## 9. Artifacts

```
reports/validation/temporal_summary.csv
reports/validation/temporal_folds.csv
reports/validation/temporal_replicates.csv
reports/validation/paired_comparison.csv
reports/validation/exposure_matched.csv
reports/validation/exposure_past_bootstrap.csv
reports/validation/prediction_quality.csv
reports/validation/prediction_quantiles.csv
reports/validation/prediction_ic_summary.csv
reports/validation/quantile_monotonicity.csv
reports/validation/allocation_attribution.csv
reports/validation/position_bounds.csv
reports/validation/cost_sensitivity.csv
reports/validation/turnover_cost_sensitivity.csv
reports/validation/summary.json
reports/validation/manifest.json
reports/validation/config_used.yaml
reports/validation/plots/01..12*.png
```

代码：

```
common/validation.py
scripts/run_validation.py
tests/test_validation.py
configs/validation.yaml
```

复现：

```bash
python scripts/run_validation.py
```

测试：`pytest -q` 当前 35 passed。
