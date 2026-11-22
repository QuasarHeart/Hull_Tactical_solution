# Phase 6 — Signal robustness, concentration and regime attribution

日期：2026-11-22  
协议：`docs/phase6_preregistration.md`（在运行 2026-07-06 之后窗口之前固定）  
配置：`configs/gdbt/phase6.yaml`  
脚本：`scripts/run_phase6_signal_robustness.py`  
产物：`reports/phase6_signal_robustness/`

> 评分说明：live 数据没有真实的 `forward_returns` / `risk_free_rate`，本报告使用
> `target_metric` 作为 forward return、`rf=0` 的 proxy score。它与此前 Phase 5 的
> 2.192119 同口径，但不能和 Kaggle 隐藏分数直接等同。

## 1. 结论先行

1. **信号有正的 point estimate，但没有通过多种稳健性门槛。** 官方窗口 daily 的日
   excess mean 为 0.000859（HAC t=2.193，单侧 p=0.014），但六个预注册检验的
   Holm 校正后 p=0.085；与 Buy & Hold、Constant 1.25、Vol-only 的差异也都不显著。
   Rank IC 0.114、Q5−Q1 0.00257，block bootstrap CI 都跨 0。
2. **信号在低/中波动环境下更强，在高波动下几乎消失。** 低波动 Rank IC 0.314、
   Q5−Q1 0.00878；高波动 Rank IC 0.077、Q5−Q1 −0.00005。但低/中波动各只有
   21 天，每个 quintile 只有 4–5 天，不能把这些点估计当作稳定的 regime 规律。
3. **收益高度集中。** 2026-01 和 2026-04 的前 3 个正贡献日分别占当月正贡献的
   62.1% 和 53.3%；前 5 个正贡献日占 90.3% 和 74.2%。删除 2026-04-10 至
   2026-04-16 的 5 日 block，官方 score 下降 0.653；删除 2026-01-20 至
   2026-01-26 的 5 日 block，score 下降 0.382。Leave-one-out 的最大单日影响为
   −0.214（2026-04-16）。
4. **multiplier 曲线是宽平台，不是单调最优。** 原始 600/400 为 2.192；fixed 400
   为 2.181、fixed 600 为 2.178；200 的 point estimate 最高（2.496），但这是
   事后观察，不能据此选参。fixed multiplier 比 600/400 switch 更简单，且
   100–800 的点估计都在 2.13–2.50 区间；是否存在更优的固定值，必须用独立窗口
   预注册验证。
5. **2026-07-07 之后的独立窗口没有推翻上述谨慎结论。** Daily 95 天 score
   1.382、累计 +7.42%、MDD −4.70%；Offline 2.711、累计 +10.98%、MDD −1.05%；
   但 daily 的 HAC 检验全部不显著，Offline 只是描述性结果，不能据此回头选模型。
6. **当前证据不足以把“正收益”升级为“可靠预测 alpha”。** 如果继续，GBDT 应作为
   冻结基线；不应再用这个窗口调 multiplier、regime 阈值或特征。

## 2. 实验一：Signal × Regime

Regime 阈值来自官方窗口之前 800 天的因果 volatility 分布：低波动
`vol <= 0.05552`，高波动 `vol >= 0.26433`；中间为 mid。阈值在 Phase 6 运行前
固定，官方窗口和独立窗口共用。

### 2.1 官方窗口（2025-12-16 至 2026-07-06，137 天）

| Regime | Days | Rank IC | Pearson IC | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 | Q1..Q5 样本数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| low_vol | 21 | 0.314 | 0.373 | −0.00632 | 0.00319 | 0.00547 | 0.00215 | 0.00246 | **0.00878** | 5/4/4/4/4 |
| mid_vol | 21 | 0.164 | 0.366 | −0.00328 | 0.00042 | 0.00098 | −0.00046 | 0.00075 | **0.00403** | 5/4/4/4/4 |
| high_vol | 95 | 0.077 | 0.017 | 0.00100 | 0.00059 | −0.00187 | 0.00352 | 0.00094 | **−0.00005** | 19/19/19/19/19 |

低波动的 Q1 明显为负、Q5 为正，方向与 IC 一致；但 Q2–Q5 不单调。中波动同样
不单调。高波动 95 天里 Q3 为负、Q4 最高，Q5 反而回落，几乎没有可解释的
monotonic signal。

### 2.2 独立 post-window（2026-07-07 至 2026-11-17，95 天）

| Regime | Days | Rank IC | Pearson IC | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_vol | 92 | 0.070 | 0.101 | 0.00081 | −0.00416 | 0.00007 | −0.00025 | 0.00178 | **0.00098** |
| mid_vol | 3 | −0.500 | −0.407 | — | — | — | — | — | — |

post-window 只有 3 天落在 mid-vol，没有 high-vol；这不是一个平衡的 regime
对照，不能据此说“信号在低波动更可靠”。低波动的 IC 也比官方窗口低得多。

## 3. 实验二：Multiplier robustness

固定 multiplier 网格在运行前写死为
`1, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200`；`1` 表示不使用
multiplier。原始 600/400 switch 单独列出。

| Multiplier | Mean alloc | Strategy vol (%) | Official score | Raw Sharpe | Cum. return | MDD | Turnover | Return penalty |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original 600/400 | 0.711 | 9.67 | **2.192** | 2.192 | 12.21% | −4.01% | 0.530 | 1.000 |
| 1 | 0.015 | 0.34 | 0.556 | 1.846 | 0.34% | −0.15% | 0.016 | 3.318 |
| 50 | 0.220 | 3.26 | 1.858 | 2.744 | 4.98% | −1.42% | 0.182 | 1.477 |
| 100 | 0.309 | 4.18 | 2.161 | 2.653 | 6.21% | −1.42% | 0.251 | 1.227 |
| 200 | 0.469 | 6.35 | **2.496** | 2.496 | 9.00% | −2.13% | 0.365 | 1.000 |
| 300 | 0.600 | 8.19 | 2.287 | 2.287 | 10.71% | −3.26% | 0.458 | 1.000 |
| 400 | 0.706 | 9.66 | 2.181 | 2.181 | 12.13% | −4.01% | 0.529 | 1.000 |
| 500 | 0.778 | 10.46 | 2.132 | 2.132 | 12.88% | −4.52% | 0.560 | 1.000 |
| 600 | 0.829 | 11.04 | 2.178 | 2.178 | 13.96% | −4.83% | 0.571 | 1.000 |
| 800 | 0.904 | 12.15 | 2.233 | 2.233 | 15.89% | −5.28% | 0.599 | 1.000 |
| 1000 | 0.943 | 12.94 | 2.144 | 2.144 | 16.27% | −5.64% | 0.612 | 1.000 |
| 1200 | 0.970 | 13.69 | 2.042 | 2.042 | 16.41% | −5.77% | 0.617 | 1.000 |

读法：

- multiplier=1 时平均仓位只有 0.015，score 0.556；没有 multiplier 的
  “纯信号”在这个评分制度下几乎不建仓，return penalty 3.318。高分确实依赖
  exposure layer，而不是 raw signal 单独完成。
- 从 100 到 800，score 在 2.13–2.50 的宽平台内；200 的 2.496 是最高的 point
  estimate，但曲线没有单调性，也没有预注册选择 200，不能把它当作最优参数。
- **400 与 600 几乎等价**（2.181 vs 2.178），尽管平均仓位 0.706 vs 0.829。
  这说明“600/400 的 switch”本身不是收益来源；一个固定 multiplier 就能达到
  相近的暴露与分数，机制可以更简单。
- 但“更简单”不等于“已经知道最优值”。fixed 200 在这个窗口更高，可能是
  regime/路径噪声；在没有独立确认前，不应把 200 写进部署配置。

## 4. 实验三：Signal concentration

### 4.1 正贡献集中度

| Month | Days | Total contribution | Positive contribution | Top1 share | Top3 share | Top5 share | Positive days needed for 50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | 20 | 2.99% | 4.11% | 24.3% | 62.1% | 90.3% | 3 |
| 2026-04 | 21 | 5.74% | 7.36% | 18.1% | 53.3% | 74.2% | 3 |

两个月都只需要 3 个正贡献日就覆盖一半的正贡献；1 月前 5 个正贡献日贡献了
90.3%。这不是一个由很多小 edge 累积起来的收益形态。

### 4.2 Leave-one-out

对 1 月和 4 月的每一天，删除该日并重新计算整个 137 天窗口的官方 score：

| 删除日 | 贡献 | Δ score | Δ cumulative |
| --- | ---: | ---: | ---: |
| 2026-04-16 | +1.33% | **−0.214** | −1.47% |
| 2026-04-07 | +1.33% | **−0.213** | −1.47% |
| 2026-04-14 | +1.26% | **−0.204** | −1.40% |
| 2026-01-02 | +1.00% | **−0.165** | −1.11% |
| 2026-04-20 | −1.13% | +0.263 | +1.29% |
| 2026-01-13 | −0.78% | +0.176 | +0.89% |

删除负贡献日反而提高分数，说明路径中既有“关键正贡献日”，也有“关键拖累日”；
净结果对少数日期敏感。

### 4.3 Block deletion

删除每月内每个连续的 3 日或 5 日 block，重新计算全窗口 score：

| Month | Block | Blocks | Min Δ score | Median Δ score | Max Δ score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | 3 | 18 | −0.256 | −0.005 | +0.180 |
| 2026-01 | 5 | 16 | **−0.382** | −0.074 | +0.198 |
| 2026-04 | 3 | 19 | −0.489 | −0.126 | +0.260 |
| 2026-04 | 5 | 17 | **−0.653** | −0.157 | +0.192 |

最差的 5 日 block：

- 2026-04-10 至 2026-04-16：Δ score −0.653，Δ cumulative −4.24%；
- 2026-04-07 至 2026-04-13：Δ score −0.536，Δ cumulative −3.62%；
- 2026-01-20 至 2026-01-26：Δ score −0.382，Δ cumulative −2.56%。

结论：**不能把 2.192 当作一个广泛、稳定的日频 edge**；它至少部分来自少数
交易日/短期 block。高 concentration 也解释了为什么 block bootstrap 的区间很宽。

## 5. 实验五：统计检验（官方窗口）

预注册的六个检验；HAC 使用 fixed 10-lag Newey-West；score 差异使用 moving
block bootstrap（block=20，B=2,000）；p 值经 Holm 校正。

| Hypothesis | Estimate | HAC t | raw one-sided p | Holm p | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Daily mean excess > 0 | 0.000859 | 2.193 | 0.014 | 0.085 | [0.00009, 0.00163] |
| Daily vs Buy & Hold | 0.000193 | 0.330 | 0.371 | 1.000 | [−0.00096, 0.00134] |
| Daily vs Constant 1.25 | 0.000027 | 0.036 | 0.486 | 1.000 | [−0.00143, 0.00148] |
| Daily vs Vol-only | 0.000439 | 1.099 | 0.136 | 0.679 | [−0.00034, 0.00122] |
| Rank IC > 0 | 0.114 | — | 0.485 | 1.000 | [−0.111, 0.234] |
| Q5−Q1 > 0 | 0.002575 | — | 0.389 | 1.000 | [−0.00359, 0.00668] |

Score-difference bootstrap（不是 Holm family 的 p 值，只作为区间诊断）：

| Pair | Actual score diff | 95% CI | two-sided p |
| --- | ---: | ---: | ---: |
| Daily − Buy & Hold | 1.040 | [−3.554, 3.368] | 0.474 |
| Daily − Constant 1.25 | 1.111 | [−3.349, 3.247] | 0.455 |
| Daily − Vol-only | 1.411 | [−0.239, 4.106] | 0.101 |

**不要只看原始 Sharpe/score**：daily 的 2.192 看似很高，但它相对于简单基准的
日收益差在 HAC 下不显著；score 差的 bootstrap CI 也很宽，包含 0。

## 6. 实验四：独立 post-window

协议在运行前固定；策略、参数、regime 阈值、bootstrap 设置均不因 post 结果改变。
起始日 2026-07-07，模型/特征在起始日前训练；policy 从零仓位开始；最后三个
target 尚未实现，所以窗口到 2026-11-17，共 95 天。

| Mode | Score (proxy) | Raw Sharpe | Cum. return | MDD | Turnover | Mean alloc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| daily | 1.382 | 1.382 | 7.42% | −4.70% | 0.704 | 0.867 |
| fixed_rolling | 1.382 | 1.382 | 7.42% | −4.70% | 0.704 | 0.867 |
| offline | **2.711** | 2.711 | 10.98% | −1.05% | 0.393 | 0.354 |
| vol_only | −0.185 | −0.776 | −6.99% | −12.60% | 0.021 | 1.955 |
| buy_hold | −0.607 | −0.607 | −2.82% | −6.33% | 0.011 | 1.000 |
| constant_1.25 | −0.567 | −0.623 | −3.60% | −7.90% | 0.013 | 1.250 |

Post-window 的简单基准是负的，daily 和 offline 是正的；但样本只有 95 天，且
post-window 的 HAC 检验全部不显著（daily mean excess t=0.991，p=0.161；
vs Buy & Hold p=0.110；rank IC p=0.376；Q5−Q1 p=0.489）。Offline 的 2.711 是
描述性结果；如果现在因为它高就改选 Offline/某个 multiplier，就是典型的
post-selection。Protocol 不允许这样做。

## 7. 结论与下一步

- **保留的结论**：信号方向为正、低/中波动点估计更强、multiplier 与 smoothing
  是主要 exposure 来源；但这些都还没有通过严格的多重检验和集中度检查。
- **不能保留的结论**：不能说“模型有稳定 alpha”“低波动一定更好”“200 是最优
  multiplier”“post-window 的 Offline 更好”。这些都需要新的、预注册的独立样本。
- **决策建议**：GBDT 不再调参；把 Daily Retrain、Offline、Vol-only、Buy & Hold
  作为冻结基线。若进入 DL，必须复用同一窗口、同一 allocation/cost、同一
  HAC/block bootstrap 和多重检验接口，并在看到结果前冻结。
- 研究限制：137 天官方窗口 + 95 天 post-window 仍然短；live 评分缺真实 rf；
  regime 样本严重不平衡；signal 集中度高；这些都会让 point estimate 很漂亮而
  inferential evidence 很弱。

## 8. Artifacts

```text
scripts/run_phase6_signal_robustness.py
configs/gdbt/phase6.yaml
docs/phase6_preregistration.md
reports/phase6_signal_robustness/signal_regime.csv
reports/phase6_signal_robustness/post_signal_regime.csv
reports/phase6_signal_robustness/multiplier_grid.csv
reports/phase6_signal_robustness/concentration_daily.csv
reports/phase6_signal_robustness/leave_one_out.csv
reports/phase6_signal_robustness/block_deletion.csv
reports/phase6_signal_robustness/statistical_tests_official.csv
reports/phase6_signal_robustness/statistical_tests_post.csv
reports/phase6_signal_robustness/post_metrics.csv
reports/phase6_signal_robustness/manifest.json
reports/phase6_signal_robustness/summary.json
reports/phase6_signal_robustness/plots/01..06
reports/live_window_20260707_20261117/positions.csv
reports/live_window_20260707_20261117/metrics.csv
```

Reproduce:

```bash
# 1. independent post-window (protocol already fixed)
python scripts/score_live_window.py \
  --start 2026-07-07 --end 2026-11-17 --warmup-start 2026-07-07 \
  --output reports/live_window_20260707_20261117

# 2. all Phase 6 analysis
python scripts/run_phase6_signal_robustness.py \
  --output reports/phase6_signal_robustness
```
