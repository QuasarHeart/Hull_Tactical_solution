# Hull Tactical GBDT — Final Research Report

日期：2026-11-22  
仓库：`/home/yys/Hull_Tactical_solution`  
范围：六折 walk-forward、部署口径回放、官方 137 天窗口、locked 95 天 post-window、
信号 robustness、最终审计。  
状态：GBDT 研究冻结；DL 尚未实现。

![Architecture](assets/architecture.png)

## 1. Executive summary

这项工作不是“再训练一个更高的模型”，而是回答一个更窄、更重要的问题：

> 现有 GBDT 信号在严格时间顺序、统一 allocation/成本模型和多种统计检验下，
> 是否真的优于 Buy & Hold、Constant 1.25 和 Vol-only？

结论分成三层：

1. **六折 walk-forward 没有证明稳定的增量价值。** 六折均值：
   Buy & Hold **1.498**、Constant 1.25 **1.411**、Vol-only **1.267**、
   Online **1.253**、Daily Retrain **0.871**、Offline **0.833**。
   多组 paired bootstrap 区间跨 0；不能把 Online 的 1.253 说成稳定 alpha。
2. **官方 137 天窗口的 point estimate 很强，但证据不稳健。**
   Daily Retrain 的官方公式复算为 **2.192**，累计收益 12.21%，MDD −4.01%；
   但 signal 反转、multiplier、smoothing、月份和 regime 归因显示，分数主要
   由 exposure layer 和少数月份/交易日驱动。Phase 6 的 HAC 检验中，
   daily excess mean 的单侧 raw p=0.014，但六检验 Holm 后 p=0.085。
3. **locked 95 天窗口没有确认稳定性。** 2026-07-07 至 2026-11-17 的 Daily
   score 为 1.382、Offline 为 2.711，但 HAC 检验全部不显著；post-window 的
   regime 严重不平衡（low vol 92 天、mid vol 3 天、没有 high vol）。Offline
   的 2.711 是描述性结果，不能反过来作为选择模型的依据。

**最终决策：** 冻结 GBDT 作为研究基线；不再用单一窗口调 multiplier、特征或
模型；不启动 DL，除非先定义新的、独立的、预注册的验证问题。

## 2. Research question and contribution

### 2.1 问题

Hull Tactical 的评分不是普通 Sharpe：

```text
strategy = rf * (1 - position) + position * fwd
score    = annualized_sharpe / (vol_penalty * return_penalty)
```

其中 `vol_penalty` 只在策略波动超过市场 1.2 倍时增加，`return_penalty`
只在策略平均收益低于市场时增加。这个非线性评分会让“低仓位、低波动但收益不足”
的路径看起来与高 Sharpe 路径很不一样。因此，本项目的核心不是预测 IC 本身，而是：

- 预测信号是否提供了增量方向信息；
- allocation / multiplier / smoothing 是否在替代信号贡献；
- 这些结论是否能跨 fold、跨 regime、跨时间窗口成立；
- 在真实交易成本、集中度和多重检验下还剩下什么。

### 2.2 贡献

- 把 notebook 风格的 GBDT 研究拆成共享层 `common/`、模型族 `gdbt/` 和预留的
  `dl/`；
- 实现了 strict deployment-faithful daily retrain replay，明确 `train_end < t`；
- 用 2×2 factorial 分离 feature-builder 漂移与模型重训；
- 用官方评分脚本复算 137 天窗口，并做 signal/allocation 消融；
- 在读取 post-window 之前预注册 multiplier grid、regime 阈值、concentration
  规则和 HAC/block-bootstrap/Holm 检验族；
- 把“没有证明”写成可以复核的限制，而不是包装成投资结论。

## 3. Data, architecture and causal protocol

### 3.1 Data

| 数据 | 规模 | 说明 |
| --- | ---: | --- |
| `data/train.csv` | 9,048 × 98 | 1990-01-02 起的比赛训练数据；94 个特征 + `date_id` + 三个收益字段 |
| `data/test.csv` | 10 × 99 | 10 个 replay 行；不等于独立样本 |
| `data/date_map.csv` | 9,291 × 2 | `date_id` 到日历日期，覆盖到 2026-11-20 |
| `data/lake/panel_live.parquet` | 9,291 × 100 | live panel；最后 3 个 target pending；gitignored |
| `data/raw/hull/20261120_data.csv` | 9,291 × 96 | 原始 Hull CSV；gitignored |

完整哈希、行数、列数和日期范围见 `data/manifest.json` 与
`docs/data_manifest.md`。

### 3.2 Architecture

- `common/`：config、data、schema、features、transforms、splits、metrics、
  allocation、backtest、pipeline、ingest、registry、tracker、validation、
  evaluation；
- `gdbt/`：ElasticNet + XGBoost + LightGBM，notebook parity builder 与 production
  builder 分开；
- `dl/`：只有接口预留，当前没有实现；
- `configs/`：base、CV、Phase 4、Phase 5、Phase 6；
- `scripts/`：replay、CV、baseline、validation、daily retrain、Phase 4/5/6；
- `deploy/`：systemd timer/service；VPS 实机尚未验证；
- `docs/assets/`：架构图和核心结果图。

关键约束：

- 每个预测日只使用 `date_id < t` 且 target 已实现的数据；
- `train_end < t` 是 hard assertion，不是注释；
- live 数据的 pending target 不填 0；
- 缺失特征用训练窗口 median 填充；E14 当前保持 median fill；
- 所有 allocation 规则和 transaction cost 必须在比较中保持一致。

### 3.3 Evaluation windows

| 口径 | 日期/样本 | 用途 |
| --- | --- | --- |
| Notebook replay | `test.csv` 10 行 | 只验证代码对齐，不当作泛化 |
| Six-fold walk-forward | 2015/2017/2019/2021/2023/2025，共 1,488 天 | 主样本外比较 |
| Deployment replay | 同六折 | 严格每日重训口径 |
| Official live window | 2025-12-16–2026-07-06，137 天 | 官方公式复算与 attribution |
| Locked post-window | 2026-07-07–2026-11-17，95 天 | 预注册后的独立验证；last 3 targets pending |

live 数据缺真实的 `forward_returns` / `risk_free_rate`；官方分数使用
`fwd = target_metric, rf = 0` 的 proxy。这个 proxy 与此前 2.192 的复算同口径，
但不是隐藏 leaderboard 分数。

## 4. Results

![Core evidence map](assets/core_results.png)

### 4.1 Six-fold walk-forward: no stable incremental value

| Strategy | Fold-mean official score | Continuous score | Mean raw Sharpe | Mean allocation | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy & Hold | **1.498** | 1.280 | 1.498 | 1.000 | 0.004 |
| Constant 1.25 | **1.411** | 1.204 | 1.481 | 1.250 | 0.005 |
| Vol-only 1.0x | 1.267 | 1.151 | 1.486 | 1.200 | 0.066 |
| Online GBDT | 1.253 | 1.271 | 1.435 | 0.652 | 0.515 |
| Daily Retrain | 0.871 | 0.808 | 1.114 | 0.659 | 0.520 |
| Offline GBDT | 0.833 | 0.419 | 0.992 | 0.495 | 0.303 |

读法：

- Online 只比 Vol-only 低约 0.014 的 fold-mean score，远小于它与 B&H 的差距；
- Daily Retrain 在更严格的部署口径下没有复现 Online 的优势，反而低于所有简单
  基准；
- 这不是“单个坏折”造成的：fold-paired 差值在多个折上方向分散，均值仍为负；
- 因此，**历史 OOF 没有证明稳定的增量价值**。详见 `docs/gbdt_validation.md`、
  `docs/deployment_eval.md`。

### 4.2 Deployment-faithful daily retrain and 2×2 attribution

Phase 3 每个预测日重新拟合 feature builder 和三个模型，并且只使用当时已实现的
最近 800 行。六折结果：Daily Retrain 0.871，低于 Online 1.253、B&H 1.498、
Constant 1.25 1.411、Vol-only 1.267。

Phase 4 用 2×2 factorial 分离两个因素：

| Arm | Builder | Model | Fold-mean score |
| --- | --- | --- | ---: |
| A | fixed | fixed | 0.833 (Offline) |
| B | fixed | rolling | 1.253 (Online) |
| C | dynamic | fixed | 0.524 (Phase 4C) |
| D | dynamic | rolling | 0.871 (Daily Retrain) |

点估计显示 dynamic builder 是负向、rolling model 是正向，但四个直接 contrast
的 paired block-bootstrap 95% CI 全部跨 0，Holm 后全部不显著。结论只能是
**descriptive**：没有足够证据把 Online→Daily 的差距归因为单一机制。详见
`docs/feature_builder_attribution.md`。

### 4.3 Official 137-day window

官方窗口结果（官方公式，`fwd=target_metric, rf=0`）：

| Protocol | Official score | Raw Sharpe | Cumulative | MDD | Turnover | Mean alloc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily Retrain | **2.192** | 2.192 | 12.21% | −4.01% | 0.530 | 0.711 |
| Fixed builder + rolling | 2.196 | 2.196 | 12.23% | −4.01% | 0.535 | 0.714 |
| Offline | 2.043 | 2.043 | 11.24% | −3.67% | 0.353 | 0.455 |
| Vol-only 1.0x | 0.782 | 1.038 | 5.66% | −4.52% | 0.091 | 0.652 |
| Buy & Hold | 1.152 | 1.152 | 9.00% | −9.66% | 0.007 | 1.000 |
| Constant 1.25 | 1.081 | 1.135 | 11.19% | −11.97% | 0.009 | 1.250 |

这个窗口是第一个模型明显领先简单基准的真实样本外窗口。但它仍然只有一个
实现路径；下面的 attribution 和 locked window 是在检验它能否成为可重复的结论。

### 4.4 Signal and allocation attribution

Phase 5 在已保存路径上做消融，不重新训练：

| Variant | Official score | Cum. return | MDD | Turnover | Mean alloc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original signal + allocation | **2.192** | 12.21% | −4.01% | 0.530 | 0.711 |
| Reverse raw signal | 0.055 | 1.28% | −11.15% | 0.463 | 0.627 |
| No multiplier (`multiplier=1`) | 0.556 | 0.34% | −0.15% | 0.016 | 0.015 |
| Fixed 600 | 2.178 | 13.96% | −4.83% | 0.571 | 0.829 |
| Fixed 400 | 2.181 | 12.13% | −4.01% | 0.529 | 0.706 |
| No smoothing | 2.014 | 12.81% | −4.91% | 0.733 | 0.712 |

结论：

- 方向有贡献：反转信号把分数打到 0.055；
- 但 raw signal 单独不足以建立仓位：multiplier=1 时平均仓位 0.015、分数 0.556；
- 600/400 switch 不是主要来源：fixed 400/600 与原始 2.192 非常接近；
- smoothing 对风险调整分数有帮助：去掉后 turnover 上升、MDD 恶化；
- prediction bins 不单调，rank IC 0.114，shuffle p≈0.052；这不是干净的
  timing alpha。

### 4.5 Signal × regime, multiplier and concentration

**Signal × regime（官方 137 天）**

| Regime | Days | Rank IC | Pearson IC | Q5−Q1 | Q1..Q5 counts |
| --- | ---: | ---: | ---: | ---: | --- |
| low vol | 21 | 0.314 | 0.373 | 0.00878 | 5/4/4/4/4 |
| mid vol | 21 | 0.164 | 0.366 | 0.00403 | 5/4/4/4/4 |
| high vol | 95 | 0.077 | 0.017 | −0.00005 | 19/19/19/19/19 |

低/中波动看起来有信号，但各只有 21 天、每个 quintile 只有 4–5 天；高波动
95 天几乎为零。post-window 的 low vol 92 天、mid vol 3 天、没有 high vol，
regime 分布已经变化，不能把官方窗口的 pattern 外推。

**Multiplier（官方窗口，预注册 grid）**

| Multiplier | Mean alloc | Score | Cum. return | MDD | Turnover |
| ---: | ---: | ---: | ---: | ---: | ---: |
| original 600/400 | 0.711 | 2.192 | 12.21% | −4.01% | 0.530 |
| 1 | 0.015 | 0.556 | 0.34% | −0.15% | 0.016 |
| 100 | 0.309 | 2.161 | 6.21% | −1.42% | 0.251 |
| 200 | 0.469 | **2.496** | 9.00% | −2.13% | 0.365 |
| 400 | 0.706 | 2.181 | 12.13% | −4.01% | 0.529 |
| 600 | 0.829 | 2.178 | 13.96% | −4.83% | 0.571 |
| 800 | 0.904 | 2.233 | 15.89% | −5.28% | 0.599 |

200 是 point-estimate 最高，但这是事后观察；100–800 是宽平台，400/600 几乎
等价。更简单的 fixed multiplier 在机制上可行，但**不能**根据这个窗口选 200。

**Concentration（2026-01 与 2026-04）**

| Month | Days | Top1 share | Top3 share | Top5 share | Positive days for 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | 20 | 24.3% | 62.1% | 90.3% | 3 |
| 2026-04 | 21 | 18.1% | 53.3% | 74.2% | 3 |

删除 2026-04-10 至 2026-04-16 的 5 日 block，score 下降 **0.653**；删除
2026-01-20 至 2026-01-26 的 5 日 block，score 下降 **0.382**。逐日
leave-one-out 的最大单日影响是 2026-04-16 的 −0.214。这个 2.192 不是由很多
小 edge 平均累积出来的。

### 4.6 Statistical inference

官方窗口六项预注册检验（HAC 10 lags；block bootstrap block=20，B=2,000；
Holm 校正）：

| Hypothesis | Estimate | HAC t | raw one-sided p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| Daily mean excess > 0 | 0.000859 | 2.193 | 0.014 | **0.085** |
| Daily vs Buy & Hold | 0.000193 | 0.330 | 0.371 | 1.000 |
| Daily vs Constant 1.25 | 0.000027 | 0.036 | 0.486 | 1.000 |
| Daily vs Vol-only | 0.000439 | 1.099 | 0.136 | 0.679 |
| Rank IC > 0 | 0.114 | — | 0.485 | 1.000 |
| Q5−Q1 > 0 | 0.002575 | — | 0.389 | 1.000 |

Score-difference bootstrap：

| Pair | Score diff | 95% CI | two-sided p |
| --- | ---: | ---: | ---: |
| Daily − Buy & Hold | 1.040 | [−3.554, 3.368] | 0.474 |
| Daily − Constant 1.25 | 1.111 | [−3.349, 3.247] | 0.455 |
| Daily − Vol-only | 1.411 | [−0.239, 4.106] | 0.101 |

raw score 2.192 不能单独作为证据。经多重检验后，没有任何预注册检验达到 5%
显著水平；正确的表述是“没有检出可靠差异”，而不是“已经证明模型更好”。

### 4.7 Locked post-window

2026-07-07 至 2026-11-17（95 个已实现交易日；最后 3 个 target pending）：

| Mode | Score (proxy) | Cum. return | MDD | Turnover | Mean alloc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daily Retrain | 1.382 | 7.42% | −4.70% | 0.704 | 0.867 |
| Fixed builder + rolling | 1.382 | 7.42% | −4.70% | 0.704 | 0.867 |
| Offline | **2.711** | 10.98% | −1.05% | 0.393 | 0.354 |
| Vol-only | −0.185 | −6.99% | −12.60% | 0.021 | 1.955 |
| Buy & Hold | −0.607 | −2.82% | −6.33% | 0.011 | 1.000 |
| Constant 1.25 | −0.567 | −3.60% | −7.90% | 0.013 | 1.250 |

Daily 和 fixed_rolling 在这 95 天的 `raw_pred`、`allocation`、`vol`、`multiplier`
最大差异都是 0；daily 的完整重训没有比 fixed-builder rolling 产生额外差异。
Offline 更高，但这是预注册之外的事后观察，不能据此改选 Offline。

## 5. Limitations — 必须和结果一起读

这一节不是给项目“降温”，而是让结论可信。以下六点是结论的适用边界。

### 5.1 单一比赛窗口的收益集中问题

137 天窗口的 2.192 对少数月份和少数交易日高度敏感。1 月和 4 月贡献了大部分
收益；前 3 个正贡献日占当月正贡献的 62%/53%，前 5 个占 90%/74%。删除一个
5 日 block 可以使全窗口 score 下降 0.65/0.38。这意味着：

- 2.192 不是“日频稳定 edge”的证据；
- 一个短期窗口的分数可能由一个 regime、一次行情或几天仓位路径决定；
- 任何基于这段窗口的参数选择都会过拟合到这些日期。

### 5.2 预测信号统计证据有限

- rank IC 0.114，block bootstrap CI 跨 0；Q5−Q1 0.002575，CI 跨 0；
- prediction bins 不单调，Q4 最好、Q5 回落；
- Phase 5 shuffle p≈0.052，未达到 5%；
- Phase 6 daily excess mean 的 raw p=0.014，但六个检验 Holm 后 p=0.085；
- 与 B&H、Constant、Vol-only 的日收益差 HAC p 分别 0.371/0.486/0.136。

因此，可以说“信号方向可能有贡献”，不能说“已经证明存在稳定 alpha”。

### 5.3 历史 OOF 未证明稳定增量

六折均值：B&H 1.498、Constant 1.25 1.411、Vol-only 1.267、Online 1.253、
Daily 0.871。Online 只比 Vol-only 高 0.014；Daily 在严格部署口径下低于所有
简单基准。Phase 4 的 2×2 直接 contrast 的 paired CI 全部跨 0。历史 OOF 的
结论是 **inconclusive**，不是“模型有效”或“模型无效”的二选一。

### 5.4 独立窗口仅 95 天且 regime 不平衡

post-window 只有 95 天；其中 low vol 92 天、mid vol 3 天、没有 high vol。
它无法覆盖官方窗口里 95 天的高波动状态，也无法稳定估计 regime-specific IC。
Daily 的 HAC 结果全部不显著；Offline 2.711 也没有被预注册为 primary test。
95 天可以否决“立刻扩大 GBDT 调参”的冲动，但不足以确认一个长期 alpha。

### 5.5 尚未充分证明交易成本后的经济价值

- 当前 allocation 里的成本是 `3e-5` 的乘法项，不是完整的 bid-ask、滑点、
  融资/借券、税费和市场冲击模型；
- 137 天 turnover 0.530，post-window daily 0.704；成本假设稍微变大就可能
  吞掉大部分日频 edge；
- 没有真实资金曲线、没有 borrow availability、没有容量分析；
- 因此，2.192 是评分口径下的 proxy，不应直接解释为可实现净收益。

### 5.6 Offline 和 Daily 在不同时期表现不同

| Period | Daily | Offline | 观察 |
| --- | ---: | ---: | --- |
| 六折 CV | 0.871 | 0.833 | Daily 略高 |
| 官方 137 天 | 2.192 | 2.043 | Daily 略高 |
| locked 95 天 | 1.382 | 2.711 | Offline 明显更高 |

排序在窗口之间翻转；这不是“选一个更好的模型”就能解决的问题，而是 regime、
路径依赖和样本量的共同结果。不能用 locked 窗口的最高分反选 Offline，也不能
用官方窗口的最高分反选 Daily。

## 6. Decision and next steps

### 6.1 已决定

- 冻结当前 GBDT；不再用单一窗口调 multiplier、smoothing、特征或模型；
- 保留 Daily、Offline、Vol-only、B&H、Constant 作为后续比较基线；
- 不把 post-window 的 Offline 2.711 当作选择依据；
- 不启动 DL；DL 只有在新的预注册问题下才进入；
- 所有后续实验复用 `common/evaluation.py` 的 score、bootstrap、HAC/validation
  接口和同一 allocation/cost 口径。

### 6.2 若要继续 DL，最低门槛

1. 冻结新的 hypothesis、窗口、regime 阈值和检验族；
2. 在同一 6 折和 locked 窗口上比较，不只报 raw Sharpe；
3. 报告 HAC、block bootstrap、Holm 和成本敏感性；
4. 证明增量不是由少数交易日、单一 regime 或 exposure 变化造成；
5. 若没有通过，保持 DL 不实现，而不是用更多模型堆叠来寻找显著性。

## 7. Reproducibility and version audit

### 7.1 Tests

```bash
cd /home/yys/Hull_Tactical_solution
/home/yys/miniconda3/envs/dlwpt/bin/python -m pytest -q --disable-warnings
# 42 passed
```

覆盖：metrics、panel/date_map、feature builder、no-leakage、notebook parity、
official score、experiments、validation、evaluation、ingest、Phase 6 HAC /
bootstrap。

### 7.2 Data manifest

- `data/manifest.json`：machine-readable hashes, row counts, columns, date ranges,
  target nulls, raw source and ingest warnings；
- `docs/data_manifest.md`：人类可读的 provenance / verification 说明；
- `scripts/build_data_manifest.py`：重新生成 manifest；
- raw 与 `data/lake/` 保持 gitignored，但哈希仍在 manifest 中。

### 7.3 Presentation assets

- `docs/assets/architecture.html`：可交互阅读的架构图源文件；
- `docs/assets/architecture.png`：README / report 使用的静态图；
- `docs/assets/core_results.png` 与 `.svg`：四面板核心结果图；
- `scripts/make_phase8_assets.py`：从 reports 重建核心图。

### 7.4 Git and release

Phase 6 与 Phase 8 的代码、配置、报告、图表和审计文件按功能拆分提交；
`reports/`、`models/`、`wandb/`、`data/raw/`、`data/lake/` 不进入 Git。
最终提交记录用以下命令查看：

```bash
git log --oneline --decorate -8
git status --short
```

push 由用户决定，不自动执行。

## 8. Artifact index

| Artifact | Purpose |
| --- | --- |
| `docs/repro_gdbt.md` | notebook / CV / daily deployment 复现 |
| `docs/experiments.md` | 全项目实验索引与 Phase 2 细节 |
| `docs/gbdt_validation.md` | 时间 null、bootstrap、exposure matching |
| `docs/deployment_eval.md` | Phase 3 部署口径与 E14 |
| `docs/feature_builder_attribution.md` | Phase 4 2×2 归因 |
| `docs/phase5_competition_window_attribution.md` | Phase 5 官方窗口消融 |
| `docs/live_window_20251216_20260706.md` | 137 天窗口与扩展说明 |
| `docs/phase6_preregistration.md` | locked-window 预注册 |
| `docs/phase6_signal_robustness.md` | Phase 6 signal/regime/multiplier/concentration |
| `docs/research_decisions.md` | D1–D16 决策历史 |
| `docs/data_manifest.md`, `data/manifest.json` | 数据/版本审计 |
| `docs/assets/architecture.png` | 系统架构图 |
| `docs/assets/core_results.png` | 核心结果图 |

## 9. Final statement

这个仓库现在证明的不是“GBDT 可以稳定赚钱”，而是：

> 我们能够在严格时间顺序下复现一个 GBDT 研究管线，能够把它的分数拆成
> signal、allocation、multiplier、smoothing、regime 和少数交易日的贡献，
> 能够在独立窗口上提前冻结假设，并诚实地报告“证据不足”。

在这个标准下，Phase 6/8 的结论是 **inconclusive but auditable**：保留模型作为
基线，停止用同一窗口寻找更好看的数字；如果未来进入 DL，必须先解决样本、
成本和统计功效问题，而不是换一个更复杂的模型。
