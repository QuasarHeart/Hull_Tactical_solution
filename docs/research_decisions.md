# Research Decisions — GBDT Validation

日期：2026-11-21  
范围：在进入 Deep Learning 之前，验证现有 GBDT 策略相对简单基准的增量价值。  
约束：只复用已保存的 6 折 OOF；不重新训练、不调参、不修改正式预测文件。

## Decision log

### D1. 暂停 Deep Learning 实现

在 GBDT 的增量价值没有通过统计门槛之前，不启动 DL 模型实现。理由：如果当前收益主要来自
exposure/volatility scaling，而不是预测信号，换一个序列模型不会自动产生增量价值；先建立可
证伪的 GBDT 基线更便宜。

### D2. 评估只使用已实现的 OOF，不反向调整模型

Temporal nulls、bootstrap、exposure baselines、IC、quantile 和 attribution 都只读取
`oof_predictions.csv` 与已实现的 `forward_returns/risk_free_rate`。没有使用 OOF 结果重新训练
或选择模型；本阶段的输出不能反向修改正式预测。

### D3. Block length 主口径 20，敏感性 5/10/20/40/60

20 个交易日约等于一个交易月，保留短中期自相关；同时报告 5/10/40/60 的敏感性。
Block permutation 和 circular shift 都只改变预测与日期的对齐，保留 `vol`、`multiplier`、
smoothing 和成本规则。主检验不预设某个 block length 最好；如果不同 L 下结论翻转，标记为
inconclusive。

### D4. Paired comparison 使用 fold-stratified moving-block bootstrap

每个 fold 内按 block 重采样日期，同一组索引同时用于 Online 和 baseline；不跨 fold 拼接
block。报告两个层级：

- `concatenated`：把 6 个 fold 的重采样结果按顺序拼成一条 OOF 曲线；
- `fold_mean`：每个 fold 单独算 score，再平均。

Bootstrap 只用于不确定性，不用于模型选择；主比较 3 个，exposure 比较 2 个，使用 Holm
校正。fold-level 的逐折结果不做事后多重检验校正，只作为 exploratory evidence。

### D5. Exposure-matched baselines 必须因果

严格 matched baseline 用 `lag1`：第 t 天使用第 t-1 天的 Online allocation，第一天为 0。
它几乎完全保留 Online 的仓位分布（mean 0.651 vs 0.652，median 0.383 vs 0.383），但不是
当前预测。另加 `expanding_mean`（只用过去 allocation 的均值）和 `past_bootstrap`（从过去
allocation 的经验分布中抽样）作为低换手/分布对照。

`vol_matched` 只作为额外诊断：它尝试用过去 exposure 和 vol ratio 匹配，但实际平均仓位
0.890 高于 Online 的 0.652，**不应当称作严格 exposure-matched**。

### D6. 现金和不可验证情形的约定

全零仓位的官方 score 是 0/0；所有现金对照统一记为 `score=sharpe=0`，并在表格中标注 `cash`。
预测 ablation 的训练均值为负时选择空仓，不强行做多。

### D7. 结论标签

- `directional`：bootstrap CI 不跨 0，且方向在多数 fold 一致；仍只能算弱证据。
- `inconclusive`：CI 跨 0、fold 方向混杂、样本太小或口径不唯一。
- 不做“没有显著差异 = 没有差异”的表述；正确的说法是“没有检出可靠差异”。

### D8. 阶段结论

截至本阶段：**没有足够证据支持进入 Deep Learning**。GBDT Online 与 Buy & Hold、Constant
1.25、Vol-only 的平均差异在 paired bootstrap 下均不显著；在线模型的预测 IC 也不显著。
详见 `docs/gbdt_validation.md`。

### D9. Deployment-faithful 是最终验收口径

Phase 3 增加了逐日重训的回放：每个测试日 `t` 只用 `date_id < t` 且 target 已实现的最近
800 行，重新拟合 feature builder（缺失率、median、派生特征）和三个 GBDT 模型，再预测 `t`。
fold 开始时重置仓位状态；`t` 的 target 只在预测之后用于更新 policy。任何 `train_end >= t`
都视为泄漏并直接失败。这个口径比固定 builder 的 Online 更接近实际部署。

### D10. E14 outage 与 OOF 的关系

E14 的 111 行缺失从 2026-06-16 开始，而六折最后一折的测试日在 2025-12-03；六折 OOF
与 outage 的重叠为 0。因此 Phase 2/3 的六折结果不能归因于 E14 缺失。对 post-competition
live window 的检查是探索性的，且只能使用 proxy score（`fwd=target_metric`, `rf=0`）：

- 删除 E14 在两个 setting 下都更差；中位数填充不劣于删除。
- `E14_MISSING` 的 XGBoost importance 为 0，预测路径与中位数填充完全一致；当前没有证据
  支持加入 missing indicator。
- static 与 rolling-retrain 的巨大差异混入了市场 regime、重训和样本期变化，不能当作
  E14 的因果效应。

生产决策：继续 median fill，记录缺失率；暂不加入 indicator，等有对照 A/B 或更长已实现
窗口再决定。

### D11. Phase 3 结论

Daily retrain 六折官方分数均值 0.871，低于 Online 1.253，也低于 Buy & Hold 1.498、
Constant 1.25 1.411、Vol-only 1.267；continuous score 0.808，同样低于 Online 和简单基准。
更严格的部署口径**没有复现** Online 的优势，反而暴露了 feature builder 逐日重建带来的
路径不稳定。因此不进入 Deep Learning；下一步优先做固定 feature 与动态 feature 的对照，
并先稳定仓位路径。详见 `docs/deployment_eval.md`。

### D12. Phase 4 2×2 factorial 归因

Phase 4 用同一批六折、同一套 allocation 和 hyperparameters 构造 2×2：

- A = fixed builder + fixed model（existing Offline）
- B = fixed builder + rolling model（existing Online）
- C = dynamic builder + fixed model（new Phase 4C）
- D = dynamic builder + rolling model（existing Daily Retrain）

C 的 fixed model 保持 fold-start 的输入 schema；dynamic builder 缺失的 fixed 列用 fold-start
median 填充（实际为 0 个），额外列忽略并记录。所有 OOF 都复制到
`reports/phase4_attribution/oof/` 并记录 SHA-256，不修改正式预测。

结果：A 0.833，B 1.253，C 0.524，D 0.871（fold-mean official score）。Point estimate
显示 dynamic builder 是负向主效应（fixed-builder mean 1.043 vs dynamic 0.697），rolling
model 是正向主效应（0.678 vs 1.062），interaction -0.072；但四个直接 contrast 的
paired block-bootstrap 95% CI 全部跨 0，Holm 后全部不显著。因此只能把机制写成
descriptive，不能声称已建立因果关系。

### D13. Phase 4 decision gate

没有任何 configuration 对 Buy & Hold、Constant 1.25、Vol-only 形成 robust incremental
value；B 的点估计最高，但 B 就是 existing Online，且所有 paired CI 跨 0。继续调 GBDT
不会解决这个问题，**停止 further GBDT optimization**，本阶段不实现 DL。

可部署性判断：B（冻结 feature builder + rolling model）是唯一在点估计上接近 Online 的
可部署形式；当前 D（每日重建 builder + 每日重训）没有复现它。若要保留 GBDT，下一步是
实现 fixed-builder rolling 的 live job，而不是继续增加模型复杂度；但在没有稳健增量证据
前不建议投入。

Common evaluation interface 已提供：`paired_block_bootstrap`（fold-aware，primary
fold-mean，secondary 合成 concatenated）、`summarize_turnover_adjusted`、
`holm_adjust`，后续 GRU/TCN 复用同一六折、labels 和 allocation 评估。详见
`docs/feature_builder_attribution.md`。

### D14. Phase 5 competition-window attribution

Phase 5 在官方 137 天窗口（2025-12-16～2026-07-06）上做 signal/allocation 消融，
不重新训练模型，只复用已保存的 raw signal、vol、multiplier 和 allocation。
原始路径官方脚本复算 **2.192119**；`simulate_positions` 与原 allocation 最大差 0。

关键结果：

- 反转 raw signal：0.055；方向重要，但 long-only 下不是对称空头。
- `multiplier=1`：0.556，mean allocation 0.015；高分依赖 multiplier 提供的
  exposure，而不是 raw signal 单独产生仓位。
- fixed 600 / fixed 400：2.178 / 2.181；600/400 切换本身不是主要来源。
- 去掉 smoothing：2.014，turnover 0.530→0.733，MDD -4.01%→-4.91%；平滑改善
  风险调整收益。
- IID shuffle p=0.0525，block L=20 p=0.0519；实际 2.192 位于 null 尾部但
  **未达到 5% 显著**。prediction bins 不单调，rank IC 0.114；signal timing
  的增量仍是弱证据。
- 月度优势集中在 2026-01 和 2026-04；2026-02 为负。vol regime 中 high-vol
  贡献 9.23% 绝对收益，low-vol 分数最高（4.79），mid-vol 基本走平。

结论：官方窗口的分数复现无误，但 score 对 multiplier/smoothing 很敏感；
signal 方向有贡献，但统计上仍不够强。继续调 GBDT 不符合收益/复杂度权衡；本阶段
不实现 DL。

## Anti-leakage guardrails

1. Phase 2/3 validation 不重新训练；Phase 4C 重新训练 C，但严格使用 `date_id < t` 的数据，
   且没有用 OOF 结果调参或选择模型。
2. Temporal nulls 只重排/平移已实现的测试期预测。
3. Exposure baselines 只使用严格早于当天的 allocation。
4. Regime 阈值来自训练窗口的 trailing volatility；测试期只做分类。
5. Block bootstrap 只用于置信区间，不参与模型选择。
6. 所有随机实验固定 seed，并保存配置和重复次数。
7. Daily retrain 对每个测试日强制要求 `train_end < test_date_id`，并使用当时已实现的
   target；失败即中止，不静默跳过。

## Open decisions

- 是否接受“平均 score 不显著优于 Buy & Hold”作为停止 DL 的依据？
- 是否先重做仓位/暴露层，再重新跑本报告的全部检验？
- 是否把评估扩展到更多 rolling folds，以缩小 CI？
- 是否将 turnover cost 作为正式评分的附加约束？
- E14 暂定 median fill、无 indicator；需要更长已实现窗口或 live A/B 再决定。
- Daily retrain 与 Online 的差距需分解为 feature median drift、模型重训随机性和仓位路径
  依赖；在解释前不要把它归因为单一原因。
