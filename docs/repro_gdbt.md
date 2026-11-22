# GBDT 本地复现报告

日期：2026-11-21

环境：`/home/yys/miniconda3/envs/dlwpt/bin/python`（Python 3.12，xgboost 3.3.0，lightgbm 4.6.0，polars 1.44.2，wandb 0.31.2）

状态：本地复现完成，W&B（Weights & Biases）已接入；GBDT Notebook、交叉验证（CV）的离线（offline）与在线（online）两种模式均已跑完，每日数据链路已打通（2026-11-21）。Phase 3 的忠实部署（deployment-faithful）逐日重训与 E14 检查、Phase 4 的特征构建器（feature builder）归因也已完成，详见 `docs/deployment_eval.md` 和 `docs/feature_builder_attribution.md`。

## 1. 实现内容

```
common/
  config.py       YAML 配置、路径、列常量、target 偏移
  schema.py       94 特征、面板校验
  data.py         比赛 CSV -> panel（date_id, target, target_metric, fwd, rf, 94 特征）
  features.py     共享因果特征（DL 后续使用）
  transforms.py   Standardizer（fit on train，支持 2D/3D）
  splits.py       rolling-window folds + purge gap
  metrics.py      官方评分向量化 + proxy + R2/IC/Sharpe/回撤/换手/bootstrap
  allocation.py   NotebookAllocationPolicy（精确复现 notebook 的仓位逻辑）
  backtest.py     净值/回撤
  model.py        Batch / Predictor / DatasetBuilder 协议
  pipeline.py     FullPipeline / PipelineCV / PipelineEnsemble
  tracker.py      W&B tracker + NullTracker
  registry.py     run 目录、manifest、数据哈希
  ingest.py       每日 CSV 解析/校验/upsert/state/health
gdbt/
  config.py       GBDTConfig（notebook 参数）
  features.py     NotebookFeatureBuilder（逐行复现 create_features）
  model.py        SklearnEnsemble（enet + xgb + lgb）
  inference.py    live panel 最后 800 行重训 + 预测最新行
scripts/
  run_replay.py   本地回放（in-sample，复现 notebook）
  run_cv.py       6 折 walk-forward（out-of-sample）
  run_daily_retrain.py  逐日重训的 deployment-faithful 6 折 OOF
  compare_deployment_modes.py  Daily/Offline/Online/B&H/Constant/Vol-only 统一比较
  run_e14_stress.py  post-competition E14 特征处理压力测试
  run_phase4_dynamic_fixed.py  Phase 4C: dynamic builder + fixed model OOF
  run_feature_builder_attribution.py  Phase 4 2×2 归因、诊断和 bootstrap
  notebook_reference.py  直接 exec 提交 notebook 作为参照
  ingest_daily.py / predict_latest.py / daily_job.py  每日链路
deploy/           systemd service + timer + 安装说明
tests/            48 个测试：指标、数据、特征、泄漏、管线、notebook 对齐、每日 ingest、验证、评估、Phase 6 与 Phase 8
```

## 2. Notebook 复现（样本内，in-sample）

命令：

```bash
python scripts/run_replay.py --no-wandb
```

训练：`train.csv` 最后 800 行（date_id 8248..9047），ElasticNet + XGBoost + LightGBM，权重 0.45/0.55/0.45（未归一化，与 Notebook 一致）。

预测：`test.csv` 的 10 行（date_id 8980..8989），逐行调用。

| date_id | raw_pred | allocation | vol | multiplier |
|---|---|---|---|---|
| 8980 | -0.004492 | 0.000000 | 0.3686 | 400 |
| 8981 | -0.006351 | 0.000000 | 0.5783 | 400 |
| 8982 | 0.004414 | 1.499955 | 0.5312 | 400 |
| 8983 | 0.006584 | 1.874933 | 0.5567 | 400 |
| 8984 | -0.002304 | 0.468719 | 0.5025 | 400 |
| 8985 | 0.002744 | 1.617131 | 0.4173 | 400 |
| 8986 | 0.002082 | 1.581198 | 0.4422 | 400 |
| 8987 | 0.002878 | 1.830315 | 0.5014 | 400 |
| 8988 | 0.006765 | 1.957520 | 0.3868 | 400 |
| 8989 | -0.000491 | 0.489365 | 0.2668 | 400 |

9 个计分日（scored day）的官方评分：**13.690181**（vol_penalty=1.0，return_penalty=1.0，mean_allocation=1.2033，turnover=0.6052）。

与直接执行提交 Notebook 得到的逐行仓位（allocation）对比：**最大绝对误差 4.97e-08**（`tests/test_notebook_parity.py`）。

注意：该分数是**样本内（in-sample）**的。Notebook 用最后 800 行训练，而 `test.csv` 的 8980..8989 本来就在这 800 行里；本地回放不是样本外估计。XGBoost 在 800 行上的训练 R2 是 0.9956，明显过拟合，因此 13.69 不能当作真实水平。

## 3. 滚动前向交叉验证（walk-forward CV，样本外）

命令：

```bash
python scripts/run_cv.py --no-wandb
```

每个折（fold）：滚动 800 行训练窗口，1 天清除间隔（purge gap），整年测试。

| 折 | 评分 | Sharpe | IC | 方向准确率 | 平均仓位 |
|---|---|---|---|---|---|
| 2015 | 0.2264 | 0.2845 | 0.1112 | 0.5675 | 0.642 |
| 2017 | 0.3161 | 1.2179 | 0.1013 | 0.4303 | 0.041 |
| 2019 | 1.9791 | 1.9791 | 0.0610 | 0.5040 | 0.508 |
| 2021 | 2.3603 | 2.3603 | 0.1018 | 0.5079 | 0.649 |
| 2023 | 0.1663 | 0.4686 | 0.0209 | 0.4480 | 0.275 |
| 2025 | -0.0489 | -0.3572 | -0.0140 | 0.5108 | 0.883 |
| 平均 | **0.8332** | — | **0.0637** | **0.4947** | **0.495** |
| 最差 | **-0.0489** | — | — | — | — |

解读：

- 2019/2021 的评分很高（约 2.0），2015/2017/2023 在 0.2-0.3，2025（比赛期）略负；折间方差很大。
- 方向准确率平均 49.5%，说明模型的主要收益来自仓位缩放与波动率目标，而不是方向预测。
- 2017 折的平均仓位只有 0.04（几乎空仓），但评分仍有 0.316：低波动带来高 Sharpe，return_penalty=3.85 将其压低。2017 是低波动牛市，空仓会被收益惩罚（return penalty）拖累；这说明 Notebook 的 600/400 乘数与 1.45 权重在不同市场状态（regime）下仓位差异很大。
- 这是当前 GBDT 的诚实基线；后续深度学习（DL）应与本表比较，而不是与 13.69 比较。
- 2025 折的测试期是 2025-01 至 2025-12-03，训练窗口在 2024 年底之前结束，没有用到测试期数据。

### 3.1 在线与离线对比（同 6 折）

`run_cv.py --online` 在每个测试日先用当日之前已实现的 target 更新模型，再预测下一日；这是因果（causal）的在线学习，不是使用未来数据。

| 折 | 离线评分 | 在线评分 | 离线 IC | 在线 IC |
|---|---|---|---|---|
| 2015 | 0.2264 | 0.4600 | 0.1112 | 0.0505 |
| 2017 | 0.3161 | 0.5213 | 0.1013 | -0.0443 |
| 2019 | 1.9791 | 2.5878 | 0.0610 | 0.1039 |
| 2021 | 2.3603 | 1.2049 | 0.1018 | -0.0866 |
| 2023 | 0.1663 | 2.8512 | 0.0209 | 0.1553 |
| 2025 | -0.0489 | -0.1073 | -0.0140 | -0.1093 |
| 平均 | **0.8332** | **1.2530** | **0.0637** | **0.0116** |
| 最差 | **-0.0489** | **-0.1073** | — | — |

解读：

- 在线学习把平均评分从 0.83 提到 1.25，但 2021、2025 两折变差，且平均 IC 从 0.064 降到 0.012；收益主要来自仓位与波动率路径，而不是方向预测变准。
- 只有 6 折，折间方差很大；不能据此宣布在线模式更优。它的价值是作为第二条基线，后续 DL 应同时与离线和在线两张表比较。
- 在线分支在每个测试日重训 enet/xgb/lgb 三个模型，6 折约 40-50 分钟；部署时如果每天只做一次增量，应改为 `update()` 或固定窗口重训，避免每日全量重训的成本。

## 4. 关键实现说明

1. **Notebook 的在线分支是死代码**：`previous_lagged = test` 存的是 `create_features` 之后的帧，`lagged_market_forward_excess_returns` 已被 `select` 裁掉，所以 `if` 永远不成立。实际是“启动时训练一次”。复现配置使用 `online_update: false`；`SklearnEnsemble.update()` 与 `FullPipeline(online_update=True)` 已实现，供后续启用。
2. **Notebook 使用 `target`（market_forward_excess_returns）训练**；官方评分使用 `pos*(fwd-rf)`。两者相差约 3.1bp/日，复现保持 Notebook 口径；CV/DL 后续应使用 `target_metric` 或做截距校准。
3. **权重未归一化**：0.45+0.55+0.45=1.45，复现保持该设置；可通过 `normalize_weights: true` 切换。
4. **数据泄漏边界**：`NotebookFeatureBuilder` 保留 Notebook 的 `backward_fill`（训练窗口内），但测试行是逐行变换，不使用测试期的未来数据；`tests/test_no_leakage.py` 验证单行特征在截断面板下保持不变。
5. **W&B 未启动**：`common/tracker.py` 只在脚本调用 `init()` 时初始化；`--no-wandb` 使用 `NullTracker`。

## 5. W&B 接入（CoreWeave Forge）与运行结果

W&B 已并入 CoreWeave Forge。自 SDK 0.31.0（2026-10-16）起，默认 API 为 `https://api.forge.coreweave.com`，应用端（App）为 `https://forge.coreweave.com/wandb`，API 密钥（API key）在 `https://forge.coreweave.com/settings` 生成；旧版 `api.wandb.ai` 的密钥仍然兼容。本项目的 `.env` 已设置：

```
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_PROJECT=hull_tactical
WANDB_ENTITY=          # 默认解析为 quasarheart-personal
WANDB_MODE=online
```

账号校验：`wandb.Api().viewer` 返回 `quasarheart`（2189775585@qq.com），密钥有效。

网络说明：本机 WSL 直连 `api.forge.coreweave.com` / `api.wandb.ai` 返回 403 HTML（`via: 1.1 google`），但 `id.coreweave.com` 直连正常；通过机器上已有的 `127.0.0.1:7897` 代理可以正常认证。**没有修改任何 WSL/Windows 代理设置**；只在 wandb 命令前临时设置了 `HTTPS_PROXY` / `HTTP_PROXY`。如果不想使用代理，可以在 VPS 上运行同样的命令。

### 已完成的运行

| 模式 | 运行 | 指标 | URL |
|---|---|---|---|
| 回放离线（causal/OOS） | `gdbt_notebook_replay_offline_causal_ef4e16` | score 8.5243, mean alloc 0.5571, turnover 0.4606 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/h8x6ruw7 |
| 回放在线（causal/OOS） | `gdbt_notebook_replay_online_causal_ebf36d` | score 11.2707, mean alloc 0.7052, turnover 0.7311 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/rhpmhs66 |
| CV 离线（6 折） | `gdbt_notebook_cv_offline_ae2a75` | mean 0.8332, worst -0.0489, IC 0.0637 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/1x1kf8g5 |
| CV 在线（6 折） | `gdbt_notebook_cv_online_0746df` | finished：mean 1.2530, worst -0.1073, IC 0.0116 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/wjyw8imb |

说明：

- 回放的因果（causal）模式把训练窗口截到 8180..8979，测试 8980..8989，是 9 个计分日的短样本外对比；分数噪声很大，不能当作稳定结论。
- CV 在线已完成（`models/20261121_170851_gdbt_notebook_cv_online_0746df`）：mean 1.2530 / worst -0.1073 / IC 0.0116；2021 和 2025 两折变差，不能只看 mean。
- 之前有一次 300 秒超时的回放在线运行（`l9rsjr1j`）是连接测试中断留下的，不是最终结果。

## 6. 每日数据链路（已打通）

源站：`https://www.hulltactical.com/kaggle-dataset-download/` 302 跳转到 `https://www.hulltactical.com/wp-content/uploads/kaggle/YYYYMMDD_data.csv`。文件是全历史 96 列（空表头日期列 + `target` + 94 特征），缺失值为字符串 `NA`，最后约 3 行 target 未实现；行号就是比赛的 `date_id`。

实现：

```
common/ingest.py       resolve/download/validate/attach labels/upsert panel/state/health
scripts/ingest_daily.py  --check / --dry-run / --force
scripts/predict_latest.py 最后 800 行已实现 target 重训 + 预测最新行
scripts/daily_job.py    ingest + predict，一行命令给 cron/systemd
deploy/                systemd service + timer + 安装说明
```

当前实测（2026-11-21 拉取 2026-11-20 文件）：

- 9,291 行 × 96 列，9,288 个已实现 target，3 个待实现；
- 前 9,048 行日期与 `data/date_map.csv` 完全一致，重叠部分的 target/target_metric 与比赛 train 完全一致（max diff = 0.0）；
- live 243 行的 `forward_returns` / `risk_free_rate` 全为 null；
- `data/lake/panel_live.parquet` 100 列（6 个元数据 + 94 特征），3.3MB；
- 预测 `date_id=9290`（2026-11-20）：allocation **1.499955**，raw_pred 0.006903，vol 0.027538，V1 0.031746 < median 0.182540 → multiplier 600；
- `data/lake/predictions.parquet` 每个 data_date 一行（重跑替换），供 dashboard 使用；
- 重复运行数据摄取（ingest）会返回 `skipped`，不会重复下载或覆盖；只有 `--force` 才会重新下载。

数据质量告警（必须关注）：`E14` 从 2026-06-16 起连续缺失（111 行，含最新行）。Notebook 的 0.5 缺失阈值会保留它（800 行窗口缺失率 13.9%），模型使用中位数填充。`health.json` / stdout 会给出警告。Phase 3 的赛后（post-competition）压力测试显示：删除 E14 更差，`E14_MISSING` 的 XGBoost 重要性（importance）为 0 且不改变仓位路径，因此当前继续采用中位数填充，暂不加入缺失指示变量；等更长已实现窗口或实时 A/B 测试再决定。详见 `docs/deployment_eval.md`。

部署：见 `deploy/README.md`。在 VPS 上安装后先运行

```bash
python scripts/ingest_daily.py --check
python scripts/ingest_daily.py --dry-run
python scripts/daily_job.py
```

再启用 `hull-daily.timer`（每天 08:30 Asia/Shanghai，`Persistent=true`）。失败会以非零状态退出并保留上一版 panel/prediction；每次成功写入 `ingest_log.jsonl` 和 `health.json`。测试覆盖：`tests/test_ingest.py`（schema、target 尾部、特征不匹配、幂等 upsert、拒绝缩表、state/health），全套 `pytest -q` **48 passed**。

## 7. 忠实部署评估（Phase 3）

```bash
python scripts/run_daily_retrain.py
python scripts/compare_deployment_modes.py
python scripts/run_e14_stress.py --retrain-every 20 --output reports/deployment_eval/e14_rolling20
```

逐日重训严格使用 `date_id < t` 的最近 800 行，逐日重建特征构建器并重训 enet/xgb/lgb；六折折外（OOF）的 `train_end < date_id` 全部成立。结果：官方评分均值 0.871，低于在线模式 1.253、买入持有（Buy & Hold）1.498、恒定仓位（Constant 1.25）1.411、仅波动率（Vol-only）1.267；连续评分（continuous score）0.808 也低于在线模式与简单基准。更严格的部署口径没有复现在线模式的优势。

E14 中断与六折测试期重叠为 0；赛后代理窗口见 `docs/deployment_eval.md`。结论：暂不进入深度学习，先分解逐日模式与在线模式在特征中位数与路径上的差异。

## 8. Phase 4 特征与重训归因

```bash
python scripts/run_phase4_dynamic_fixed.py
python scripts/run_feature_builder_attribution.py
```

2×2 析因（factorial）：A 固定构建器 + 固定模型（Offline）、B 固定构建器 + 滚动重训（Online）、C 动态构建器 + 固定模型（new）、D 动态构建器 + 滚动重训（Daily Retrain）。所有 OOF 复制到 `reports/phase4_attribution/oof/` 并记录 SHA-256；未用 OOF 调参。结果：A 0.833、B 1.253、C 0.524、D 0.871；所有成对分块自助法（paired block bootstrap）对比的置信区间（CI）都跨 0。详见 `docs/feature_builder_attribution.md`。
