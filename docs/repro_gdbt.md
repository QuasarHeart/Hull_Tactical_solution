# GBDT 本地复现报告

日期：2026-11-21  
环境：`/home/yys/miniconda3/envs/dlwpt/bin/python`（Python 3.12，xgboost 3.3.0，lightgbm 4.6.0，polars 1.44.2，wandb 0.31.2）  
状态：本地复现完成，W&B 已接入；GBDT notebook、CV offline/online 均已跑完，每日数据链路已打通（2026-11-21）。

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
  notebook_reference.py  直接 exec 提交 notebook 作为参照
  ingest_daily.py / predict_latest.py / daily_job.py  每日链路
deploy/           systemd service + timer + 安装说明
tests/            18 个测试：指标、数据、特征、泄漏、管线、notebook 对齐、每日 ingest
```

## 2. Notebook 复现（in-sample）

命令：

```bash
python scripts/run_replay.py --no-wandb
```

训练：`train.csv` 最后 800 行（date_id 8248..9047），ElasticNet + XGBoost + LightGBM，权重 0.45/0.55/0.45（未归一化，和 notebook 一致）。  
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

9 个 scored 日的官方评分：**13.690181**（vol_penalty=1.0，return_penalty=1.0，mean_allocation=1.2033，turnover=0.6052）。

与直接执行提交 notebook 的逐行 allocations 对比：**最大绝对误差 4.97e-08**（`tests/test_notebook_parity.py`）。

注意：这个分数是 **in-sample** 的。notebook 用最后 800 行训练，而 `test.csv` 的 8980..8989 本来就在这 800 行里；本地回放不是样本外估计。XGBoost 在 800 行上的训练 R2 是 0.9956，明显过拟合，所以 13.69 不能当作真实水平。

## 3. Walk-forward CV（out-of-sample）

命令：

```bash
python scripts/run_cv.py --no-wandb
```

每个 fold：滚动 800 行训练窗口，1 天 purge gap，整年测试。

| fold | score | sharpe | IC | 方向准确率 | 平均仓位 |
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

- 2019/2021 的分数很高（约 2.0），2015/2017/2023 在 0.2-0.3，2025（比赛期）略负；fold 间方差很大。
- 方向准确率平均 49.5%，说明模型的主要收益来自仓位缩放/波动率目标，而不是方向预测。
- 2017 fold 的平均仓位只有 0.04（几乎空仓），但 score 仍有 0.316：低波动带来高 Sharpe，return_penalty=3.85 把它压下来。2017 是低波动牛市，空仓会被 return penalty 惩罚；这说明 notebook 的 600/400 乘数 + 1.45 权重在不同 regime 下仓位差异很大。
- 这是当前 GBDT 的诚实基线；后续 DL 应该和这张表比，而不是和 13.69 比。
- 2025 fold 的测试期是 2025-01 至 2025-12-03，训练窗口在 2024 年底之前结束，没有用到测试期数据。

### 3.1 Online vs offline（同 6 折）

`run_cv.py --online` 每个测试日先用当日之前已实现的 target 更新模型，再预测下一日；
这是 causal 的在线学习，不是用未来数据。

| fold | offline score | online score | offline IC | online IC |
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

- 在线学习把平均 score 从 0.83 提到 1.25，但 2021、2025 两折变差，且平均 IC 从 0.064 降到 0.012；
  收益主要来自仓位/波动率路径，而不是方向预测变准。
- 只有 6 折，fold 间方差很大；不能据此宣布 online 更好。它的价值是作为第二条基线，
  后续 DL 应同时和 offline/online 两张表比。
- 在线分支每个测试日重训 enet/xgb/lgb 三个模型，6 折约 40-50 分钟；部署时如果每天只做一次增量，
  应改成 `update()` 或固定窗口重训，避免每日全量重训的成本。

## 4. 关键实现说明

1. **Notebook 的 online 分支是死代码**：`previous_lagged = test` 存的是 `create_features` 之后的帧，`lagged_market_forward_excess_returns` 已被 `select` 裁掉，所以 `if` 永远不成立。实际是"启动时训练一次"。复现配置用 `online_update: false`；`SklearnEnsemble.update()` 和 `FullPipeline(online_update=True)` 已实现，供后续打开。
2. **notebook 用 `target`（market_forward_excess_returns）训练**；官方评分用 `pos*(fwd-rf)`。两者差约 3.1bp/日，复现保持 notebook 口径；CV/DL 后续应使用 `target_metric` 或做截距校准。
3. **权重未归一化**：0.45+0.55+0.45=1.45，复现保持；`normalize_weights: true` 可切换。
4. **数据泄漏边界**：`NotebookFeatureBuilder` 保留 notebook 的 `backward_fill`（训练窗口内），但测试行是逐行 transform，不使用测试期未来数据；`tests/test_no_leakage.py` 验证单行特征在截断面板下不变。
5. **W&B 未启动**：`common/tracker.py` 只在脚本调用 `init()` 时初始化；`--no-wandb` 使用 `NullTracker`。

## 5. W&B 接入（CoreWeave Forge）与运行结果

W&B 已并入 CoreWeave Forge。SDK 0.31.0（2026-10-16）起默认 API 为
`https://api.forge.coreweave.com`，App 为 `https://forge.coreweave.com/wandb`，
API key 在 `https://forge.coreweave.com/settings` 生成；旧 `api.wandb.ai` 的 key
仍然兼容。本项目的 `.env` 已设置：

```
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_PROJECT=hull_tactical
WANDB_ENTITY=          # 默认解析为 quasarheart-personal
WANDB_MODE=online
```

账号校验：`wandb.Api().viewer` 返回 `quasarheart`（2189775585@qq.com），key 有效。

网络说明：本机 WSL 直连 `api.forge.coreweave.com` / `api.wandb.ai` 返回 403 HTML
（`via: 1.1 google`），但 `id.coreweave.com` 直连正常；通过机器上已有的
`127.0.0.1:7897` 代理可以正常认证。**没有修改任何 WSL/Windows 代理设置**；只在
wandb 命令前临时设置了 `HTTPS_PROXY` / `HTTP_PROXY`。如果不想使用代理，可以在
VPS 上运行同样的命令。

### 已完成的 runs

| 模式 | run | 指标 | URL |
|---|---|---|---|
| replay offline（causal/OOS） | `gdbt_notebook_replay_offline_causal_ef4e16` | score 8.5243, mean alloc 0.5571, turnover 0.4606 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/h8x6ruw7 |
| replay online（causal/OOS） | `gdbt_notebook_replay_online_causal_ebf36d` | score 11.2707, mean alloc 0.7052, turnover 0.7311 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/rhpmhs66 |
| CV offline（6 折） | `gdbt_notebook_cv_offline_ae2a75` | mean 0.8332, worst -0.0489, IC 0.0637 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/1x1kf8g5 |
| CV online（6 折） | `gdbt_notebook_cv_online_0746df` | finished：mean 1.2530, worst -0.1073, IC 0.0116 | https://forge.coreweave.com/wandb/quasarheart-personal/hull_tactical/runs/wjyw8imb |

说明：

- replay 的 causal 模式把训练窗口截到 8180..8979，测试 8980..8989，是 9 个 scored 日的短样本外对比；分数噪声很大，不能当作稳定结论。
- CV online 已完成（`models/20261121_170851_gdbt_notebook_cv_online_0746df`）：mean 1.2530 / worst -0.1073 / IC 0.0116；2021 和 2025 两折变差，不能只看 mean。
- 之前有一次 300 秒超时的 replay online run（`l9rsjr1j`）是连接测试中断留下的，不是最终结果。

## 6. 每日数据链路（已打通）

源站：`https://www.hulltactical.com/kaggle-dataset-download/` 302 到
`https://www.hulltactical.com/wp-content/uploads/kaggle/YYYYMMDD_data.csv`。
文件是全历史 96 列（空表头日期列 + `target` + 94 特征），缺失值为字符串 `NA`，
最后约 3 行 target 未实现；行号就是比赛的 `date_id`。

实现：

```
common/ingest.py       resolve/download/validate/attach labels/upsert panel/state/health
scripts/ingest_daily.py  --check / --dry-run / --force
scripts/predict_latest.py 最后 800 行已实现 target 重训 + 预测最新行
scripts/daily_job.py    ingest + predict，一行命令给 cron/systemd
deploy/                systemd service + timer + 安装说明
```

当前实测（2026-11-21 拉 2026-11-20 文件）：

- 9,291 行 × 96 列，9,288 个已实现 target，3 个 pending；
- 前 9,048 行日期与 `data/date_map.csv` 完全一致，overlap 的 target/target_metric
  与比赛 train 完全一致（max diff = 0.0）；
- live 243 行的 `forward_returns` / `risk_free_rate` 全为 null；
- `data/lake/panel_live.parquet` 100 列（6 个元数据 + 94 特征），3.3MB；
- 预测 `date_id=9290`（2026-11-20）：allocation **1.499955**，raw_pred 0.006903，
  vol 0.027538，V1 0.031746 < median 0.182540 → multiplier 600；
- `data/lake/predictions.parquet` 每个 data_date 一行（重跑替换），供 dashboard 使用；
- 重复运行 ingest 会 `skipped`，不会重复下载/覆盖；`--force` 才重下。

数据质量告警（必须关注）：`E14` 从 2026-06-16 起连续缺失（111 行，含最新行）。
notebook 的 0.5 缺失阈值会保留它（800 行窗口缺失率 13.9%），模型用中位数填充。
`health.json` / stdout 会给出警告；如果 E14 长期不恢复，应考虑在特征筛选里显式丢弃
或在 DL 版本中把它作为 missing indicator，而不是继续中位数填充。

部署：见 `deploy/README.md`。VPS 上安装后先跑

```bash
python scripts/ingest_daily.py --check
python scripts/ingest_daily.py --dry-run
python scripts/daily_job.py
```

再启用 `hull-daily.timer`（每天 08:30 Asia/Shanghai，`Persistent=true`）。失败会
非零退出并保留上一版 panel/prediction；每次成功写 `ingest_log.jsonl` 和 `health.json`。
测试覆盖：`tests/test_ingest.py`（schema、target 尾部、特征不匹配、幂等 upsert、拒绝缩表、
state/health），全套 `pytest -q` 18 passed。
