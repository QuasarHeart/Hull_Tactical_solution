# 数据清单与来源审计

日期：2026-11-22
机器可读文件：`data/manifest.json`
生成脚本：`scripts/build_data_manifest.py`

## 1. 目的

这不是“数据已经完全没有问题”的声明，而是一个版本审计边界：

- 哪些文件进入了 Git；
- 每个文件的 SHA-256、行数、列数、日期范围和目标缺失数；
- 哪些文件是实盘数据或生成产物，已被 `.gitignore` 排除；
- 原始远程 CSV 的 URL、哈希和数据接入状态；
- 复现实验前需要准备什么。

`data/manifest.json` 是当前快照；数据更新后重新运行生成脚本即可。如果哈希与数据清单不一致，不要静默继续做回测。

## 2. 受版本控制的比赛数据

| 文件 | 字节数 | 行 × 列 | 日期 / `date_id` | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `data/train.csv` | 8,465,196 | 9,048 × 98 | `date_id` 0–9047 | `8c1aba83e1bf34f91b234b7979f1fc7c397186b33a8d08bdd3634c08eb3810ca` |
| `data/test.csv` | 11,569 | 10 × 99 | `date_id` 8980–8989 | `b643378a7db4f868ccc23146cdd6bfa94e0f1926d670c509115713f953376b49` |
| `data/date_map.csv` | 156,851 | 9,291 × 2 | 1990-01-02 – 2026-11-20 | `3e61265e6e794be96489d1c27c57dcf9011f2b28492737c5a88d26217199038f` |

`train.csv` 有 94 个特征列，加上 `date_id` 和三个收益率字段；`test.csv` 增加 `is_scored` 与滞后收益字段。完整列名在 `data/manifest.json` 中。

## 3. 生成数据和本地数据（不进入 Git）

| 文件 | 字节数 | 行 × 列 | 说明 | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `data/lake/panel_live.parquet` | 3,440,367 | 9,291 × 100 | 实盘数据面板，3 个待实现目标 | `0a3fb6d2130eb98e2b8ea5e53a9361fade6985ec7a4ec68185946c454f51d2b8` |
| `data/raw/hull/20261120_data.csv` | 8,926,839 | 9,291 × 96 | 原始 Hull CSV | `b66e50d7ef0c032b0cca39e08f9ec8a356a633950328261466075f8a18049a6a` |

这两项被 `data/raw/` 与 `data/lake/` 的忽略规则排除；不提交是为了避免把每日更新的大文件、模型缓存和 W&B 运行产物带进仓库。它们仍然在数据清单中记录哈希，便于本机审计。

原始来源：

```text
https://www.hulltactical.com/wp-content/uploads/kaggle/20261120_data.csv
```

`data/lake/ingest_state.json` 记录 `last_data_date=2026-11-20`、`last_ingested_at=2026-11-21T09:16:57+00:00`、`rows=9291`，并报告 E14 的最近缺失率为 44.4%。这是数据质量警告，不是被静默填平的异常。

## 4. 快速验证

```bash
cd /home/yys/Hull_Tactical_solution

# 重新生成数据清单（会更新 generated_at）
/home/yys/miniconda3/envs/dlwpt/bin/python scripts/build_data_manifest.py

# 手工核对三个受版本控制的文件
sha256sum data/train.csv data/test.csv data/date_map.csv

# 确认被忽略的产物不会进入 Git
git check-ignore -v data/lake/panel_live.parquet data/raw/hull/20261120_data.csv
```

## 5. 边界

- `date_map.csv` 覆盖到 2026-11-20；实盘数据面板中的 3 个待实现目标不能被当作已实现标签。
- `panel_live.parquet` 使用 `target_metric` 作为实盘代理（proxy）的目标；官方评分需要真实的 `forward_returns` 和 `risk_free_rate`，当前数据源没有提供。
- 任何新的数据文件或日期扩展，都必须重新生成数据清单并重新跑测试；不要在数据清单未更新的情况下声称实验结果可复现。
