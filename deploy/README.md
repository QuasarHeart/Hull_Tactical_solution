# 每日部署

每日任务只需要一条命令：

```bash
/opt/hull/venv/bin/python scripts/daily_job.py
```

任务会从 `https://www.hulltactical.com/kaggle-dataset-download/` 解析最新 CSV，下载到 `data/raw/hull/YYYYMMDD_data.csv`，校验后替换 `data/lake/panel_live.parquet`，然后用最近 800 行已实现数据训练 GBDT，并写入 `reports/latest_prediction.json` 与 `data/lake/health.json`。

## 1. 目录布局

```text
/opt/hull/
├── app/       # 本仓库
├── venv/      # 独立的 Python 3.12 环境
├── data/
│   ├── raw/hull/
│   └── lake/
├── models/
├── reports/
└── logs/
```

仓库按仓库根目录解析路径，因此如果没有在 `configs/base.yaml` 中覆盖，实际使用的是 `app/data`、`app/models` 等目录。

## 2. 安装

```bash
sudo useradd -r -m -d /opt/hull hull
sudo mkdir -p /opt/hull/{app,venv,models,logs}
# 将仓库复制或克隆到 /opt/hull/app

sudo -u hull python3.12 -m venv /opt/hull/venv
sudo -u hull /opt/hull/venv/bin/pip install -e "/opt/hull/app[dev]"

sudo -u hull cp /opt/hull/app/.env.example /opt/hull/app/.env
sudo -u hull chmod 600 /opt/hull/app/.env
# 编辑 /opt/hull/app/.env；如果需要 W&B 跟踪，设置 WANDB_API_KEY
```

数据接入需要 `data/train.csv`（用于特征字段约定和精确重叠校验）。`data/date_map.csv` 用于 9,048 个日期的对齐检查；如果存在，请随 `app/` 一起复制；如果不存在，该检查会被跳过。`data/raw/` 和 `data/lake/` 会按需创建。

## 3. 冒烟测试

```bash
cd /opt/hull/app
/opt/hull/venv/bin/python scripts/ingest_daily.py --check
/opt/hull/venv/bin/python scripts/ingest_daily.py --dry-run
/opt/hull/venv/bin/python scripts/daily_job.py
```

预期的 `health.json`：

```json
{
  "last_data_date": "2026-11-20",
  "lag_days": 1,
  "rows": 9291,
  "target_rows": 9288,
  "pending_targets": 3,
  "last_ingest_at": "...",
  "prediction_history_rows": 1,
  "last_prediction": {"allocation": 1.23, "...": "..."}
}
```

## 4. systemd 定时器

```bash
sudo cp deploy/hull-daily.service /etc/systemd/system/
sudo cp deploy/hull-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hull-daily.timer
systemctl list-timers hull-daily.timer
journalctl -u hull-daily.service -f
```

定时器在亚洲/上海时间 08:30 和 12:30 运行（美国收盘后；第二次运行用于捕获延迟更新的数据集，且是幂等的），并通过 `Persistent=true` 在停机后补跑。如果你的 systemd 不支持时区后缀，请把定时器改为 `OnCalendar=*-*-* 00:30:00 UTC` 和 `OnCalendar=*-*-* 04:30:00 UTC`。

## 5. 失败行为

- 下载或校验失败会以非零状态退出；systemd 记录失败，之前的实盘日度数据表和预测结果保持不变。
- `data/lake/ingest_log.jsonl` 保留每次尝试（状态、URL、SHA-256、行数）。
- `--force` 会重新下载同一天的数据；正常每日任务会跳过已下载的日期。
- 如果希望每天预测都进入 W&B，在 `ExecStart` 中加入 `--wandb`。
- 如果 VPS 通过代理访问互联网，把 `HTTPS_PROXY` / `HTTP_PROXY` 写入 `/opt/hull/app/.env`；`python-dotenv` 会为任务加载它们。
