# Daily deployment (Linux VPS)

The daily job is a single command:

```bash
/opt/hull/venv/bin/python scripts/daily_job.py
```

It resolves the latest CSV at
`https://www.hulltactical.com/kaggle-dataset-download/`, downloads it to
`data/raw/hull/YYYYMMDD_data.csv`, validates it, replaces
`data/lake/panel_live.parquet`, then trains the GBDT on the last 800 realized
rows and writes `reports/latest_prediction.json` plus `data/lake/health.json`.

## 1. Layout

```
/opt/hull/
├── app/       # this repository
├── venv/      # dedicated Python 3.12 environment
├── data/
│   ├── raw/hull/
│   └── lake/
├── models/
├── reports/
└── logs/
```

The repository resolves its paths relative to the repo root, so `app/data`,
`app/models` etc. are used if you do not override them in `configs/base.yaml`.

## 2. Install

```bash
sudo useradd -r -m -d /opt/hull hull
sudo mkdir -p /opt/hull/{app,venv,models,logs}
# copy/clone the repo into /opt/hull/app

sudo -u hull python3.12 -m venv /opt/hull/venv
sudo -u hull /opt/hull/venv/bin/pip install -e "/opt/hull/app[dev]"

sudo -u hull cp /opt/hull/app/.env.example /opt/hull/app/.env
sudo -u hull chmod 600 /opt/hull/app/.env
# edit /opt/hull/app/.env and set WANDB_API_KEY if you want W&B tracking
```

The ingest requires `data/train.csv` (feature schema and exact overlap labels).
`data/date_map.csv` is used for the 9,048-date alignment check; copy it with
`app/` if present (if absent, the check is skipped). `data/raw/` and
`data/lake/` are created on demand.

## 3. Smoke test

```bash
cd /opt/hull/app
/opt/hull/venv/bin/python scripts/ingest_daily.py --check
/opt/hull/venv/bin/python scripts/ingest_daily.py --dry-run
/opt/hull/venv/bin/python scripts/daily_job.py
```

Expected `health.json`:

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

## 4. systemd timer

```bash
sudo cp deploy/hull-daily.service /etc/systemd/system/
sudo cp deploy/hull-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hull-daily.timer
systemctl list-timers hull-daily.timer
journalctl -u hull-daily.service -f
```

The timer runs at 08:30 and 12:30 Asia/Shanghai (after the US close; the second
run catches a late dataset update and is idempotent) and catches up after
downtime with `Persistent=true`. If your systemd does not support the timezone
suffix, edit the timer to `OnCalendar=*-*-* 00:30:00 UTC` and
`OnCalendar=*-*-* 04:30:00 UTC`.

## 5. Failure behaviour

- A failed download/validation exits non-zero; systemd records the failure and
  the previous panel/prediction remains in place.
- `data/lake/ingest_log.jsonl` keeps every attempt (status, URL, SHA-256, rows).
- `--force` re-downloads the same date; a normal daily run skips it.
- Add `--wandb` to `ExecStart` if you want each daily prediction in W&B.
- If the VPS reaches the internet through a proxy, put `HTTPS_PROXY` /
  `HTTP_PROXY` in `/opt/hull/app/.env`; `python-dotenv` loads them for the job.
