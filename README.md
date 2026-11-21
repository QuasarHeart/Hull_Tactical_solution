# Hull Tactical research pipeline

Shared data/metric/allocation layer plus two model families:

```
common/   data, schema, features, transforms, splits, metrics, allocation,
          backtest, pipeline orchestration, tracker, registry
gdbt/     tabular model family (ElasticNet + XGBoost + LightGBM)
dl/       sequence model family (added after the GBDT baseline is validated)
scripts/  thin entry points (replay, CV, later daily ingest/dashboard)
tests/    metric parity, notebook parity, leakage and smoke tests
```

The current milestone reproduces the submitted notebook's GBDT solution and
logs the run to Weights & Biases.

## Setup

Python 3.12. A dedicated environment is recommended (do not reuse the Kaggle CLI
environment):

```bash
conda create -n hull python=3.12 -y
conda activate hull
pip install -e ".[dev]"
```

The project dependencies are declared in `pyproject.toml` (polars, pandas,
numpy, scikit-learn, xgboost, lightgbm, torch, wandb, pyarrow, requests,
pytest).

## Data

`data/train.csv` and `data/test.csv` are the competition files. `data/date_map.csv`
maps `date_id` to calendar dates; it is generated from the daily Hull CSV and is
used by the walk-forward CV.

## Daily data (deployment)

The live source is the full-history CSV at
`https://www.hulltactical.com/kaggle-dataset-download/`, which 302-redirects to
`.../wp-content/uploads/kaggle/YYYYMMDD_data.csv`. The source has 96 columns:
an empty-name date column, `target`, and the same 94 features as `train.csv`.
The last few target values are pending (`NA`).

```bash
# resolve the current dated URL only (no body download)
python scripts/ingest_daily.py --check

# download + validate, but do not write the panel
python scripts/ingest_daily.py --dry-run

# download, validate, atomically replace data/lake/panel_live.parquet
python scripts/ingest_daily.py

# ingest + train the last 800 realized rows + predict the latest row
python scripts/daily_job.py
```

What the ingest guarantees:

- 96 columns, exactly the 94 training feature names, unique and sorted dates;
- the first 9,048 dates match `data/date_map.csv` exactly;
- the competition overlap (`date_id <= 9047`) uses the exact
  `forward_returns - risk_free_rate` labels, while live rows have null
  `forward_returns` / `risk_free_rate`;
- only the tail of the target column may be null (pending labels);
- a SHA-256 of the raw file, a row count, and any data-quality warnings are
  written to `data/lake/ingest_state.json`, `ingest_log.jsonl`, and
  `health.json`;
- reruns are idempotent: the same source date is skipped unless `--force`.

`scripts/predict_latest.py` trains on the last 800 **realized** rows and writes
`reports/latest_prediction.json`, one row per data date in
`data/lake/predictions.parquet`, plus `last_allocation` in the ingest state, so
the next day's smoothing starts from the previous allocation. Rerunning for the
same data date reuses the original seed and is reproducible.

Smoke test on 2026-11-21 against the 2026-11-20 file: 9,291 rows, 9,288 realized
targets, 3 pending, 243 live rows; `date_id=9290`, allocation `1.499955`.

Data-quality caveat in the current source: `E14` is missing from 2026-06-16
onward (111 rows, including the latest row). The GBDT keeps it because its
800-row missing rate is 13.9% (below the notebook's 0.5 filter) and
median-fills it. The ingest reports this in `health.json` and on stdout; watch
it before trusting live allocations.

The deployment unit, timer and install steps are in `deploy/README.md`.

## Reproduce the notebook (in-sample replay)

```bash
python scripts/run_replay.py --no-wandb
```

This trains on the last 800 rows of `train.csv`, predicts the 10 rows of
`test.csv` one by one, applies the notebook's allocation policy, and scores the
9 `is_scored` rows. The replay is **in-sample** by construction: the test rows
are historical copies of train rows. Use the CV for an honest estimate.

The pipeline's allocations match the executed notebook code to <1e-6 (see
`tests/test_notebook_parity.py`).

## Walk-forward CV (out-of-sample)

```bash
python scripts/run_cv.py --no-wandb
```

Six folds (2015, 2017, 2019, 2021, 2023, 2025), each with a rolling 800-day
training window, a 1-day purge gap and a calendar-year test block. Reports
per-fold score, Sharpe, IC, direction accuracy and mean allocation.

## Weights & Biases

W&B is now part of CoreWeave Forge. With SDK >= 0.31 the default API endpoint
is `https://api.forge.coreweave.com` and the app is at
`https://forge.coreweave.com/wandb`. Create API keys at
`https://forge.coreweave.com/settings`; existing `api.wandb.ai` keys keep
working. Set `WANDB_BASE_URL` in `.env` only for a dedicated/self-managed W&B
server.

Tracking is opt-in at the entry point and is not initialised on import.

1. Copy `.env.example` to `.env`.
2. Put your key in `WANDB_API_KEY=...` (the file is gitignored; never commit it).
3. Run without `--no-wandb`:

```bash
python scripts/run_replay.py
python scripts/run_cv.py
```

If `WANDB_API_KEY` is missing, the tracker raises a clear error and prints the
`--no-wandb` escape hatch. `WANDB_MODE=offline` works without a key.

Logged artifacts: run config, train diagnostics, per-fold metrics, prediction
tables, model joblib, manifest with git SHA and data hashes.

## Notes

- The old `common/transformers.py`, `common/setup_env.py` and
  `gdbt/data_processor.py` are kept for reference only. The new pipeline uses
  `common/transforms.py`, `common/config.load_env`, `common/tracker.py` and
  `gdbt/features.py`.
- `market_forward_excess_returns` (the competition target) is not exactly
  `forward_returns - risk_free_rate`; the stable offset is ~0.0003112/day. The
  panel keeps both `target` and `target_metric = fwd - rf`; the notebook replay
  uses the competition target for parity, the CV/allocator can use `target_metric`.
- No `kaggle submit` is ever run by this repository; submissions remain manual.
- The DL family will implement the same `Predictor` + `DatasetBuilder` contracts
  and reuse the same CV, metrics and allocation policy.
