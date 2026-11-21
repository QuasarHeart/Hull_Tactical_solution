"""Predict the latest live panel row with the GBDT model.

This is the second half of the daily job; it assumes the live panel has already
been ingested by `scripts/ingest_daily.py` (or pass `--ingest`).
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from common.config import COL_DATE, COL_DATE_ID, Config, load_env
from common.ingest import (
    ingest_latest,
    load_panel_live,
    load_state,
    save_state,
    upsert_prediction_history,
    utc_now,
    write_health,
)
from common.registry import write_json
from common.tracker import make_tracker
from gdbt.config import GBDTConfig
from gdbt.inference import predict_latest


def run(config: Config, log_wandb: bool = False) -> dict:
    panel = load_panel_live(config.paths.panel_path)
    model_config = GBDTConfig.from_dict(config.section("model"))
    state = load_state(config.paths.state_path)

    latest_id = int(panel[COL_DATE_ID].max())
    latest_date_iso = panel.filter(pl.col(COL_DATE_ID) == latest_id)[COL_DATE][0].isoformat()
    last_prediction = state.get("last_prediction") or {}
    if last_prediction.get("data_date") == latest_date_iso:
        # Rerun for the same data date: reuse the seed used by the first run so
        # the smoothing is reproducible instead of compounding its own output.
        initial_allocation = float(last_prediction.get("initial_allocation", 0.0) or 0.0)
    else:
        initial_allocation = float(state.get("last_allocation", 0.0) or 0.0)

    result = predict_latest(
        panel,
        model_config,
        initial_allocation=initial_allocation,
        latest_date_id=latest_id,
    )
    predictions = result["predictions"]
    row = predictions.row(0, named=True)

    report = {
        "data_date": result["latest_date"].isoformat(),
        "date_id": int(result["latest_id"]),
        "allocation": float(row["allocation"]),
        "raw_pred": float(row["raw_pred"]),
        "vol": float(row["vol"]),
        "v1": float(row["v1"]),
        "v1_median": float(row["v1_median"]),
        "multiplier": float(row["multiplier"]),
        "train_start": int(result["train_start"]),
        "train_end": int(result["train_end"]),
        "n_train": int(result["n_train"]),
        "n_features": len(result["features"]),
        "initial_allocation": initial_allocation,
        "created_at": utc_now(),
    }

    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.paths.reports_dir / "latest_prediction.json", report)
    history = upsert_prediction_history(config.paths.predictions_path, report)

    state["last_prediction"] = report
    state["last_allocation"] = report["allocation"]
    state["last_prediction_at"] = report["created_at"]
    save_state(config.paths.state_path, state)
    write_health(
        config.paths.health_path,
        panel,
        state,
        extra={
            "prediction": report,
            "prediction_history_rows": history.height,
        },
    )

    if log_wandb:
        tracker = make_tracker(
            config.wandb,
            f"daily_{report['data_date']}",
            config=config.raw,
            enabled=True,
        )
        tracker.init()
        if tracker.url:
            print(f"W&B run: {tracker.url}")
        tracker.log(
            {
                "daily/allocation": report["allocation"],
                "daily/raw_pred": report["raw_pred"],
                "daily/vol": report["vol"],
                "daily/v1": report["v1"],
                "daily/multiplier": report["multiplier"],
                "daily/train_rows": report["n_train"],
            }
        )
        tracker.log_table("daily/prediction", predictions)
        tracker.set_summary(report)
        tracker.finish()

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/notebook.yaml")
    parser.add_argument("--ingest", action="store_true", help="run the daily ingest first")
    parser.add_argument("--force", action="store_true", help="force re-download during ingest")
    parser.add_argument("--wandb", action="store_true", help="log the daily prediction to W&B")
    args = parser.parse_args()

    load_env()
    config = Config.load(args.config)
    config.paths.ensure()

    if args.ingest:
        result = ingest_latest(config, force=args.force)
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))

    report = run(config, log_wandb=args.wandb)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
