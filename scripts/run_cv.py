"""Walk-forward CV for the GBDT family (out-of-sample estimate).

Usage:
    python scripts/run_cv.py --no-wandb              # verify, no tracking
    python scripts/run_cv.py --offline               # 6-fold CV, no online updates
    python scripts/run_cv.py --online                # 6-fold CV, daily updates
    python scripts/run_cv.py --online --folds 2025   # one fold (timing/debug)

The folds use a rolling 800-day train window and a 1-day purge gap before each
calendar-year test block. The notebook's in-sample replay lives in
scripts/run_replay.py.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import polars as pl

from common.allocation import NotebookAllocationPolicy
from common.config import Config, load_env
from common.data import date_ids_for_years, load_date_map, load_panel
from common.pipeline import FullPipeline, PipelineCV
from common.registry import (
    build_manifest,
    create_run_dir,
    make_run_id,
    write_json,
    write_manifest,
)
from common.splits import rolling_window_folds
from common.tracker import make_tracker
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder
from gdbt.model import SklearnEnsemble


def pipeline_factory(model_config: GBDTConfig):
    def factory() -> FullPipeline:
        policy = NotebookAllocationPolicy(
            vol_window=model_config.vol_window,
            multiplier_low=model_config.signal_multiplier_low_vol,
            multiplier_high=model_config.signal_multiplier_high_vol,
            min_signal=model_config.min_signal,
            max_signal=model_config.max_signal,
            vol_scaling=model_config.vol_scaling,
            smoothing=model_config.smoothing,
            transaction_cost=model_config.transaction_cost,
        )
        return FullPipeline(
            predictor=SklearnEnsemble(model_config),
            builder=NotebookFeatureBuilder(model_config),
            policy=policy,
            online_update=model_config.online_update,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/cv.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--online", action="store_true", help="enable daily online updates")
    parser.add_argument("--offline", action="store_true", help="disable online updates")
    parser.add_argument("--folds", default=None, help="comma-separated fold names (default: all)")
    args = parser.parse_args()
    if args.online and args.offline:
        parser.error("--online and --offline are mutually exclusive")

    load_env()
    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    online = model_config.online_update
    if args.online:
        online = True
    if args.offline:
        online = False
    model_config = replace(model_config, online_update=online)
    mode = "online" if online else "offline"
    cv_config = config.section("cv")
    run_name = args.run_name or f"{config.run_name}_{mode}"

    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    date_map = load_date_map(config.paths.date_map_csv)
    panel_dates = set(panel["date_id"].to_list())
    test_years = [int(year) for year in cv_config.get("test_years", [])]
    by_year = date_ids_for_years(date_map, test_years)
    by_year = {
        year: [date_id for date_id in dates if date_id in panel_dates]
        for year, dates in by_year.items()
    }
    by_year = {year: dates for year, dates in by_year.items() if dates}
    folds = rolling_window_folds(
        panel["date_id"].to_list(),
        by_year,
        train_window=int(cv_config.get("train_window", 800)),
        gap=int(cv_config.get("gap", 1)),
    )
    if args.folds:
        keep = {name.strip() for name in args.folds.split(",") if name.strip()}
        folds = [fold for fold in folds if fold.name in keep]
        if not folds:
            parser.error(f"No folds matched --folds {args.folds}")

    run_config = deepcopy(config.raw)
    run_config.setdefault("model", {})["online_update"] = online
    run_config["cv"] = {
        **(run_config.get("cv") or {}),
        "folds_used": [fold.name for fold in folds],
        "online_update": online,
    }

    run_id = make_run_id(run_name)
    run_dir = create_run_dir(config.paths.models_dir, run_id)
    manifest = build_manifest(
        run_id=run_id,
        run_name=run_name,
        config=run_config,
        data_paths={
            "train": config.paths.train_csv,
            "test": config.paths.test_csv,
            "date_map": config.paths.date_map_csv,
        },
        extra={
            "kind": "gdbt_cv",
            "folds": [fold.name for fold in folds],
            "online_update": online,
        },
    )
    write_manifest(run_dir, manifest)

    tracker = make_tracker(
        replace(config.wandb, tags=[*config.wandb.tags, mode]),
        run_id,
        config=run_config,
        enabled=not args.no_wandb,
    )
    tracker.init()
    if tracker.url:
        print(f"W&B run: {tracker.url}")

    print("Folds:")
    for fold in folds:
        print(
            f"  {fold.name}: train {fold.train_dates[0]}..{fold.train_dates[-1]} "
            f"({len(fold.train_dates)}), test {fold.test_dates[0]}..{fold.test_dates[-1]} "
            f"({len(fold.test_dates)})"
        )

    cv = PipelineCV(pipeline_factory(model_config)).fit(panel, folds)

    for fold in cv.folds:
        metrics = fold.report
        tracker.log(
            {
                f"fold/{fold.name}/score": metrics["score"],
                f"fold/{fold.name}/sharpe": metrics["sharpe"],
                f"fold/{fold.name}/ic": metrics["ic"],
                f"fold/{fold.name}/direction_accuracy": metrics["direction_accuracy"],
                f"fold/{fold.name}/mean_allocation": float(fold.scored["allocation"].mean()),
            }
        )
        print(
            f"fold {fold.name}: score={metrics['score']:.4f} "
            f"sharpe={metrics['sharpe']:.4f} ic={metrics['ic']:.4f} "
            f"dir={metrics['direction_accuracy']:.4f}"
        )

    tracker.log(
        {
            "cv/score_mean": cv.summary["score_mean"],
            "cv/score_worst": cv.summary["score_worst"],
            "cv/score_best": cv.summary["score_best"],
            "cv/ic_mean": cv.summary["ic_mean"],
            "cv/direction_mean": cv.summary["direction_mean"],
            "cv/n_folds": len(cv.folds),
        }
    )
    tracker.log_table("cv/oof_predictions", cv.oof)
    for fold in cv.folds:
        tracker.log_table(f"cv/fold_{fold.name}", fold.scored)
    tracker.set_summary(
        {
            key: value
            for key, value in cv.summary.items()
            if isinstance(value, (int, float, bool, str))
        }
    )

    oof_path = run_dir / "oof_predictions.csv"
    cv.oof.write_csv(oof_path)
    summary_path = run_dir / "cv_summary.json"
    write_json(summary_path, cv.summary)
    tracker.log_artifact(summary_path, name=f"{run_id}-cv-summary", artifact_type="report")
    tracker.log_artifact(oof_path, name=f"{run_id}-oof", artifact_type="dataset")
    tracker.log_artifact(
        run_dir / "manifest.json", name=f"{run_id}-manifest", artifact_type="metadata"
    )
    tracker.finish()

    print("\nSummary:")
    print(json.dumps(cv.summary, indent=2))
    print(f"artifacts: {run_dir}")


if __name__ == "__main__":
    main()
