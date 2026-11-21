"""Reproduce the submitted GBDT notebook locally and (optionally) log to W&B.

Usage:
    python scripts/run_replay.py --no-wandb                 # verify, no tracking
    python scripts/run_replay.py --causal --offline         # OOS replay, no updates
    python scripts/run_replay.py --causal --online          # OOS replay, daily updates
    python scripts/run_replay.py                            # notebook parity, in-sample

By default the training window is the notebook's last 800 rows, which include
the replay dates 8980..8989 (in-sample). `--causal` instead trains on the 800
rows immediately before the first test date, which is required for a meaningful
online-vs-offline comparison.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from common.allocation import NotebookAllocationPolicy
from common.config import Config, load_env
from common.data import attach_labels, load_competition_test, load_panel
from common.metrics import ic_spearman, official_score, proxy_score, r2_weighted, turnover
from common.pipeline import FullPipeline
from common.registry import (
    build_manifest,
    create_run_dir,
    make_run_id,
    write_json,
    write_manifest,
)
from common.tracker import make_tracker
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder
from gdbt.model import SklearnEnsemble


def build_pipeline(model_config: GBDTConfig) -> FullPipeline:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/notebook.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--online", action="store_true", help="enable daily online updates")
    parser.add_argument("--offline", action="store_true", help="disable online updates")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="train on the 800 rows before the first test date (out-of-sample replay)",
    )
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
    run_name = args.run_name or f"{config.run_name}_{mode}" + ("_causal" if args.causal else "")

    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    test = load_competition_test(config.paths.test_csv)
    all_dates = panel["date_id"].to_list()
    test_dates = test["date_id"].to_list()
    if args.causal:
        first_test_idx = all_dates.index(test_dates[0])
        train_end = first_test_idx
        train_start = max(0, train_end - model_config.max_train_rows)
        train_dates = all_dates[train_start:train_end]
    else:
        train_dates = all_dates[-model_config.max_train_rows :]
    in_sample = bool(train_dates[-1] >= test_dates[0])
    if online and not args.causal:
        print(
            "WARNING: the notebook's last-800 window contains the replay dates; "
            "online updates would re-add rows already in training. Use --causal."
        )

    run_config = deepcopy(config.raw)
    run_config.setdefault("model", {})["online_update"] = online
    run_config["replay"] = {
        **(run_config.get("replay") or {}),
        "causal": bool(args.causal),
        "in_sample": in_sample,
        "train_dates": [int(train_dates[0]), int(train_dates[-1])],
        "test_dates": [int(test_dates[0]), int(test_dates[-1])],
    }

    run_id = make_run_id(run_name)
    run_dir = create_run_dir(config.paths.models_dir, run_id)
    manifest = build_manifest(
        run_id=run_id,
        run_name=run_name,
        config=run_config,
        data_paths={"train": config.paths.train_csv, "test": config.paths.test_csv},
        extra={
            "kind": "notebook_replay",
            "in_sample": in_sample,
            "online_update": online,
            "causal": bool(args.causal),
        },
    )
    write_manifest(run_dir, manifest)

    tracker = make_tracker(
        replace(
            config.wandb,
            tags=[*config.wandb.tags, mode, "causal" if args.causal else "notebook"],
        ),
        run_id,
        config=run_config,
        enabled=not args.no_wandb,
    )
    tracker.init()
    if tracker.url:
        print(f"W&B run: {tracker.url}")

    pipeline = build_pipeline(model_config)
    pipeline.fit(panel, train_dates)
    predictions = pipeline.predict_dates(panel, test_dates)

    scored = attach_labels(panel, predictions).join(
        test.select(["date_id", "is_scored"]), on="date_id", how="left"
    )
    scored_only = scored.filter(pl.col("is_scored"))
    report = official_score(
        scored_only["allocation"], scored_only["forward_returns"], scored_only["risk_free_rate"]
    )
    proxy = proxy_score(scored_only["allocation"], scored_only["target_metric"])
    mean_allocation = float(scored_only["allocation"].mean())
    turnover_value = turnover(scored_only["allocation"])

    train_batch = pipeline.train_batch
    train_metrics: dict[str, dict[str, float]] = {}
    if train_batch is not None and train_batch.y is not None:
        components_fn = getattr(pipeline.predictor, "predict_components", None)
        if components_fn is None:
            components = {"ensemble": pipeline.predictor.predict(train_batch)}
        else:
            components = components_fn(train_batch)
        for name, values in components.items():
            train_metrics[name] = {
                "r2": r2_weighted(train_batch.y, values),
                "ic": ic_spearman(train_batch.y, values),
            }
        tracker.log(
            {
                "train/in_sample": int(in_sample),
                "train/rows": len(train_batch),
                "train/features": len(train_batch.features),
                "train/v1_median": float(getattr(pipeline.builder, "v1_median", float("nan"))),
            }
        )
        for name, values in train_metrics.items():
            tracker.log({f"train/r2_{name}": values["r2"], f"train/ic_{name}": values["ic"]})

    metrics = {
        "replay/in_sample": int(in_sample),
        "replay/online_update": int(online),
        "replay/score": report.score,
        "replay/sharpe": report.sharpe,
        "replay/strategy_vol": report.strategy_vol,
        "replay/market_vol": report.market_vol,
        "replay/vol_penalty": report.vol_penalty,
        "replay/return_penalty": report.return_penalty,
        "replay/strategy_mean_excess": report.strategy_mean_excess,
        "replay/market_mean_excess": report.market_mean_excess,
        "replay/proxy_score": proxy.score,
        "replay/mean_allocation": mean_allocation,
        "replay/turnover": turnover_value,
        "replay/n_scored_days": report.n_days,
    }
    tracker.log(metrics)
    tracker.log_table("replay/predictions", scored)
    tracker.set_summary(
        {
            "score": report.score,
            "proxy_score": proxy.score,
            "in_sample": in_sample,
            "online_update": online,
            "train_rows": len(train_batch) if train_batch is not None else 0,
            "features": len(pipeline.builder.features),
        }
    )

    model_path = run_dir / "model.joblib"
    pipeline.predictor.save(model_path)
    predictions_path = run_dir / "predictions.csv"
    scored.write_csv(predictions_path)
    report_path = run_dir / "report.json"
    write_json(
        report_path,
        {
            "run_id": run_id,
            "in_sample": in_sample,
            "online_update": online,
            "causal": bool(args.causal),
            "metrics": metrics,
            "train_metrics": train_metrics,
            "features": list(pipeline.builder.features),
            "train_dates": [int(train_dates[0]), int(train_dates[-1])],
            "test_dates": [int(test_dates[0]), int(test_dates[-1])],
        },
    )
    tracker.log_artifact(model_path, name=f"{run_id}-model", artifact_type="model")
    tracker.log_artifact(report_path, name=f"{run_id}-report", artifact_type="report")
    features_path = run_dir / "features.json"
    write_json(
        features_path,
        {
            "features": list(pipeline.builder.features),
            "raw_feature_columns": list(
                getattr(pipeline.builder, "raw_feature_columns", pipeline.builder.features)
            ),
            "n_features": len(pipeline.builder.features),
        },
    )
    tracker.log_artifact(features_path, name=f"{run_id}-features", artifact_type="dataset")
    tracker.log_artifact(
        run_dir / "manifest.json", name=f"{run_id}-manifest", artifact_type="metadata"
    )
    tracker.finish()

    print(f"\nrun_id: {run_id}")
    print(scored.select(
        ["date_id", "raw_pred", "allocation", "vol", "v1", "multiplier", "target_metric", "is_scored"]
    ))
    print(
        f"\nscore={report.score:.6f} sharpe={report.sharpe:.6f} "
        f"vol_penalty={report.vol_penalty:.4f} return_penalty={report.return_penalty:.4f} "
        f"mean_allocation={mean_allocation:.4f} turnover={turnover_value:.4f}"
    )
    print(f"train metrics: {json.dumps(train_metrics, indent=2)}")
    print(f"artifacts: {run_dir}")


if __name__ == "__main__":
    main()
