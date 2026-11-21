"""Deployment-faithful daily-retrain GBDT CV.

For every test day t, this script retrains ElasticNet + XGBoost + LightGBM on
the last 800 rows with target strictly before t, then predicts t and applies the
same allocation policy as the notebook. No future row is used for training.

Usage:
    python scripts/run_daily_retrain.py --folds 2025 --limit-days 5  # smoke
    python scripts/run_daily_retrain.py                             # all folds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from common.allocation import NotebookAllocationPolicy
from common.config import (
    COL_DATE_ID,
    COL_FWD,
    COL_RF,
    COL_TARGET,
    COL_TARGET_METRIC,
    Config,
)
from common.data import date_ids_for_years, load_date_map, load_panel
from common.evaluation import summarize_positions
from common.registry import (
    build_manifest,
    create_run_dir,
    make_run_id,
    write_json,
    write_manifest,
)
from common.splits import rolling_window_folds
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder
from gdbt.model import SklearnEnsemble


def make_folds(config: Config, only: set[str] | None = None) -> tuple[pl.DataFrame, list[Any]]:
    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    date_map = load_date_map(config.paths.date_map_csv)
    cv = config.section("cv")
    test_years = [int(year) for year in cv.get("test_years", [])]
    by_year = date_ids_for_years(date_map, test_years)
    panel_dates = set(panel[COL_DATE_ID].to_list())
    by_year = {
        year: [date_id for date_id in dates if date_id in panel_dates]
        for year, dates in by_year.items()
    }
    by_year = {year: dates for year, dates in by_year.items() if dates}
    folds = rolling_window_folds(
        panel[COL_DATE_ID].to_list(),
        by_year,
        train_window=int(cv.get("train_window", 800)),
        gap=int(cv.get("gap", 1)),
    )
    if only:
        folds = [fold for fold in folds if fold.name in only]
        if not folds:
            raise ValueError(f"No folds matched {sorted(only)}")
    return panel, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/cv.yaml")
    parser.add_argument("--folds", default=None, help="comma-separated fold names")
    parser.add_argument("--limit-days", type=int, default=None, help="limit test days per fold")
    parser.add_argument("--run-name", default="gdbt_daily_retrain")
    args = parser.parse_args()

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    only = {v.strip() for v in args.folds.split(",")} if args.folds else None
    panel, folds = make_folds(config, only=only)
    panel = panel.sort(COL_DATE_ID)

    run_id = make_run_id(args.run_name)
    run_dir = create_run_dir(config.paths.models_dir, run_id)
    manifest = build_manifest(
        run_id=run_id,
        run_name=args.run_name,
        config=config.raw,
        data_paths={
            "train": config.paths.train_csv,
            "test": config.paths.test_csv,
            "date_map": config.paths.date_map_csv,
        },
        extra={
            "kind": "gdbt_daily_retrain",
            "folds": [fold.name for fold in folds],
            "limit_days": args.limit_days,
            "protocol": "retrain last 800 rows with date_id < t for every test day",
        },
    )
    write_manifest(run_dir, manifest)
    print(f"run dir: {run_dir}")

    all_rows: list[dict[str, Any]] = []
    fold_reports: dict[str, Any] = {}

    for fold in folds:
        test_dates = list(fold.test_dates)
        if args.limit_days is not None:
            test_dates = test_dates[: int(args.limit_days)]
        first_test = test_dates[0]
        initial = (
            panel.filter(pl.col(COL_DATE_ID) < first_test)
            .filter(pl.col(COL_TARGET).is_not_null())
            .sort(COL_DATE_ID)
            .tail(model_config.max_train_rows)
        )
        initial_dates = initial[COL_DATE_ID].to_list()
        initial_builder = NotebookFeatureBuilder(model_config)
        initial_builder.fit(panel, initial_dates)
        initial_batch = initial_builder.build(panel, initial_dates, with_target=True)
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
        policy.fit(initial_batch)

        fold_rows: list[dict[str, Any]] = []
        start_time = time.time()
        for day_index, date_id in enumerate(test_dates):
            train = (
                panel.filter(pl.col(COL_DATE_ID) < date_id)
                .filter(pl.col(COL_TARGET).is_not_null())
                .sort(COL_DATE_ID)
                .tail(model_config.max_train_rows)
            )
            train_dates = train[COL_DATE_ID].to_list()
            if len(train_dates) < model_config.max_train_rows:
                raise RuntimeError(
                    f"Fold {fold.name}: only {len(train_dates)} training rows before {date_id}"
                )

            builder = NotebookFeatureBuilder(model_config)
            builder.fit(panel, train_dates)
            train_batch = builder.build(panel, train_dates, with_target=True)
            model = SklearnEnsemble(model_config)
            model.fit(train_batch)
            test_batch = builder.build(panel, [date_id], with_target=False)
            components = model.predict_components(test_batch)
            raw_pred = float(components["ensemble"][0])
            context = builder.context(panel, date_id)
            allocation = policy.transform(raw_pred, context)

            row = panel.filter(pl.col(COL_DATE_ID) == date_id).row(0, named=True)
            target = float(row[COL_TARGET])
            fold_rows.append(
                {
                    "date_id": int(date_id),
                    "raw_pred": raw_pred,
                    "allocation": float(allocation),
                    "raw_pred_enet": float(components["enet"][0]),
                    "raw_pred_xgb": float(components["xgb"][0]),
                    "raw_pred_lgb": float(components["lgb"][0]),
                    "raw_pred_ensemble": raw_pred,
                    "vol": float(context.vol),
                    "v1": float(context.v1) if context.v1 is not None else float("nan"),
                    "v1_median": float(context.v1_median) if context.v1_median is not None else float("nan"),
                    "multiplier": float(context.multiplier),
                    "target": target,
                    "target_metric": float(row[COL_TARGET_METRIC]),
                    "forward_returns": float(row[COL_FWD]),
                    "risk_free_rate": float(row[COL_RF]),
                    "train_start": int(train_dates[0]),
                    "train_end": int(train_dates[-1]),
                    "n_train": len(train_dates),
                }
            )
            policy.update(target)

            if (day_index + 1) % 20 == 0 or day_index == 0:
                elapsed = time.time() - start_time
                print(
                    f"fold {fold.name}: {day_index + 1}/{len(test_dates)} "
                    f"date_id={date_id} raw={raw_pred:.6f} alloc={allocation:.4f} "
                    f"elapsed={elapsed:.1f}s"
                )

        fold_frame = pl.DataFrame(fold_rows).sort(COL_DATE_ID)
        if (fold_frame["train_end"] >= fold_frame[COL_DATE_ID]).any():
            raise RuntimeError(f"Fold {fold.name}: training window includes test date")
        fold_frame.write_csv(run_dir / f"fold_{fold.name}.csv")
        positions = fold_frame["allocation"].to_numpy()
        report = summarize_positions(
            positions,
            fold_frame[COL_FWD].to_numpy(),
            fold_frame[COL_RF].to_numpy(),
        )
        report["fold"] = fold.name
        report["train_start"] = int(fold_frame["train_start"][0])
        report["train_end"] = int(fold_frame["train_end"][-1])
        report["n_test_days"] = fold_frame.height
        fold_reports[fold.name] = {k: v for k, v in report.items() if k != "fold"}
        all_rows.extend(fold_frame.to_dicts())
        print(
            f"fold {fold.name}: score={report['score']:.4f} sharpe={report['sharpe']:.4f} "
            f"alloc={report['mean_allocation']:.3f} ret={report['cumulative_return']:.3f} "
            f"mdd={report['max_drawdown']:.3f}"
        )

    oof = pl.DataFrame(all_rows).sort(COL_DATE_ID)
    oof.write_csv(run_dir / "oof_predictions.csv")
    continuous = summarize_positions(
        oof["allocation"].to_numpy(),
        oof[COL_FWD].to_numpy(),
        oof[COL_RF].to_numpy(),
    )
    scores = [fold_reports[name]["score"] for name in fold_reports]
    summary = {
        "mode": "daily_retrain",
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_best": float(np.max(scores)),
        "score_worst": float(np.min(scores)),
        "sharpe_mean": float(np.mean([fold_reports[name]["sharpe"] for name in fold_reports])),
        "score_continuous": float(continuous["score"]),
        "sharpe_continuous": float(continuous["sharpe"]),
        "cumulative_return_continuous": float(continuous["cumulative_return"]),
        "max_drawdown_continuous": float(continuous["max_drawdown"]),
        "turnover_continuous": float(continuous["turnover"]),
        "mean_allocation_continuous": float(continuous["mean_allocation"]),
        "n_days": int(continuous["n_days"]),
        "folds": fold_reports,
    }
    write_json(run_dir / "cv_summary.json", summary)
    write_json(run_dir / "report.json", {"summary": summary, "manifest": manifest})
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {run_dir}")


if __name__ == "__main__":
    main()
