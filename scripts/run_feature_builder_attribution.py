"""Phase 4: 2x2 feature-builder / retraining attribution.

A = fixed feature builder + fixed model          (existing Offline OOF)
B = fixed feature builder + rolling model        (existing Online OOF)
C = dynamic feature builder + fixed model        (new Phase 4C OOF)
D = dynamic feature builder + rolling model      (existing Daily-Retrain OOF)

The script treats A/B/D OOFs as immutable references, copies them with hashes,
and compares all four configurations on the same six folds. It never tunes
against the OOF. Inference uses the shared paired moving-block bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common.config import (
    COL_DATE_ID,
    COL_FWD,
    COL_RF,
    COL_TARGET,
    COL_TARGET_METRIC,
    Config,
)
from common.data import date_ids_for_years, load_date_map, load_panel
from common.evaluation import (
    holm_adjust,
    paired_block_bootstrap,
    summarize_positions,
    summarize_turnover_adjusted,
)
from common.experiments import causal_vol_estimates, simulate_vol_target
from common.registry import build_manifest, utc_now, write_json
from common.splits import rolling_window_folds
from common.validation import pearson_ic, position_stats, quantile_returns, rank_ic
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)

CONFIGS = {
    "A_fixed_fixed": "A: fixed builder + fixed model",
    "B_fixed_rolling": "B: fixed builder + rolling model",
    "C_dynamic_fixed": "C: dynamic builder + fixed model",
    "D_dynamic_rolling": "D: dynamic builder + rolling model",
}
BASELINES = {
    "buy_hold": "Buy & Hold",
    "constant_1.25": "Constant 1.25",
    "vol_only": "Vol-only 1.0x",
    "existing_online": "Existing Online",
    "existing_daily": "Existing Daily",
}
ALL_LABELS = {**CONFIGS, **BASELINES}
FACTOR_LABELS = {
    "builder_effect_fixed_model": "Dynamic - fixed builder\n(fixed model)",
    "builder_effect_rolling_model": "Dynamic - fixed builder\n(rolling model)",
    "retraining_effect_fixed_builder": "Rolling - fixed model\n(fixed builder)",
    "retraining_effect_dynamic_builder": "Rolling - fixed model\n(dynamic builder)",
    "interaction_difference_in_differences": "Interaction\n(D-C) - (B-A)",
}
COLORS = {
    "A_fixed_fixed": "tab:orange",
    "B_fixed_rolling": "tab:green",
    "C_dynamic_fixed": "tab:purple",
    "D_dynamic_rolling": "tab:blue",
    "buy_hold": "black",
    "constant_1.25": "tab:brown",
    "vol_only": "tab:red",
    "existing_online": "tab:green",
    "existing_daily": "tab:blue",
}


def make_folds(config: Config) -> tuple[pl.DataFrame, list[Any]]:
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
    return panel, folds


def latest_non_smoke(models_dir: Path, pattern: str) -> Path:
    paths = [
        path
        for path in models_dir.glob(pattern)
        if "smoke" not in path.parent.name
    ]
    if not paths:
        raise FileNotFoundError(f"No non-smoke OOF matching {pattern}")
    return sorted(paths, key=lambda path: path.parent.name)[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy(source: Path, destination: Path) -> str:
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError(f"Refusing to overwrite immutable OOF: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(destination) != digest:
        raise RuntimeError(f"Copy hash mismatch: {destination}")
    return digest


def load_oof(path: Path, panel: pl.DataFrame, folds: list[Any]) -> dict[str, pl.DataFrame]:
    frame = pl.read_csv(path).sort(COL_DATE_ID)
    if COL_DATE_ID not in frame.columns:
        raise ValueError(f"{path} has no {COL_DATE_ID}")
    labels = panel.select([COL_DATE_ID, COL_TARGET, COL_TARGET_METRIC, COL_FWD, COL_RF])
    frame = frame.join(labels, on=COL_DATE_ID, how="left", suffix="_panel")
    if frame[COL_FWD].null_count() > 0 or frame[COL_RF].null_count() > 0:
        raise ValueError(f"{path}: missing fwd/rf after join")
    if int(frame[COL_DATE_ID].is_duplicated().sum()) > 0:
        raise ValueError(f"{path}: duplicate date_id")
    expected = {int(date_id) for fold in folds for date_id in fold.test_dates}
    actual = set(int(value) for value in frame[COL_DATE_ID].to_list())
    if actual != expected:
        raise ValueError(
            f"{path}: date set differs from folds; missing={len(expected - actual)} "
            f"extra={len(actual - expected)}"
        )
    output = {}
    for fold in folds:
        sub = frame.filter(pl.col(COL_DATE_ID).is_in(fold.test_dates)).sort(COL_DATE_ID)
        if sub.height != len(fold.test_dates):
            raise ValueError(f"{path}: fold {fold.name} has {sub.height} rows")
        output[fold.name] = sub
    return output


def base_positions(frames: dict[str, pl.DataFrame], folds: list[Any], name: str) -> dict[str, np.ndarray]:
    return {fold.name: frames[fold.name][name].to_numpy() for fold in folds}


def make_vol_only(
    daily: dict[str, pl.DataFrame],
    panel: pl.DataFrame,
    folds: list[Any],
    model_config: GBDTConfig,
) -> dict[str, np.ndarray]:
    out = {}
    for fold in folds:
        train_vol = causal_vol_estimates(
            panel, fold.train_dates, vol_window=model_config.vol_window
        )
        out[fold.name] = simulate_vol_target(
            daily[fold.name]["vol"],
            float(np.median(train_vol)),
            smoothing=model_config.smoothing,
            transaction_cost=model_config.transaction_cost,
        )
    return out


def fold_labels(frames: dict[str, pl.DataFrame], folds: list[Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {fold.name: frames[fold.name][COL_FWD].to_numpy() for fold in folds},
        {fold.name: frames[fold.name][COL_RF].to_numpy() for fold in folds},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/phase4.yaml")
    parser.add_argument("--offline-oof", default=None)
    parser.add_argument("--online-oof", default=None)
    parser.add_argument("--daily-oof", default=None)
    parser.add_argument("--c-oof", default=None)
    parser.add_argument("--output", default="reports/phase4_attribution")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    phase4 = config.section("phase4")
    seed = int(args.seed if args.seed is not None else phase4.get("seed", 42))
    block_length = int(
        args.block_length if args.block_length is not None else phase4.get("block_length", 20)
    )
    bootstrap_reps = int(
        args.bootstrap_reps
        if args.bootstrap_reps is not None
        else phase4.get("bootstrap_reps", 2000)
    )
    cost_bps = float(phase4.get("cost_bps", 10.0))
    output = Path(args.output)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    (output / "oof").mkdir(parents=True, exist_ok=True)

    panel, folds = make_folds(config)
    sources = {
        "A_fixed_fixed": Path(args.offline_oof)
        if args.offline_oof
        else latest_non_smoke(
            Path(config.paths.models_dir), "*_gdbt_notebook_cv_offline_*/oof_predictions.csv"
        ),
        "B_fixed_rolling": Path(args.online_oof)
        if args.online_oof
        else latest_non_smoke(
            Path(config.paths.models_dir), "*_gdbt_notebook_cv_online_*/oof_predictions.csv"
        ),
        "C_dynamic_fixed": Path(args.c_oof)
        if args.c_oof
        else latest_non_smoke(
            Path(config.paths.models_dir), "*_gdbt_phase4_dynamic_fixed_*/oof_predictions.csv"
        ),
        "D_dynamic_rolling": Path(args.daily_oof)
        if args.daily_oof
        else latest_non_smoke(
            Path(config.paths.models_dir), "*_gdbt_daily_retrain_*/oof_predictions.csv"
        ),
    }
    print("sources:")
    for name, path in sources.items():
        print(f"  {name}: {path}")
    frames = {name: load_oof(path, panel, folds) for name, path in sources.items()}
    oof_hashes = {
        name: safe_copy(path, output / "oof" / f"oof_{name}.csv")
        for name, path in sources.items()
    }

    fwd, rf = fold_labels(frames["D_dynamic_rolling"], folds)
    strategies: dict[str, dict[str, np.ndarray]] = {}
    for name in CONFIGS:
        strategies[name] = base_positions(frames[name], folds, "allocation")
    n_days = {fold.name: len(fold.test_dates) for fold in folds}
    strategies["buy_hold"] = {fold.name: np.ones(n_days[fold.name]) for fold in folds}
    strategies["constant_1.25"] = {
        fold.name: np.full(n_days[fold.name], 1.25) for fold in folds
    }
    strategies["vol_only"] = make_vol_only(
        frames["D_dynamic_rolling"], panel, folds, model_config
    )
    strategies["existing_online"] = strategies["B_fixed_rolling"]
    strategies["existing_daily"] = strategies["D_dynamic_rolling"]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for name, by_fold in strategies.items():
        if name in ("existing_online", "existing_daily"):
            continue
        for fold in folds:
            frame = frames["D_dynamic_rolling"][fold.name]
            report = summarize_positions(
                by_fold[fold.name],
                frame[COL_FWD].to_numpy(),
                frame[COL_RF].to_numpy(),
            )
            stats = position_stats(by_fold[fold.name])
            trade = summarize_turnover_adjusted(
                by_fold[fold.name],
                frame[COL_FWD].to_numpy(),
                frame[COL_RF].to_numpy(),
                cost_bps=cost_bps,
            )
            fold_rows.append(
                {
                    "strategy": name,
                    "fold": fold.name,
                    "score": report["score"],
                    "sharpe": report["sharpe"],
                    "cumulative_return": report["cumulative_return"],
                    "max_drawdown": report["max_drawdown"],
                    "turnover": report["turnover"],
                    "mean_allocation": report["mean_allocation"],
                    "zero_share": stats.get("zero_share", float("nan")),
                    "two_share": stats.get("two_share", float("nan")),
                    "net_sharpe_10bps": trade["net_sharpe"],
                    "net_cumulative_return_10bps": trade["net_cumulative_return"],
                    "n_days": report["n_days"],
                }
            )
            position_rows.append(
                {
                    "strategy": name,
                    "fold": fold.name,
                    **stats,
                    "clip_low": float(np.mean(by_fold[fold.name] <= 1e-9)),
                    "clip_high": float(np.mean(by_fold[fold.name] >= 2.0 - 1e-9)),
                }
            )

        positions = np.concatenate([by_fold[fold.name] for fold in folds])
        fwd_all = np.concatenate([fwd[fold.name] for fold in folds])
        rf_all = np.concatenate([rf[fold.name] for fold in folds])
        continuous = summarize_positions(positions, fwd_all, rf_all)
        trade_all = summarize_turnover_adjusted(positions, fwd_all, rf_all, cost_bps=cost_bps)
        rows = [row for row in fold_rows if row["strategy"] == name]
        stats_all = position_stats(positions)
        summary_rows.append(
            {
                "strategy": name,
                "label": ALL_LABELS[name],
                "score_mean": float(np.mean([row["score"] for row in rows])),
                "score_median": float(np.median([row["score"] for row in rows])),
                "score_best": float(np.max([row["score"] for row in rows])),
                "score_worst": float(np.min([row["score"] for row in rows])),
                "score_continuous": continuous["score"],
                "sharpe_mean": float(np.mean([row["sharpe"] for row in rows])),
                "sharpe_continuous": continuous["sharpe"],
                "cumulative_return_continuous": continuous["cumulative_return"],
                "max_drawdown_continuous": continuous["max_drawdown"],
                "turnover_continuous": continuous["turnover"],
                "mean_allocation_continuous": continuous["mean_allocation"],
                "zero_share": stats_all.get("zero_share", float("nan")),
                "two_share": stats_all.get("two_share", float("nan")),
                "net_sharpe_10bps": trade_all["net_sharpe"],
                "net_cumulative_return_10bps": trade_all["net_cumulative_return"],
                "n_days": continuous["n_days"],
            }
        )

    pl.DataFrame(fold_rows).write_csv(output / "fold_metrics.csv")
    pl.DataFrame(position_rows).write_csv(output / "allocation_stats.csv")
    pl.DataFrame(summary_rows).write_csv(output / "summary.csv")

    # ------------------------------------------------------------------
    # Prediction diagnostics
    # ------------------------------------------------------------------
    prediction_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    prediction_by_fold: dict[str, dict[str, pl.DataFrame]] = {
        name: frames[name] for name in CONFIGS
    }
    prediction_by_fold["existing_online"] = frames["B_fixed_rolling"]
    prediction_by_fold["existing_daily"] = frames["D_dynamic_rolling"]
    for name, by_fold in prediction_by_fold.items():
        for fold in folds:
            frame = by_fold[fold.name]
            pred = frame["raw_pred"].to_numpy()
            y = frame[COL_TARGET_METRIC].to_numpy()
            prediction_rows.append(_prediction_row(name, fold.name, pred, y))
            for item in quantile_returns(pred, y, n_quantiles=5):
                bin_rows.append(
                    {
                        "config": name,
                        "fold": fold.name,
                        "quantile": item["quantile"],
                        "count": item["count"],
                        "mean_target": item["mean_target"],
                    }
                )
        pred_all = np.concatenate([by_fold[fold.name]["raw_pred"].to_numpy() for fold in folds])
        target_all = np.concatenate(
            [by_fold[fold.name][COL_TARGET_METRIC].to_numpy() for fold in folds]
        )
        prediction_rows.append(_prediction_row(name, "all", pred_all, target_all))
        for item in quantile_returns(pred_all, target_all, n_quantiles=5):
            bin_rows.append(
                {
                    "config": name,
                    "fold": "all",
                    "quantile": item["quantile"],
                    "count": item["count"],
                    "mean_target": item["mean_target"],
                }
            )

    for name in CONFIGS:
        for reference in ("existing_online", "existing_daily"):
            for fold in folds:
                pred = prediction_by_fold[name][fold.name]["raw_pred"].to_numpy()
                ref = prediction_by_fold[reference][fold.name]["raw_pred"].to_numpy()
                alloc = prediction_by_fold[name][fold.name]["allocation"].to_numpy()
                ref_alloc = prediction_by_fold[reference][fold.name]["allocation"].to_numpy()
                correlation_rows.append(
                    {
                        "config": name,
                        "reference": reference,
                        "fold": fold.name,
                        "raw_pearson": _corr(pred, ref),
                        "raw_spearman": _corr(pred, ref, rank=True),
                        "allocation_pearson": _corr(alloc, ref_alloc),
                        "allocation_spearman": _corr(alloc, ref_alloc, rank=True),
                    }
                )
            pred = np.concatenate(
                [prediction_by_fold[name][fold.name]["raw_pred"].to_numpy() for fold in folds]
            )
            ref = np.concatenate(
                [prediction_by_fold[reference][fold.name]["raw_pred"].to_numpy() for fold in folds]
            )
            alloc = np.concatenate(
                [prediction_by_fold[name][fold.name]["allocation"].to_numpy() for fold in folds]
            )
            ref_alloc = np.concatenate(
                [prediction_by_fold[reference][fold.name]["allocation"].to_numpy() for fold in folds]
            )
            correlation_rows.append(
                {
                    "config": name,
                    "reference": reference,
                    "fold": "all",
                    "raw_pearson": _corr(pred, ref),
                    "raw_spearman": _corr(pred, ref, rank=True),
                    "allocation_pearson": _corr(alloc, ref_alloc),
                    "allocation_spearman": _corr(alloc, ref_alloc, rank=True),
                }
            )

    pl.DataFrame(prediction_rows).write_csv(output / "prediction_stats.csv")
    pl.DataFrame(bin_rows).write_csv(output / "prediction_bins.csv")
    pl.DataFrame(correlation_rows).write_csv(output / "prediction_correlations.csv")

    pairwise_rows: list[dict[str, Any]] = []
    for metric in ("raw_pred", "allocation"):
        for left in CONFIGS:
            for right in CONFIGS:
                a = np.concatenate(
                    [prediction_by_fold[left][fold.name][metric].to_numpy() for fold in folds]
                )
                b = np.concatenate(
                    [prediction_by_fold[right][fold.name][metric].to_numpy() for fold in folds]
                )
                pairwise_rows.append(
                    {
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "pearson": _corr(a, b),
                    }
                )
    pl.DataFrame(pairwise_rows).write_csv(output / "pairwise_correlations.csv")

    # ------------------------------------------------------------------
    # Feature statistics: fixed builders are reconstructed; dynamic stats
    # come from the Phase 4C run (C and D use the same dynamic builder).
    # ------------------------------------------------------------------
    feature_rows: list[dict[str, Any]] = []
    for fold in folds:
        initial = (
            panel.filter(pl.col(COL_DATE_ID) < fold.test_dates[0])
            .filter(pl.col(COL_TARGET).is_not_null())
            .sort(COL_DATE_ID)
            .tail(model_config.max_train_rows)
        )
        builder = NotebookFeatureBuilder(model_config).fit(panel, initial[COL_DATE_ID].to_list())
        for config_name in ("A_fixed_fixed", "B_fixed_rolling"):
            feature_rows.append(
                {
                    "config": config_name,
                    "fold": fold.name,
                    "date_id": None,
                    "n_features": len(builder.features),
                    "n_missing_vs_fixed": 0,
                    "n_extra_vs_fixed": 0,
                    "v1_median": float(builder.v1_median),
                    "median_shift_mean": 0.0,
                    "median_shift_max": 0.0,
                }
            )
    c_feature_path = sources["C_dynamic_fixed"].parent / "feature_stats.csv"
    if c_feature_path.exists():
        c_features = pl.read_csv(c_feature_path)
        c_features = c_features.with_columns(pl.lit("C_dynamic_fixed").alias("config"))
        feature_rows.extend(c_features.to_dicts())
        d_features = c_features.with_columns(pl.lit("D_dynamic_rolling").alias("config"))
        feature_rows.extend(d_features.to_dicts())
    pl.DataFrame(feature_rows).write_csv(output / "feature_stats.csv")
    c_medians = sources["C_dynamic_fixed"].parent / "feature_medians_initial.csv"
    if c_medians.exists():
        safe_copy(c_medians, output / "feature_medians_initial.csv")

    # ------------------------------------------------------------------
    # Runtime / computational cost
    # ------------------------------------------------------------------
    runtime_rows: list[dict[str, Any]] = []
    for name, path in sources.items():
        run_dir = path.parent
        manifest = run_dir / "manifest.json"
        created = None
        if manifest.exists():
            try:
                created = json.loads(manifest.read_text(encoding="utf-8")).get("created_at")
            except json.JSONDecodeError:
                created = None
        runtime_rows.append(
            {
                "config": name,
                "source_run": run_dir.name,
                "n_days": sum(n_days.values()),
                "model_fits": _model_fits(name),
                "model_updates": _model_updates(name),
                "builder_refits": _builder_refits(name),
                "observed_seconds": _run_wall_seconds(path, created),
                "source_created_at": created,
                "note": (
                    "wall time from manifest creation to OOF write; approximate "
                    "when the source run did not instrument timing"
                ),
            }
        )
    c_runtime = sources["C_dynamic_fixed"].parent / "runtime.csv"
    if c_runtime.exists():
        c_rows = pl.read_csv(c_runtime)
        runtime_rows = [
            row for row in runtime_rows if row["config"] != "C_dynamic_fixed"
        ]
        runtime_rows.extend(
            {
                "config": "C_dynamic_fixed",
                "source_run": sources["C_dynamic_fixed"].parent.name,
                "n_days": int(row["n_days"]),
                "model_fits": int(row["model_fits"]),
                "model_updates": 0,
                "builder_refits": int(row["builder_refits"]),
                "observed_seconds": float(row["seconds"]),
                "source_created_at": None,
                "note": "measured by scripts/run_phase4_dynamic_fixed.py",
            }
            for row in c_rows.to_dicts()
        )
    pl.DataFrame(runtime_rows).write_csv(output / "runtime_costs.csv")

    # ------------------------------------------------------------------
    # Paired block bootstrap against all requested comparators
    # ------------------------------------------------------------------
    bootstrap_rows: list[dict[str, Any]] = []
    for config_name in CONFIGS:
        for baseline_name in BASELINES:
            result = paired_block_bootstrap(
                strategies[config_name],
                strategies[baseline_name],
                fwd,
                rf,
                block_length=block_length,
                n_boot=bootstrap_reps,
                seed=seed,
            )
            bootstrap_rows.append(
                {
                    "config": config_name,
                    "baseline": baseline_name,
                    "actual_fold_mean_diff": result["actual_fold_mean_diff"],
                    "fold_ci_low": result["fold_ci_low"],
                    "fold_ci_high": result["fold_ci_high"],
                    "fold_p_two_sided": result["fold_p_two_sided"],
                    "fold_significant_5pct": result["fold_significant_5pct"],
                    "fold_diffs_positive": int(sum(value > 0 for value in result["fold_diffs"])),
                    "actual_concat_diff": result["actual_concat_diff"],
                    "concat_ci_low": result["concat_ci_low"],
                    "concat_ci_high": result["concat_ci_high"],
                    "concat_p_two_sided": result["concat_p_two_sided"],
                    "n_boot": result["n_boot"],
                    "block_length": result["block_length"],
                }
            )
    adjusted = holm_adjust([row["fold_p_two_sided"] for row in bootstrap_rows])
    for row, value in zip(bootstrap_rows, adjusted):
        row["fold_p_holm"] = value
        row["fold_significant_holm"] = bool(value < 0.05)
    pl.DataFrame(bootstrap_rows).write_csv(output / "bootstrap.csv")

    # Direct factorial contrasts, each with the same paired block bootstrap.
    factorial_rows: list[dict[str, Any]] = []
    factorial_pairs = [
        ("builder_effect_fixed_model", "C_dynamic_fixed", "A_fixed_fixed"),
        ("builder_effect_rolling_model", "D_dynamic_rolling", "B_fixed_rolling"),
        ("retraining_effect_fixed_builder", "B_fixed_rolling", "A_fixed_fixed"),
        ("retraining_effect_dynamic_builder", "D_dynamic_rolling", "C_dynamic_fixed"),
    ]
    for effect, left, right in factorial_pairs:
        result = paired_block_bootstrap(
            strategies[left],
            strategies[right],
            fwd,
            rf,
            block_length=block_length,
            n_boot=bootstrap_reps,
            seed=seed,
        )
        factorial_rows.append(
            {
                "effect": effect,
                "left": left,
                "right": right,
                "actual_fold_mean_diff": result["actual_fold_mean_diff"],
                "fold_ci_low": result["fold_ci_low"],
                "fold_ci_high": result["fold_ci_high"],
                "fold_p_two_sided": result["fold_p_two_sided"],
                "fold_significant_5pct": result["fold_significant_5pct"],
                "fold_diffs_positive": int(sum(value > 0 for value in result["fold_diffs"])),
                "n_boot": result["n_boot"],
                "block_length": result["block_length"],
            }
        )
    factorial_adjusted = holm_adjust([row["fold_p_two_sided"] for row in factorial_rows])
    for row, value in zip(factorial_rows, factorial_adjusted):
        row["fold_p_holm"] = value
    summary_by_name = {row["strategy"]: row for row in summary_rows}
    observed_interaction = (
        (summary_by_name["D_dynamic_rolling"]["score_mean"] - summary_by_name["C_dynamic_fixed"]["score_mean"])
        - (summary_by_name["B_fixed_rolling"]["score_mean"] - summary_by_name["A_fixed_fixed"]["score_mean"])
    )
    factorial_rows.append(
        {
            "effect": "interaction_difference_in_differences",
            "left": "D-C minus B-A",
            "right": "",
            "actual_fold_mean_diff": float(observed_interaction),
            "fold_ci_low": float("nan"),
            "fold_ci_high": float("nan"),
            "fold_p_two_sided": float("nan"),
            "fold_significant_5pct": False,
            "fold_diffs_positive": None,
            "n_boot": 0,
            "block_length": block_length,
            "fold_p_holm": float("nan"),
        }
    )
    pl.DataFrame(factorial_rows).write_csv(output / "factorial_effects.csv")

    factorial_means = {
        "fixed_builder_mean": (summary_by_name["A_fixed_fixed"]["score_mean"] + summary_by_name["B_fixed_rolling"]["score_mean"]) / 2.0,
        "dynamic_builder_mean": (summary_by_name["C_dynamic_fixed"]["score_mean"] + summary_by_name["D_dynamic_rolling"]["score_mean"]) / 2.0,
        "fixed_model_mean": (summary_by_name["A_fixed_fixed"]["score_mean"] + summary_by_name["C_dynamic_fixed"]["score_mean"]) / 2.0,
        "rolling_model_mean": (summary_by_name["B_fixed_rolling"]["score_mean"] + summary_by_name["D_dynamic_rolling"]["score_mean"]) / 2.0,
        "interaction": float(observed_interaction),
    }


    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_scores(summary_rows, output / "plots" / "01_score_comparison.png")
    _plot_fold_scores(fold_rows, folds, output / "plots" / "02_fold_scores.png")
    _plot_correlation(prediction_by_fold, folds, output / "plots" / "03_prediction_correlation.png")
    _plot_allocations(strategies, frames["D_dynamic_rolling"], folds, output / "plots" / "04_allocation_paths.png")
    _plot_prediction_distributions(prediction_by_fold, folds, output / "plots" / "05_prediction_distributions.png")
    _plot_bootstrap(bootstrap_rows, output / "plots" / "06_bootstrap_forest.png")
    _plot_factorial(factorial_rows, output / "plots" / "09_factorial_effects.png")
    _plot_feature_stats(output / "feature_stats.csv", output / "plots" / "07_feature_stats.png")
    _plot_turnover(summary_rows, output / "plots" / "08_turnover_exposure.png")

    manifest = build_manifest(
        run_id="phase4_feature_builder_attribution",
        run_name="phase4_feature_builder_attribution",
        config=config.raw,
        data_paths={
            "train": config.paths.train_csv,
            "test": config.paths.test_csv,
            "date_map": config.paths.date_map_csv,
        },
        extra={
            "kind": "phase4_attribution",
            "sources": {name: str(path) for name, path in sources.items()},
            "oof_sha256": oof_hashes,
            "folds": {
                fold.name: {
                    "train": [int(fold.train_dates[0]), int(fold.train_dates[-1])],
                    "test": [int(fold.test_dates[0]), int(fold.test_dates[-1])],
                    "n_test": len(fold.test_dates),
                }
                for fold in folds
            },
            "bootstrap": {
                "block_length": block_length,
                "n_boot": bootstrap_reps,
                "seed": seed,
                "primary": "fold mean of official-score differences",
                "secondary": "synthetic concatenated OOF; gaps omitted",
            },
            "turnover_cost_bps": cost_bps,
            "factorial_means": factorial_means,
        },
    )
    manifest["created_at"] = utc_now()
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "manifest": manifest,
            "summary": summary_rows,
            "bootstrap": bootstrap_rows,
            "factorial_effects": factorial_rows,
            "factorial_means": factorial_means,
        },
    )

    print(pl.DataFrame(summary_rows).select(
        [
            "strategy",
            "score_mean",
            "score_continuous",
            "sharpe_mean",
            "cumulative_return_continuous",
            "max_drawdown_continuous",
            "turnover_continuous",
            "net_sharpe_10bps",
        ]
    ))
    print(f"wrote {output}")


def _prediction_row(name: str, fold: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if len(pred) == 0:
        return {"config": name, "fold": fold, "n": 0}
    return {
        "config": name,
        "fold": fold,
        "n": int(len(pred)),
        "mean": float(np.mean(pred)),
        "std": float(np.std(pred, ddof=1)) if len(pred) > 1 else 0.0,
        "min": float(np.min(pred)),
        "q05": float(np.quantile(pred, 0.05)),
        "q25": float(np.quantile(pred, 0.25)),
        "median": float(np.median(pred)),
        "q75": float(np.quantile(pred, 0.75)),
        "q95": float(np.quantile(pred, 0.95)),
        "max": float(np.max(pred)),
        "positive_share": float(np.mean(pred > 0)),
        "pearson_ic": float(pearson_ic(pred, target)),
        "rank_ic": float(rank_ic(pred, target)),
        "mean_target": float(np.mean(target)),
    }


def _corr(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if rank:
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def _model_fits(name: str) -> int:
    return {
        "A_fixed_fixed": 6,
        "B_fixed_rolling": 6,
        "C_dynamic_fixed": 6,
        "D_dynamic_rolling": 1488,
    }[name]


def _model_updates(name: str) -> int:
    return {
        "A_fixed_fixed": 0,
        "B_fixed_rolling": 1482,
        "C_dynamic_fixed": 0,
        "D_dynamic_rolling": 0,
    }[name]


def _run_wall_seconds(oof_path: Path, created_at: str | None) -> float | None:
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return max(0.0, float(oof_path.stat().st_mtime - created.timestamp()))
    except (ValueError, OSError):
        return None


def _builder_refits(name: str) -> int:
    return {
        "A_fixed_fixed": 6,
        "B_fixed_rolling": 6,
        "C_dynamic_fixed": 1488,
        "D_dynamic_rolling": 1488,
    }[name]


def _plot_scores(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width / 2, [row["score_mean"] for row in rows], width, label="fold mean")
    ax.bar(x + width / 2, [row["score_continuous"] for row in rows], width, label="concatenated")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("official score")
    ax.set_title("Phase 4 configurations vs baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_fold_scores(rows: list[dict[str, Any]], folds: list[Any], path: Path) -> None:
    names = list(CONFIGS)
    fold_names = [fold.name for fold in folds]
    x = np.arange(len(fold_names))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for index, name in enumerate(names):
        values = [
            next(
                row["score"]
                for row in rows
                if row["strategy"] == name and row["fold"] == fold
            )
            for fold in fold_names
        ]
        ax.bar(x + (index - (len(names) - 1) / 2) * width, values, width, label=CONFIGS[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_names)
    ax.set_ylabel("official score")
    ax.set_title("Fold-level scores: builder x retraining")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_correlation(
    frames: dict[str, dict[str, pl.DataFrame]],
    folds: list[Any],
    path: Path,
) -> None:
    names = list(CONFIGS)
    matrix = np.full((len(names), len(names)), np.nan)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            a = np.concatenate([frames[left][fold.name]["raw_pred"].to_numpy() for fold in folds])
            b = np.concatenate([frames[right][fold.name]["raw_pred"].to_numpy() for fold in folds])
            matrix[i, j] = _corr(a, b)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Raw prediction correlation (OOF)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_allocations(
    strategies: dict[str, dict[str, np.ndarray]],
    frames: dict[str, pl.DataFrame],
    folds: list[Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=True)
    for ax, fold_name in zip(axes, ["2021", "2025"]):
        x = frames[fold_name]["date_id"].to_numpy()
        for name in CONFIGS:
            ax.plot(x, strategies[name][fold_name], label=CONFIGS[name], linewidth=0.9)
        ax.set_title(f"{fold_name} fold")
        ax.set_ylabel("allocation")
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("date_id")
    fig.suptitle("Allocation paths: builder x retraining")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_prediction_distributions(
    frames: dict[str, dict[str, pl.DataFrame]],
    folds: list[Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, CONFIGS):
        values = np.concatenate(
            [frames[name][fold.name]["raw_pred"].to_numpy() for fold in folds]
        )
        values = values[np.isfinite(values)]
        ax.hist(values, bins=60, color=COLORS[name], alpha=0.75)
        ax.set_title(CONFIGS[name])
        ax.set_ylabel("days")
    for ax in axes[-1]:
        ax.set_xlabel("raw prediction")
    fig.suptitle("Raw prediction distributions")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_bootstrap(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["baseline"] in ("existing_online", "existing_daily")
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(selected))
    values = np.asarray([row["actual_fold_mean_diff"] for row in selected])
    low = np.asarray([row["fold_ci_low"] for row in selected])
    high = np.asarray([row["fold_ci_high"] for row in selected])
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="o",
        capsize=3,
        color="tab:blue",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{CONFIGS[row['config']]} vs {BASELINES[row['baseline']]}" for row in selected]
    )
    ax.set_xlabel("fold-mean official score difference (95% paired block bootstrap CI)")
    ax.set_title("Phase 4 attribution: key paired differences")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_factorial(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row["n_boot"]]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(selected))
    values = np.asarray([row["actual_fold_mean_diff"] for row in selected])
    low = np.asarray([row["fold_ci_low"] for row in selected])
    high = np.asarray([row["fold_ci_high"] for row in selected])
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="o",
        capsize=3,
        color="tab:purple",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FACTOR_LABELS.get(row["effect"], row["effect"]) for row in selected])
    ax.set_xlabel("fold-mean official score difference (95% paired block bootstrap CI)")
    ax.set_title("Factorial contrasts")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_feature_stats(path: Path, output: Path) -> None:
    if not path.exists():
        return
    frame = pl.read_csv(path)
    if frame.height == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ("C_dynamic_fixed",):
        sub = frame.filter(pl.col("config") == name)
        if sub.height == 0:
            continue
        aggregated = (
            sub.group_by("date_id")
            .agg(
                pl.col("median_shift_mean").mean().alias("median_shift_mean"),
                pl.col("n_features").mean().alias("n_features"),
            )
            .sort("date_id")
        )
        ax.plot(
            aggregated["date_id"].to_numpy(),
            aggregated["median_shift_mean"].to_numpy(),
            label="dynamic builder (C and D)",
            linewidth=0.9,
        )
    ax.set_xlabel("date_id")
    ax.set_ylabel("mean |dynamic median - fold-start median|")
    ax.set_title("Feature-builder drift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_turnover(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(x, [row["turnover_continuous"] for row in rows], color="tab:blue", alpha=0.7)
    ax1.set_ylabel("turnover")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, [row["net_sharpe_10bps"] for row in rows], color="tab:red", marker="o")
    ax2.set_ylabel("net Sharpe at 10 bps")
    ax1.set_title("Turnover and turnover-adjusted performance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
