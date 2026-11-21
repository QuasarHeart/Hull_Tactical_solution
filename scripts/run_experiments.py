"""Run allocation baselines and prediction ablations on the saved CV OOF.

Experiments
-----------
buy_hold
    Position = 1.0 every test day (S&P 500 total return).
constant_<x>
    Position = x every test day; 0.0 is the cash control.
vol_only_<scale>
    Position target = scale * median(training causal vol) / current vol.
    No return forecast is used.
prediction_ablation_train_mean
    Replace the model prediction with the fold's training-window mean target,
    then keep the same multiplier, vol scaling, smoothing and transaction cost.
prediction_ablation_zero
    Replace the prediction with zero (cash control for the smoothing machinery).
model
    The saved CV OOF allocation path (offline or online).
prediction_shuffle
    Permute raw_pred within each test fold, rerun the same sizing policy, and
    repeat B times. The p-value is the fraction of shuffled mean scores >= the
    actual model mean score.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --shuffles 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common.config import COL_DATE_ID, COL_FWD, COL_RF, COL_TARGET, Config
from common.data import date_ids_for_years, load_date_map, load_panel
from common.experiments import (
    causal_vol_estimates,
    score_positions,
    simulate_positions,
    simulate_vol_target,
)
from common.registry import git_sha, utc_now, write_json
from common.splits import rolling_window_folds
from gdbt.config import GBDTConfig

CONSTANTS = [0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0]
VOL_SCALES = [0.5, 1.0, 1.5]
SHUFFLE_SEED = 42

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

KEY_SPECS = [
    ("buy_hold", "position_1.0", "shared", "Buy & Hold 1.0"),
    ("constant", "constant_0.5", "shared", "Constant 0.5"),
    ("constant", "constant_1.5", "shared", "Constant 1.5"),
    ("model", "notebook_gbdt", "offline", "Model offline"),
    ("model", "notebook_gbdt", "online", "Model online"),
    ("vol_only", "scale_1.0", "offline", "Vol-only 1.0x"),
    ("prediction_ablation", "train_mean", "offline", "Ablation const mean"),
]


def latest_oof(models_dir: str | Path, pattern: str) -> Path:
    paths = sorted(Path(models_dir).glob(pattern), key=lambda p: p.parent.name)
    if not paths:
        raise FileNotFoundError(f"No OOF files matched {Path(models_dir) / pattern}")
    return paths[-1]


def load_oof(path: str | Path) -> pl.DataFrame:
    frame = pl.read_csv(path).sort(COL_DATE_ID)
    required = {COL_DATE_ID, "raw_pred", "allocation", "vol", "multiplier"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def make_folds(config: Config) -> tuple[pl.DataFrame, list[Any]]:
    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    date_map = load_date_map(config.paths.date_map_csv)
    cv_config = config.section("cv")
    test_years = [int(year) for year in cv_config.get("test_years", [])]
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
        train_window=int(cv_config.get("train_window", 800)),
        gap=int(cv_config.get("gap", 1)),
    )
    return panel, folds


def prepare_fold_data(
    panel: pl.DataFrame,
    oof: pl.DataFrame,
    folds: list[Any],
    model_config: GBDTConfig,
) -> list[dict[str, Any]]:
    labels = panel.select([COL_DATE_ID, COL_TARGET, COL_FWD, COL_RF])
    joined = oof.join(labels, on=COL_DATE_ID, how="left").sort(COL_DATE_ID)
    if any(joined[column].null_count() for column in (COL_FWD, COL_RF, COL_TARGET)):
        raise ValueError("OOF rows could not be joined to forward_returns/risk_free_rate")

    prepared: list[dict[str, Any]] = []
    for fold in folds:
        frame = joined.filter(pl.col(COL_DATE_ID).is_in(fold.test_dates)).sort(COL_DATE_ID)
        if frame.height != len(fold.test_dates):
            raise ValueError(
                f"Fold {fold.name}: OOF has {frame.height} rows, expected {len(fold.test_dates)}"
            )
        train = panel.filter(pl.col(COL_DATE_ID).is_in(fold.train_dates))
        train_mean = float(train[COL_TARGET].mean())
        train_vol = causal_vol_estimates(
            panel, fold.train_dates, vol_window=model_config.vol_window
        )
        prepared.append(
            {
                "fold": fold,
                "frame": frame,
                "train_mean": train_mean,
                "train_vol_median": float(np.median(train_vol)),
            }
        )
    return prepared


def add_metric(
    rows: list[dict[str, Any]],
    experiment: str,
    variant: str,
    mode: str,
    fold: str,
    metrics: dict[str, Any],
) -> None:
    rows.append(
        {
            "experiment": experiment,
            "variant": variant,
            "mode": mode,
            "fold": fold,
            "score": float(metrics["score"]),
            "sharpe": float(metrics["sharpe"]),
            "strategy_vol": float(metrics["strategy_vol"]),
            "market_vol": float(metrics["market_vol"]),
            "vol_penalty": float(metrics["vol_penalty"]),
            "return_penalty": float(metrics["return_penalty"]),
            "strategy_mean_excess": float(metrics["strategy_mean_excess"]),
            "market_mean_excess": float(metrics["market_mean_excess"]),
            "mean_allocation": float(metrics["mean_allocation"]),
            "turnover": float(metrics["turnover"]),
            "n_days": int(metrics["n_days"]),
            "cash": bool(metrics["cash"]),
        }
    )


def run(args: argparse.Namespace) -> None:
    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel, folds = make_folds(config)

    offline_path = Path(args.offline_oof) if args.offline_oof else latest_oof(
        config.paths.models_dir, "*_gdbt_notebook_cv_offline_*/oof_predictions.csv"
    )
    online_path = Path(args.online_oof) if args.online_oof else latest_oof(
        config.paths.models_dir, "*_gdbt_notebook_cv_online_*/oof_predictions.csv"
    )
    oofs = {"offline": load_oof(offline_path), "online": load_oof(online_path)}
    print(f"offline OOF: {offline_path}")
    print(f"online OOF:  {online_path}")

    fold_data = {
        mode: prepare_fold_data(panel, oof, folds, model_config)
        for mode, oof in oofs.items()
    }
    shared_fold_data = fold_data["offline"]

    # The saved OOF allocations must be exactly reproducible by our simulation.
    parity: dict[str, float] = {}
    for mode, items in fold_data.items():
        diffs = []
        for item in items:
            frame = item["frame"]
            positions = simulate_positions(
                frame["raw_pred"],
                frame["vol"],
                frame["multiplier"],
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
            )
            diffs.append(float(np.max(np.abs(positions - frame["allocation"].to_numpy()))))
        parity[mode] = max(diffs)
        if parity[mode] > 1e-7:
            raise RuntimeError(f"{mode} OOF allocation parity failed: max diff {parity[mode]}")
    print(f"OOF parity: {parity}")

    fold_rows: list[dict[str, Any]] = []
    positions_by_key: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}

    def register(
        experiment: str,
        variant: str,
        mode: str,
        items: list[dict[str, Any]],
        positions: dict[str, np.ndarray],
        metrics: dict[str, dict[str, Any]],
    ) -> None:
        key = (experiment, variant, mode)
        positions_by_key[key] = positions
        for item in items:
            name = item["fold"].name
            add_metric(fold_rows, experiment, variant, mode, name, metrics[name])

    # --- Buy & hold -------------------------------------------------------
    bh_items = shared_fold_data
    bh_positions = {
        item["fold"].name: np.ones(item["frame"].height, dtype=float)
        for item in bh_items
    }
    bh_metrics = {
        item["fold"].name: score_positions(
            bh_positions[item["fold"].name],
            item["frame"][COL_FWD].to_numpy(),
            item["frame"][COL_RF].to_numpy(),
        )
        for item in bh_items
    }
    register("buy_hold", "position_1.0", "shared", bh_items, bh_positions, bh_metrics)

    # --- Constant allocation grid ----------------------------------------
    for constant in CONSTANTS:
        variant = f"constant_{constant:g}"
        positions = {
            item["fold"].name: np.full(item["frame"].height, constant, dtype=float)
            for item in shared_fold_data
        }
        metrics = {
            item["fold"].name: score_positions(
                positions[item["fold"].name],
                item["frame"][COL_FWD].to_numpy(),
                item["frame"][COL_RF].to_numpy(),
            )
            for item in shared_fold_data
        }
        register("constant", variant, "shared", shared_fold_data, positions, metrics)

    # --- Zero-prediction ablation (cash control) --------------------------
    zero_positions = {
        item["fold"].name: np.zeros(item["frame"].height, dtype=float)
        for item in shared_fold_data
    }
    zero_metrics = {
        item["fold"].name: score_positions(
            zero_positions[item["fold"].name],
            item["frame"][COL_FWD].to_numpy(),
            item["frame"][COL_RF].to_numpy(),
        )
        for item in shared_fold_data
    }
    register(
        "prediction_ablation",
        "zero",
        "shared",
        shared_fold_data,
        zero_positions,
        zero_metrics,
    )

    # --- Model, vol-only and constant-forecast ablation ------------------
    for mode, items in fold_data.items():
        model_positions = {
            item["fold"].name: item["frame"]["allocation"].to_numpy()
            for item in items
        }
        model_metrics = {
            item["fold"].name: score_positions(
                model_positions[item["fold"].name],
                item["frame"][COL_FWD].to_numpy(),
                item["frame"][COL_RF].to_numpy(),
            )
            for item in items
        }
        register("model", "notebook_gbdt", mode, items, model_positions, model_metrics)

        for scale in VOL_SCALES:
            variant = f"scale_{scale:.1f}"
            positions = {
                item["fold"].name: simulate_vol_target(
                    item["frame"]["vol"],
                    item["train_vol_median"] * scale,
                    smoothing=model_config.smoothing,
                    transaction_cost=model_config.transaction_cost,
                )
                for item in items
            }
            metrics = {
                item["fold"].name: score_positions(
                    positions[item["fold"].name],
                    item["frame"][COL_FWD].to_numpy(),
                    item["frame"][COL_RF].to_numpy(),
                )
                for item in items
            }
            register("vol_only", variant, mode, items, positions, metrics)

        positions = {
            item["fold"].name: simulate_positions(
                np.full(item["frame"].height, max(item["train_mean"], 0.0)),
                item["frame"]["vol"],
                item["frame"]["multiplier"],
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
            )
            for item in items
        }
        metrics = {
            item["fold"].name: score_positions(
                positions[item["fold"].name],
                item["frame"][COL_FWD].to_numpy(),
                item["frame"][COL_RF].to_numpy(),
            )
            for item in items
        }
        register(
            "prediction_ablation",
            "train_mean",
            mode,
            items,
            positions,
            metrics,
        )

    # --- Prediction shuffle ----------------------------------------------
    shuffle_rows: list[dict[str, Any]] = []
    shuffle_means: dict[str, np.ndarray] = {}
    shuffle_summary: list[dict[str, Any]] = []
    for mode, items in fold_data.items():
        actual = [
            score_positions(
                item["frame"]["allocation"].to_numpy(),
                item["frame"][COL_FWD].to_numpy(),
                item["frame"][COL_RF].to_numpy(),
            )["score"]
            for item in items
        ]
        actual_mean = float(np.mean(actual))
        means = []
        for shuffle_index in range(args.shuffles):
            fold_scores = []
            for fold_index, item in enumerate(items):
                frame = item["frame"]
                rng = np.random.default_rng(
                    SHUFFLE_SEED + shuffle_index * 10007 + fold_index * 101
                    + (0 if mode == "offline" else 1_000_000)
                )
                raw_shuffled = rng.permutation(frame["raw_pred"].to_numpy())
                positions = simulate_positions(
                    raw_shuffled,
                    frame["vol"],
                    frame["multiplier"],
                    smoothing=model_config.smoothing,
                    transaction_cost=model_config.transaction_cost,
                    vol_scaling=model_config.vol_scaling,
                )
                metrics = score_positions(
                    positions,
                    frame[COL_FWD].to_numpy(),
                    frame[COL_RF].to_numpy(),
                )
                fold_scores.append(metrics["score"])
                shuffle_rows.append(
                    {
                        "experiment": "prediction_shuffle",
                        "variant": "permuted",
                        "mode": mode,
                        "fold": item["fold"].name,
                        "shuffle": shuffle_index,
                        "score": float(metrics["score"]),
                        "sharpe": float(metrics["sharpe"]),
                        "mean_allocation": float(metrics["mean_allocation"]),
                        "turnover": float(metrics["turnover"]),
                        "n_days": int(metrics["n_days"]),
                    }
                )
            means.append(float(np.mean(fold_scores)))
        shuffled = np.asarray(means)
        shuffle_means[mode] = shuffled
        p_value = float((1 + np.sum(shuffled >= actual_mean)) / (len(shuffled) + 1))
        shuffle_summary.append(
            {
                "experiment": "prediction_shuffle",
                "variant": "permuted",
                "mode": mode,
                "actual_mean_score": actual_mean,
                "shuffle_mean_score": float(shuffled.mean()),
                "shuffle_std_score": float(shuffled.std(ddof=1)),
                "shuffle_p05": float(np.percentile(shuffled, 5)),
                "shuffle_p50": float(np.percentile(shuffled, 50)),
                "shuffle_p95": float(np.percentile(shuffled, 95)),
                "p_value": p_value,
                "n_shuffles": int(args.shuffles),
            }
        )

    # --- Aggregate per experiment ----------------------------------------
    order = {
        "buy_hold": 0,
        "constant": 1,
        "vol_only": 2,
        "prediction_ablation": 3,
        "model": 4,
        "prediction_shuffle": 5,
    }
    summary_rows: list[dict[str, Any]] = []
    keys = list(positions_by_key.keys())
    keys.sort(key=lambda key: (order.get(key[0], 99), key[1], key[2]))
    for key in keys:
        experiment, variant, mode = key
        metrics = [
            row
            for row in fold_rows
            if (row["experiment"], row["variant"], row["mode"]) == key
        ]
        items = fold_data.get(mode, shared_fold_data)
        positions = np.concatenate([positions_by_key[key][item["fold"].name] for item in items])
        fwd = np.concatenate([item["frame"][COL_FWD].to_numpy() for item in items])
        rf = np.concatenate([item["frame"][COL_RF].to_numpy() for item in items])
        continuous = score_positions(positions, fwd, rf)
        scores = [row["score"] for row in metrics]
        sharpes = [row["sharpe"] for row in metrics]
        summary_rows.append(
            {
                "experiment": experiment,
                "variant": variant,
                "mode": mode,
                "score_mean": float(np.mean(scores)),
                "score_median": float(np.median(scores)),
                "score_best": float(np.max(scores)),
                "score_worst": float(np.min(scores)),
                "sharpe_mean": float(np.mean(sharpes)),
                "sharpe_median": float(np.median(sharpes)),
                "score_continuous": float(continuous["score"]),
                "sharpe_continuous": float(continuous["sharpe"]),
                "mean_allocation": float(np.mean(positions)),
                "turnover": float(np.mean([row["turnover"] for row in metrics])),
                "n_days": int(sum(row["n_days"] for row in metrics)),
                "cash": bool(all(row["cash"] for row in metrics)),
            }
        )

    # --- Write records ----------------------------------------------------
    output = Path(args.output)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pl.DataFrame(fold_rows)
    summary_frame = pl.DataFrame(summary_rows)
    shuffle_frame = pl.DataFrame(shuffle_rows) if shuffle_rows else pl.DataFrame()
    fold_frame.write_csv(output / "fold_metrics.csv")
    summary_frame.write_csv(output / "summary.csv")
    if shuffle_frame.height:
        shuffle_frame.write_csv(output / "shuffle_scores.csv")

    manifest = {
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "config": str(args.config),
        "offline_oof": str(offline_path),
        "online_oof": str(online_path),
        "folds": {
            item["fold"].name: {
                "train": [int(item["fold"].train_dates[0]), int(item["fold"].train_dates[-1])],
                "test": [int(item["fold"].test_dates[0]), int(item["fold"].test_dates[-1])],
                "n_train": len(item["fold"].train_dates),
                "n_test": len(item["fold"].test_dates),
                "train_mean_target": item["train_mean"],
                "train_vol_median": item["train_vol_median"],
            }
            for item in shared_fold_data
        },
        "definitions": {
            "buy_hold": "position = 1.0 every test day",
            "constant": "position = constant every test day; 0.0 is cash",
            "vol_only": "position target = scale * median(training causal vol) / current vol",
            "prediction_ablation_train_mean": (
                "raw_pred replaced by the fold's training-window mean target; "
                "same multiplier, vol scaling, smoothing and cost"
            ),
            "prediction_ablation_zero": "raw_pred = 0; cash control",
            "prediction_shuffle": "raw_pred permuted within each test fold",
        },
        "constants": CONSTANTS,
        "vol_scales": VOL_SCALES,
        "shuffle": {
            "n": int(args.shuffles),
            "seed": SHUFFLE_SEED,
            "scope": "within each test fold; vol and multiplier keep their actual dates",
        },
        "cash_score_convention": "cash -> score=sharpe=0.0, mean_allocation=0",
        "oof_parity_max_abs_diff": parity,
    }
    write_json(output / "experiment_manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "manifest": manifest,
            "summary": summary_rows,
            "prediction_shuffle": shuffle_summary,
        },
    )

    write_readme(output, summary_rows, shuffle_summary, manifest, fold_rows)

    # --- Plots ------------------------------------------------------------
    plot_all(
        plot_dir,
        fold_rows,
        summary_rows,
        shuffle_means,
        shuffle_summary,
        fold_data,
        positions_by_key,
        folds,
    )
    print(f"wrote {output}")
    print(summary_frame.select(["experiment", "variant", "mode", "score_mean", "score_continuous", "sharpe_mean", "mean_allocation"]))


def write_readme(
    output: Path,
    summary_rows: list[dict[str, Any]],
    shuffle_summary: list[dict[str, Any]],
    manifest: dict[str, Any],
    fold_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Allocation experiments",
        "",
        f"Generated: {manifest['created_at']}  ",
        f"Git: `{manifest['git_sha']}`  ",
        f"Offline OOF: `{manifest['offline_oof']}`  ",
        f"Online OOF: `{manifest['online_oof']}`",
        "",
        "All experiments use the same six walk-forward test folds. Buy & hold,",
        "constant allocations, volatility-only, and the constant-forecast ablation",
        "keep the same official scoring definition. The model rows use the saved",
        "OOF allocation paths.",
        "",
        "## Summary",
        "",
        "| experiment | variant | mode | mean score | continuous score | mean Sharpe | mean alloc | turnover |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['variant']} | {row['mode']} | "
            f"{row['score_mean']:.4f} | {row['score_continuous']:.4f} | "
            f"{row['sharpe_mean']:.4f} | {row['mean_allocation']:.4f} | "
            f"{row['turnover']:.4f} |"
        )
    lines += ["", "## Prediction shuffle", "", "| mode | actual mean | shuffle mean | p05 | p50 | p95 | p-value |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in shuffle_summary:
        lines.append(
            f"| {row['mode']} | {row['actual_mean_score']:.4f} | "
            f"{row['shuffle_mean_score']:.4f} | {row['shuffle_p05']:.4f} | "
            f"{row['shuffle_p50']:.4f} | {row['shuffle_p95']:.4f} | "
            f"{row['p_value']:.4f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `fold_metrics.csv`: every experiment x fold metric",
        "- `summary.csv`: aggregated table",
        "- `shuffle_scores.csv`: every permutation x fold score",
        "- `summary.json` / `experiment_manifest.json`: full records",
        "- `plots/`: Sharpe by fold, summary score, constant grid, allocation paths,",
        "  shuffle distribution, and equity curves",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_all(
    plot_dir: Path,
    fold_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    shuffle_means: dict[str, np.ndarray],
    shuffle_summary: list[dict[str, Any]],
    fold_data: dict[str, list[dict[str, Any]]],
    positions_by_key: dict[tuple[str, str, str], dict[str, np.ndarray]],
    folds: list[Any],
) -> None:
    summary = {(r["experiment"], r["variant"], r["mode"]): r for r in summary_rows}
    fold_lookup: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in fold_rows:
        key = (row["experiment"], row["variant"], row["mode"])
        fold_lookup.setdefault(key, {})[row["fold"]] = row
    fold_names = [fold.name for fold in folds]

    # 1. Sharpe by fold
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(fold_names))
    width = 0.8 / len(KEY_SPECS)
    for i, (experiment, variant, mode, label) in enumerate(KEY_SPECS):
        values = [
            fold_lookup.get((experiment, variant, mode), {}).get(fold, {}).get("sharpe", np.nan)
            for fold in fold_names
        ]
        ax.bar(x + (i - (len(KEY_SPECS) - 1) / 2) * width, values, width, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_names)
    ax.set_ylabel("raw Sharpe")
    ax.set_title("Raw Sharpe by test fold")
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_sharpe_by_fold.png")
    plt.close(fig)

    # 2. Mean score summary
    labels = [spec[3] for spec in KEY_SPECS]
    values = [summary.get((spec[0], spec[1], spec[2]), {}).get("score_mean", np.nan) for spec in KEY_SPECS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    ax.barh(labels, values, color=colors)
    for i, value in enumerate(values):
        ax.text(value + (0.015 if value >= 0 else -0.015), i, f"{value:.3f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mean official score (6 folds)")
    ax.set_title("Mean score across folds")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plot_dir / "02_score_summary.png")
    plt.close(fig)

    # 3. Constant allocation grid
    constants = [r for r in summary_rows if r["experiment"] == "constant"]
    constants.sort(key=lambda r: float(r["variant"].replace("constant_", "")))
    xs = [float(r["variant"].replace("constant_", "")) for r in constants]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, [r["score_mean"] for r in constants], marker="o", label="mean fold score")
    ax.plot(xs, [r["score_continuous"] for r in constants], marker="s", label="continuous OOF score")
    for key, label, color in [
        (("buy_hold", "position_1.0", "shared"), "Buy & Hold 1.0", "tab:red"),
        (("model", "notebook_gbdt", "offline"), "Model offline", "tab:purple"),
        (("model", "notebook_gbdt", "online"), "Model online", "tab:green"),
    ]:
        value = summary[key]["score_mean"]
        ax.axhline(value, linestyle="--", color=color, linewidth=1, label=label)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("constant position")
    ax.set_ylabel("official score")
    ax.set_title("Constant allocation vs model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_constant_grid.png")
    plt.close(fig)

    # 4. Allocation paths for the two most recent folds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    path_specs = [
        ("model", "notebook_gbdt", "offline", "Model offline"),
        ("model", "notebook_gbdt", "online", "Model online"),
        ("vol_only", "scale_1.0", "offline", "Vol-only 1.0x"),
        ("prediction_ablation", "train_mean", "offline", "Ablation const mean"),
        ("constant", "constant_0.5", "shared", "Constant 0.5"),
    ]
    for ax, fold_name in zip(axes, ["2023", "2025"]):
        for experiment, variant, mode, label in path_specs:
            key = (experiment, variant, mode)
            items = fold_data.get(mode, fold_data["offline"])
            item = next(i for i in items if i["fold"].name == fold_name)
            ax.step(
                item["frame"][COL_DATE_ID].to_numpy(),
                positions_by_key[key][fold_name],
                where="post",
                label=label,
                linewidth=1.1,
            )
        ax.set_title(f"{fold_name} test fold")
        ax.set_xlabel("date_id")
        ax.set_ylim(-0.02, 2.05)
    axes[0].set_ylabel("allocation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle("OOF allocation paths", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(plot_dir / "04_allocation_paths.png")
    plt.close(fig)

    # 5. Shuffle distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode, color in [("offline", "tab:blue"), ("online", "tab:orange")]:
        values = shuffle_means[mode]
        ax.hist(values, bins=30, alpha=0.45, color=color, label=f"{mode} shuffled means")
        row = next(r for r in shuffle_summary if r["mode"] == mode)
        ax.axvline(row["actual_mean_score"], color=color, linewidth=2,
                   label=f"{mode} actual {row['actual_mean_score']:.3f} (p={row['p_value']:.3f})")
    ax.set_xlabel("mean official score across folds")
    ax.set_ylabel("shuffle count")
    ax.set_title("Prediction shuffle null distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_shuffle_distribution.png")
    plt.close(fig)

    # 6. Equity curves (concatenated OOF test days, no calendar gaps)
    fig, ax = plt.subplots(figsize=(12, 5))
    fold_starts = []
    offset = 0
    for fold in folds:
        fold_starts.append(offset)
        offset += len(fold.test_dates)
    for experiment, variant, mode, label in [
        ("buy_hold", "position_1.0", "shared", "Buy & Hold 1.0"),
        ("constant", "constant_0.5", "shared", "Constant 0.5"),
        ("model", "notebook_gbdt", "offline", "Model offline"),
        ("model", "notebook_gbdt", "online", "Model online"),
        ("vol_only", "scale_1.0", "offline", "Vol-only 1.0x"),
        ("prediction_ablation", "train_mean", "offline", "Ablation const mean"),
    ]:
        key = (experiment, variant, mode)
        items = fold_data.get(mode, fold_data["offline"])
        positions = np.concatenate([positions_by_key[key][item["fold"].name] for item in items])
        fwd = np.concatenate([item["frame"][COL_FWD].to_numpy() for item in items])
        rf = np.concatenate([item["frame"][COL_RF].to_numpy() for item in items])
        excess = positions * (fwd - rf)
        curve = (np.cumprod(1.0 + excess) - 1.0) * 100.0
        ax.plot(np.arange(len(curve)), curve, label=label, linewidth=1.3)
    for start in fold_starts[1:]:
        ax.axvline(start, color="grey", linestyle=":", linewidth=0.7)
    ax.set_ylabel("cumulative excess return (%)")
    ax.set_xlabel("OOF test day (folds concatenated)")
    ax.set_title("OOF equity curves (compounded excess return)")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "06_equity_curves.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/cv.yaml")
    parser.add_argument("--output", default="reports/experiments")
    parser.add_argument("--offline-oof", default=None)
    parser.add_argument("--online-oof", default=None)
    parser.add_argument("--shuffles", type=int, default=500)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
