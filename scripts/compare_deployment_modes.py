"""Compare deployment-faithful daily-retrain GBDT against existing modes.

All strategies use the same six test folds and the same official scoring /
return / drawdown functions. Model modes use the notebook allocation policy;
Buy & Hold and Constant 1.25 are fixed allocations; vol-only uses the same
smoothing and cost rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common.config import COL_DATE_ID, COL_FWD, COL_RF, COL_TARGET, Config
from common.data import date_ids_for_years, load_date_map, load_panel
from common.evaluation import strategy_equity, summarize_positions
from common.experiments import causal_vol_estimates, simulate_vol_target
from common.registry import git_sha, utc_now, write_json
from common.splits import rolling_window_folds
from gdbt.config import GBDTConfig

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

STRATEGY_LABELS = {
    "daily_retrain": "Daily retrain",
    "offline": "Offline",
    "online": "Online",
    "buy_hold": "Buy & Hold",
    "constant_1.25": "Constant 1.25",
    "vol_only": "Vol-only 1.0x",
}


def latest_daily(models_dir: str | Path) -> Path:
    paths = [
        p
        for p in Path(models_dir).glob("*_gdbt_daily_retrain_*/oof_predictions.csv")
        if "smoke" not in p.parent.name
    ]
    if not paths:
        raise FileNotFoundError("No non-smoke daily-retrain OOF found")
    return sorted(paths, key=lambda p: p.parent.name)[-1]


def load_oof(path: str | Path) -> pl.DataFrame:
    frame = pl.read_csv(path).sort(COL_DATE_ID)
    required = {COL_DATE_ID, "allocation", "raw_pred", "vol", "multiplier"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    return frame


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


def align(frame: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    labels = panel.select([COL_DATE_ID, COL_TARGET, COL_FWD, COL_RF])
    joined = frame.join(labels, on=COL_DATE_ID, how="left")
    if any(joined[column].null_count() for column in (COL_FWD, COL_RF)):
        raise ValueError("Could not join forward returns / risk-free rate")
    return joined.sort(COL_DATE_ID)


def fold_frames(frame: pl.DataFrame, folds: list[Any]) -> dict[str, pl.DataFrame]:
    out = {}
    for fold in folds:
        sub = frame.filter(pl.col(COL_DATE_ID).is_in(fold.test_dates)).sort(COL_DATE_ID)
        if sub.height != len(fold.test_dates):
            raise ValueError(f"Fold {fold.name}: {sub.height} rows != {len(fold.test_dates)}")
        out[fold.name] = sub
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/cv.yaml")
    parser.add_argument("--daily-oof", default=None)
    parser.add_argument("--offline-oof", default=None)
    parser.add_argument("--online-oof", default=None)
    parser.add_argument("--output", default="reports/deployment_eval")
    args = parser.parse_args()

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel, folds = make_folds(config)

    daily_path = Path(args.daily_oof) if args.daily_oof else latest_daily(
        config.paths.models_dir
    )
    offline_path = Path(args.offline_oof) if args.offline_oof else sorted(
        Path(config.paths.models_dir).glob("*_gdbt_notebook_cv_offline_*/oof_predictions.csv")
    )[-1]
    online_path = Path(args.online_oof) if args.online_oof else sorted(
        Path(config.paths.models_dir).glob("*_gdbt_notebook_cv_online_*/oof_predictions.csv")
    )[-1]
    daily = align(load_oof(daily_path), panel)
    offline = align(load_oof(offline_path), panel)
    online = align(load_oof(online_path), panel)

    daily_by_fold = fold_frames(daily, folds)
    offline_by_fold = fold_frames(offline, folds)
    online_by_fold = fold_frames(online, folds)

    strategies: dict[str, dict[str, np.ndarray]] = {
        "daily_retrain": {},
        "offline": {},
        "online": {},
        "buy_hold": {},
        "constant_1.25": {},
        "vol_only": {},
    }
    for fold in folds:
        frame = daily_by_fold[fold.name]
        n = frame.height
        strategies["daily_retrain"][fold.name] = frame["allocation"].to_numpy()
        strategies["offline"][fold.name] = offline_by_fold[fold.name]["allocation"].to_numpy()
        strategies["online"][fold.name] = online_by_fold[fold.name]["allocation"].to_numpy()
        strategies["buy_hold"][fold.name] = np.ones(n)
        strategies["constant_1.25"][fold.name] = np.full(n, 1.25)
        train_vol = causal_vol_estimates(
            panel, fold.train_dates, vol_window=model_config.vol_window
        )
        strategies["vol_only"][fold.name] = simulate_vol_target(
            frame["vol"],
            float(np.median(train_vol)),
            smoothing=model_config.smoothing,
            transaction_cost=model_config.transaction_cost,
        )

    fold_metrics: list[dict[str, Any]] = []
    for strategy, by_fold in strategies.items():
        for fold in folds:
            frame = daily_by_fold[fold.name]
            report = summarize_positions(
                by_fold[fold.name],
                frame[COL_FWD].to_numpy(),
                frame[COL_RF].to_numpy(),
            )
            fold_metrics.append(
                {
                    "strategy": strategy,
                    "fold": fold.name,
                    **{k: report[k] for k in [
                        "score",
                        "sharpe",
                        "cumulative_return",
                        "max_drawdown",
                        "turnover",
                        "mean_allocation",
                        "n_days",
                    ]},
                }
            )

    summary: list[dict[str, Any]] = []
    for strategy, by_fold in strategies.items():
        positions = np.concatenate([by_fold[fold.name] for fold in folds])
        fwd = np.concatenate([daily_by_fold[fold.name][COL_FWD].to_numpy() for fold in folds])
        rf = np.concatenate([daily_by_fold[fold.name][COL_RF].to_numpy() for fold in folds])
        cont = summarize_positions(positions, fwd, rf)
        rows = [r for r in fold_metrics if r["strategy"] == strategy]
        summary.append(
            {
                "strategy": strategy,
                "label": STRATEGY_LABELS[strategy],
                "score_mean": float(np.mean([r["score"] for r in rows])),
                "score_median": float(np.median([r["score"] for r in rows])),
                "score_best": float(np.max([r["score"] for r in rows])),
                "score_worst": float(np.min([r["score"] for r in rows])),
                "score_continuous": cont["score"],
                "sharpe_mean": float(np.mean([r["sharpe"] for r in rows])),
                "sharpe_continuous": cont["sharpe"],
                "cumulative_return_continuous": cont["cumulative_return"],
                "max_drawdown_continuous": cont["max_drawdown"],
                "turnover_continuous": cont["turnover"],
                "mean_allocation_continuous": cont["mean_allocation"],
                "n_days": cont["n_days"],
            }
        )

    output = Path(args.output)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(fold_metrics).write_csv(output / "fold_metrics.csv")
    pl.DataFrame(summary).write_csv(output / "summary.csv")
    manifest = {
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "daily_oof": str(daily_path),
        "offline_oof": str(offline_path),
        "online_oof": str(online_path),
        "folds": {
            fold.name: {
                "train": [int(fold.train_dates[0]), int(fold.train_dates[-1])],
                "test": [int(fold.test_dates[0]), int(fold.test_dates[-1])],
            }
            for fold in folds
        },
        "cost_model": {
            "policy_transaction_cost": model_config.transaction_cost,
            "note": "Model/vol-only allocations include the notebook multiplicative cost; fixed allocations do not trade daily.",
        },
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", {"manifest": manifest, "summary": summary})

    # Plots
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_LABELS]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, [next(r["score_mean"] for r in summary if r["strategy"] == s) for s in STRATEGY_LABELS], width, label="mean fold score")
    ax.bar(x + width / 2, [next(r["score_continuous"] for r in summary if r["strategy"] == s) for s in STRATEGY_LABELS], width, label="continuous score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("official score")
    ax.set_title("Deployment-faithful comparison")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_score_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for strategy in STRATEGY_LABELS:
        positions = np.concatenate([strategies[strategy][fold.name] for fold in folds])
        fwd = np.concatenate([daily_by_fold[fold.name][COL_FWD].to_numpy() for fold in folds])
        rf = np.concatenate([daily_by_fold[fold.name][COL_RF].to_numpy() for fold in folds])
        equity, _ = strategy_equity(positions, fwd, rf)
        ax.plot(np.arange(len(equity)), (equity - 1.0) * 100.0, label=STRATEGY_LABELS[strategy])
    ax.set_xlabel("OOF test day (folds concatenated)")
    ax.set_ylabel("cumulative return (%)")
    ax.set_title("Cumulative return")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_cumulative_return.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for strategy in STRATEGY_LABELS:
        positions = np.concatenate([strategies[strategy][fold.name] for fold in folds])
        fwd = np.concatenate([daily_by_fold[fold.name][COL_FWD].to_numpy() for fold in folds])
        rf = np.concatenate([daily_by_fold[fold.name][COL_RF].to_numpy() for fold in folds])
        equity, _ = strategy_equity(positions, fwd, rf)
        peaks = np.maximum.accumulate(equity)
        drawdown = equity / peaks - 1.0
        ax.plot(np.arange(len(equity)), drawdown * 100.0, label=STRATEGY_LABELS[strategy])
    ax.set_xlabel("OOF test day (folds concatenated)")
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Drawdown")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_drawdown.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(labels, [next(r["turnover_continuous"] for r in summary if r["strategy"] == s) for s in STRATEGY_LABELS])
    ax.set_ylabel("mean daily |Δposition|")
    ax.set_title("Turnover")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_turnover.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fold_names = [fold.name for fold in folds]
    x = np.arange(len(fold_names))
    width = 0.8 / len(STRATEGY_LABELS)
    for i, strategy in enumerate(STRATEGY_LABELS):
        values = [
            next(r["score"] for r in fold_metrics if r["strategy"] == strategy and r["fold"] == fold)
            for fold in fold_names
        ]
        ax.bar(x + (i - (len(STRATEGY_LABELS) - 1) / 2) * width, values, width, label=STRATEGY_LABELS[strategy])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_names)
    ax.set_ylabel("official score")
    ax.set_title("Score by fold")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_fold_scores.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=True)
    for ax, fold_name in zip(axes, ["2021", "2025"]):
        x_values = daily_by_fold[fold_name]["date_id"].to_numpy()
        for strategy, color in [
            ("daily_retrain", "tab:blue"),
            ("offline", "tab:orange"),
            ("online", "tab:green"),
        ]:
            ax.plot(x_values, strategies[strategy][fold_name], label=STRATEGY_LABELS[strategy], color=color, linewidth=0.9)
        ax.set_title(f"{fold_name} fold")
        ax.set_ylabel("allocation")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    axes[-1].set_xlabel("date_id")
    fig.suptitle("Allocation paths: daily retrain vs Offline vs Online", y=0.99)
    fig.tight_layout()
    fig.savefig(plot_dir / "06_allocation_paths.png")
    plt.close(fig)

    print(pl.DataFrame(summary).select([
        "strategy", "score_mean", "score_continuous", "sharpe_mean", "cumulative_return_continuous", "max_drawdown_continuous", "turnover_continuous", "mean_allocation_continuous",
    ]))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
