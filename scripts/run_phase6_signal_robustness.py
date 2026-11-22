"""Phase 6: signal robustness, multiplier robustness and post-window validation.

The post-window protocol is frozen in docs/phase6_preregistration.md and
configs/gdbt/phase6.yaml before this script is run. This script only reads
already-saved position paths; it never retrains a model or tunes a parameter.

Outputs:
  signal_regime.csv, multiplier_grid.csv, concentration_*.csv,
  statistical_tests_official.csv, statistical_tests_post.csv,
  post_metrics.csv, post_signal_regime.csv, plots/*
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm

from common.config import COL_DATE_ID, COL_TARGET_METRIC, Config
from common.evaluation import (
    holm_adjust,
    paired_block_bootstrap,
    summarize_positions,
)
from common.experiments import causal_vol_estimates, simulate_positions
from common.metrics import ParticipantVisibleError
from common.registry import utc_now, write_json
from common.validation import (
    moving_block_indices,
    pearson_ic,
    position_stats,
    quantile_returns,
    rank_ic,
)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_positions(positions: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if np.allclose(positions, 0.0):
        return {
            "score": 0.0,
            "sharpe": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "mean_allocation": 0.0,
            "strategy_vol": 0.0,
            "vol_penalty": 1.0,
            "return_penalty": 1.0,
            "strategy_mean_excess": 0.0,
        }
    try:
        report = summarize_positions(positions, target, np.zeros_like(target))
        return report
    except ParticipantVisibleError:
        return {
            "score": float("nan"),
            "sharpe": float("nan"),
            "cumulative_return": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
            "mean_allocation": float("nan"),
            "strategy_vol": float("nan"),
            "vol_penalty": float("nan"),
            "return_penalty": float("nan"),
            "strategy_mean_excess": float("nan"),
        }


def hac_mean(values: np.ndarray, max_lags: int = 10) -> dict[str, float]:
    """Newey-West/HAC inference for the mean of a daily series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {key: float("nan") for key in ("mean", "se", "t", "p_two", "p_positive", "ci_low", "ci_high")}
    mean = float(np.mean(x))
    e = x - mean
    gamma0 = float(np.dot(e, e) / n)
    variance = gamma0
    for lag in range(1, min(int(max_lags), n - 1) + 1):
        weight = 1.0 - lag / (int(max_lags) + 1.0)
        gamma = float(np.dot(e[lag:], e[:-lag]) / n)
        variance += 2.0 * weight * gamma
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance / n))
    if se == 0.0 or not np.isfinite(se):
        return {key: float("nan") for key in ("mean", "se", "t", "p_two", "p_positive", "ci_low", "ci_high")}
    t = mean / se
    return {
        "mean": mean,
        "se": se,
        "t": float(t),
        "p_two": float(2.0 * norm.sf(abs(t))),
        "p_positive": float(norm.sf(t)),
        "ci_low": float(mean - 1.96 * se),
        "ci_high": float(mean + 1.96 * se),
    }


def signal_bootstrap(
    prediction: np.ndarray,
    target: np.ndarray,
    block_length: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    n = len(pred)
    actual_ic = rank_ic(pred, y)
    bins = quantile_returns(pred, y, n_quantiles=5)
    actual_spread = float(bins[-1]["mean_target"] - bins[0]["mean_target"]) if bins else float("nan")
    rng = np.random.default_rng(int(seed))
    ics = np.empty(int(reps), dtype=float)
    spreads = np.empty(int(reps), dtype=float)
    for rep in range(int(reps)):
        indices = moving_block_indices(n, block_length, rng)
        ics[rep] = rank_ic(pred[indices], y[indices])
        sampled = quantile_returns(pred[indices], y[indices], n_quantiles=5)
        spreads[rep] = (
            float(sampled[-1]["mean_target"] - sampled[0]["mean_target"])
            if sampled
            else float("nan")
        )
    finite_ic = ics[np.isfinite(ics)]
    finite_spread = spreads[np.isfinite(spreads)]
    return {
        "actual_rank_ic": float(actual_ic),
        "actual_q5_minus_q1": actual_spread,
        "ic_ci_low": float(np.percentile(finite_ic, 2.5)) if len(finite_ic) else float("nan"),
        "ic_ci_high": float(np.percentile(finite_ic, 97.5)) if len(finite_ic) else float("nan"),
        "ic_p_positive": (
            (1.0 + float(np.sum(finite_ic >= actual_ic))) / (len(finite_ic) + 1.0)
            if len(finite_ic)
            else float("nan")
        ),
        "spread_ci_low": float(np.percentile(finite_spread, 2.5)) if len(finite_spread) else float("nan"),
        "spread_ci_high": float(np.percentile(finite_spread, 97.5)) if len(finite_spread) else float("nan"),
        "spread_p_positive": (
            (1.0 + float(np.sum(finite_spread >= actual_spread))) / (len(finite_spread) + 1.0)
            if len(finite_spread)
            else float("nan")
        ),
        "n_boot": int(reps),
        "block_length": int(block_length),
    }


def regime_thresholds(panel: pl.DataFrame, first_date_id: int, model_config: GBDTConfig) -> tuple[float, float]:
    initial = (
        panel.filter((pl.col(COL_DATE_ID) < first_date_id) & pl.col("target").is_not_null())
        .sort(COL_DATE_ID)
        .tail(model_config.max_train_rows)
    )
    vol = causal_vol_estimates(panel, initial[COL_DATE_ID].to_list(), vol_window=model_config.vol_window)
    q33, q67 = np.nanquantile(vol, [0.33, 0.67])
    return float(q33), float(q67)


def signal_regime_rows(frame: pl.DataFrame, thresholds: tuple[float, float]) -> list[dict[str, Any]]:
    low, high = thresholds
    labels = np.where(
        frame["vol"].to_numpy() <= low,
        "low_vol",
        np.where(frame["vol"].to_numpy() >= high, "high_vol", "mid_vol"),
    )
    frame = frame.with_columns(pl.Series("regime", labels))
    rows = []
    for regime in ("low_vol", "mid_vol", "high_vol"):
        sub = frame.filter(pl.col("regime") == regime)
        if sub.height == 0:
            continue
        pred = sub["raw_pred"].to_numpy()
        target = sub[COL_TARGET_METRIC].to_numpy()
        bins = quantile_returns(pred, target, n_quantiles=5)
        quantities = {f"q{i + 1}": bins[i]["mean_target"] for i in range(len(bins))}
        quantities.update(
            {
                f"q{i + 1}_count": bins[i]["count"] for i in range(len(bins))
            }
        )
        rows.append(
            {
                "regime": regime,
                "n_days": sub.height,
                "rank_ic": rank_ic(pred, target),
                "pearson_ic": pearson_ic(pred, target),
                **quantities,
                "q5_minus_q1": (
                    float(bins[-1]["mean_target"] - bins[0]["mean_target"])
                    if bins
                    else float("nan")
                ),
            }
        )
    return rows


def multiplier_rows(
    frame: pl.DataFrame,
    model_config: GBDTConfig,
    multipliers: list[float],
) -> list[dict[str, Any]]:
    raw = frame["raw_pred"].to_numpy()
    vol = frame["vol"].to_numpy()
    target = frame[COL_TARGET_METRIC].to_numpy()
    rows = [
        {
            "multiplier": "original_switch",
            "mean_allocation": float(np.mean(frame["allocation"].to_numpy())),
            "strategy_vol": float(score_positions(frame["allocation"].to_numpy(), target)["strategy_vol"]),
            "score": float(score_positions(frame["allocation"].to_numpy(), target)["score"]),
            "raw_sharpe": float(score_positions(frame["allocation"].to_numpy(), target)["sharpe"]),
            "cumulative_return": float(score_positions(frame["allocation"].to_numpy(), target)["cumulative_return"]),
            "max_drawdown": float(score_positions(frame["allocation"].to_numpy(), target)["max_drawdown"]),
            "turnover": float(score_positions(frame["allocation"].to_numpy(), target)["turnover"]),
            "vol_penalty": float(score_positions(frame["allocation"].to_numpy(), target)["vol_penalty"]),
            "return_penalty": float(score_positions(frame["allocation"].to_numpy(), target)["return_penalty"]),
        }
    ]
    for multiplier in multipliers:
        positions = simulate_positions(
            raw,
            vol,
            np.full_like(vol, float(multiplier)),
            smoothing=model_config.smoothing,
            transaction_cost=model_config.transaction_cost,
            vol_scaling=model_config.vol_scaling,
            max_position=model_config.max_signal,
        )
        metrics = score_positions(positions, target)
        rows.append(
            {
                "multiplier": float(multiplier),
                "mean_allocation": float(np.mean(positions)),
                "strategy_vol": float(metrics["strategy_vol"]),
                "score": float(metrics["score"]),
                "raw_sharpe": float(metrics["sharpe"]),
                "cumulative_return": float(metrics["cumulative_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "turnover": float(metrics["turnover"]),
                "vol_penalty": float(metrics["vol_penalty"]),
                "return_penalty": float(metrics["return_penalty"]),
            }
        )
    return rows


def concentration(
    frame: pl.DataFrame,
    months: list[str],
    block_lengths: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target = frame[COL_TARGET_METRIC].to_numpy()
    positions = frame["allocation"].to_numpy()
    contribution = positions * target
    base_score = float(score_positions(positions, target)["score"])
    base_cumulative = float(score_positions(positions, target)["cumulative_return"])
    daily_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for month in months:
        mask = (frame["month"] == month).to_numpy()
        indices = np.flatnonzero(mask)
        month_contrib = contribution[indices]
        positive = month_contrib[month_contrib > 0]
        positive_total = float(np.sum(positive))
        order = np.sort(positive)[::-1]
        cumulative = np.cumsum(order)
        days_to_half = int(np.searchsorted(cumulative, 0.5 * positive_total) + 1) if positive_total > 0 else 0
        summary[month] = {
            "n_days": int(len(indices)),
            "total_contribution": float(np.sum(month_contrib)),
            "positive_contribution": positive_total,
            "top1_share": float(np.sum(order[:1]) / positive_total) if positive_total > 0 else float("nan"),
            "top3_share": float(np.sum(order[:3]) / positive_total) if positive_total > 0 else float("nan"),
            "top5_share": float(np.sum(order[:5]) / positive_total) if positive_total > 0 else float("nan"),
            "days_for_half_positive": days_to_half,
        }
        for global_index in indices:
            daily_rows.append(
                {
                    "month": month,
                    "date_id": int(frame[COL_DATE_ID].to_numpy()[int(global_index)]),
                    "allocation": float(positions[global_index]),
                    "target_metric": float(target[global_index]),
                    "contribution": float(contribution[global_index]),
                }
            )
            keep = np.ones(len(frame), dtype=bool)
            keep[global_index] = False
            metrics = score_positions(positions[keep], target[keep])
            loo_rows.append(
                {
                    "month": month,
                    "removed_date_id": int(frame[COL_DATE_ID].to_numpy()[int(global_index)]),
                    "removed_contribution": float(contribution[global_index]),
                    "score": float(metrics["score"]),
                    "cumulative_return": float(metrics["cumulative_return"]),
                    "delta_score": float(metrics["score"] - base_score),
                    "delta_cumulative": float(metrics["cumulative_return"] - base_cumulative),
                }
            )
        for block_length in block_lengths:
            for start in range(0, len(indices) - int(block_length) + 1):
                removed = indices[start : start + int(block_length)]
                keep = np.ones(len(frame), dtype=bool)
                keep[removed] = False
                metrics = score_positions(positions[keep], target[keep])
                block_rows.append(
                    {
                        "month": month,
                        "block_length": int(block_length),
                        "start_date_id": int(frame[COL_DATE_ID].to_numpy()[int(removed[0])]),
                        "end_date_id": int(frame[COL_DATE_ID].to_numpy()[int(removed[-1])]),
                        "removed_contribution": float(np.sum(contribution[removed])),
                        "score": float(metrics["score"]),
                        "cumulative_return": float(metrics["cumulative_return"]),
                        "delta_score": float(metrics["score"] - base_score),
                        "delta_cumulative": float(metrics["cumulative_return"] - base_cumulative),
                    }
                )
    return daily_rows, loo_rows, block_rows, summary


def test_family(
    frame: pl.DataFrame,
    baseline_frames: dict[str, pl.DataFrame],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = frame[COL_TARGET_METRIC].to_numpy()
    positions = frame["allocation"].to_numpy()
    raw = frame["raw_pred"].to_numpy()
    daily_return = positions * target
    tests = []

    mean_test = hac_mean(daily_return, int(config["hac_max_lags"]))
    tests.append(
        {
            "hypothesis": "daily_mean_excess_positive",
            "estimate": mean_test["mean"],
            "hac_se": mean_test["se"],
            "hac_t": mean_test["t"],
            "p_two": mean_test["p_two"],
            "p_positive": mean_test["p_positive"],
            "ci_low": mean_test["ci_low"],
            "ci_high": mean_test["ci_high"],
        }
    )
    for baseline in ("buy_hold", "constant_1.25", "vol_only"):
        base = baseline_frames[baseline]["allocation"].to_numpy()
        difference = (positions - base) * target
        result = hac_mean(difference, int(config["hac_max_lags"]))
        tests.append(
            {
                "hypothesis": f"daily_vs_{baseline.replace('.', '')}",
                "estimate": result["mean"],
                "hac_se": result["se"],
                "hac_t": result["t"],
                "p_two": result["p_two"],
                "p_positive": result["p_positive"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
            }
        )

    boot = signal_bootstrap(
        raw,
        target,
        block_length=int(config["block_length"]),
        reps=int(config["bootstrap_reps"]),
        seed=seed,
    )
    tests.append(
        {
            "hypothesis": "rank_ic_positive",
            "estimate": boot["actual_rank_ic"],
            "hac_se": float("nan"),
            "hac_t": float("nan"),
            "p_two": float("nan"),
            "p_positive": boot["ic_p_positive"],
            "ci_low": boot["ic_ci_low"],
            "ci_high": boot["ic_ci_high"],
        }
    )
    tests.append(
        {
            "hypothesis": "q5_minus_q1_positive",
            "estimate": boot["actual_q5_minus_q1"],
            "hac_se": float("nan"),
            "hac_t": float("nan"),
            "p_two": float("nan"),
            "p_positive": boot["spread_p_positive"],
            "ci_low": boot["spread_ci_low"],
            "ci_high": boot["spread_ci_high"],
        }
    )
    adjusted = holm_adjust([row["p_positive"] for row in tests])
    for row, value in zip(tests, adjusted):
        row["p_holm"] = float(value)
        row["significant_holm"] = bool(value < 0.05)

    score_boot_rows = []
    for baseline in ("buy_hold", "constant_1.25", "vol_only"):
        result = paired_block_bootstrap(
            {"window": positions},
            {"window": baseline_frames[baseline]["allocation"].to_numpy()},
            {"window": target},
            {"window": np.zeros_like(target)},
            block_length=int(config["block_length"]),
            n_boot=int(config["bootstrap_reps"]),
            seed=seed + 1,
        )
        score_boot_rows.append(
            {
                "baseline": baseline,
                "actual_score_diff": result["actual_fold_mean_diff"],
                "ci_low": result["fold_ci_low"],
                "ci_high": result["fold_ci_high"],
                "p_two": result["fold_p_two_sided"],
            }
        )
    return tests, {"signal_bootstrap": boot, "score_bootstrap": score_boot_rows}


def _plot_regime_ic(rows: list[dict[str, Any]], path: Path) -> None:
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, [r["rank_ic"] for r in rows], width, label="Rank IC")
    ax.bar(x + width / 2, [r["pearson_ic"] for r in rows], width, label="Pearson IC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["regime"] for r in rows])
    ax.set_ylabel("IC")
    ax.set_title("Signal IC by volatility regime")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_regime_quantiles(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for row in rows:
        values = [row[f"q{i}"] for i in range(1, 6)]
        ax.plot(range(1, 6), values, marker="o", label=row["regime"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("prediction quintile")
    ax.set_ylabel("mean target_metric")
    ax.set_title("Q1-Q5 by volatility regime")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_multiplier(rows: list[dict[str, Any]], path: Path) -> None:
    numeric = [row for row in rows if isinstance(row["multiplier"], (int, float))]
    original = next(row for row in rows if row["multiplier"] == "original_switch")
    x = np.asarray([float(row["multiplier"]) for row in numeric])
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.plot(x, [row["score"] for row in numeric], marker="o", color="tab:blue")
    ax1.axhline(
        original["score"],
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="original 600/400 switch",
    )
    ax1.annotate(
        f"original 600/400: {original['score']:.3f}",
        xy=(1.08, original["score"]),
        xycoords=("axes fraction", "data"),
        color="tab:red",
        fontsize=8,
        va="bottom",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("constant multiplier (log scale)")
    ax1.set_ylabel("official score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="lower left", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["mean_allocation"] for row in numeric], marker="s", color="tab:orange")
    ax2.set_ylabel("mean allocation", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Multiplier robustness curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_concentration(daily: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=False)
    for ax, month in zip(axes, ["2026-01", "2026-04"]):
        sub = daily.filter(pl.col("month") == month).sort("date_id")
        ax.bar(range(sub.height), sub["contribution"].to_numpy() * 100.0, color="tab:green", alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(0, sub.height, max(1, sub.height // 5)))
        ax.set_xticklabels([str(v) for v in sub["date_id"].to_numpy()[:: max(1, sub.height // 5)]], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("daily contribution (%)")
        ax.set_title(month)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_loo(loo: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for month, color in [("2026-01", "tab:blue"), ("2026-04", "tab:orange")]:
        sub = loo.filter(pl.col("month") == month).sort("removed_date_id")
        ax.plot(range(sub.height), sub["delta_score"].to_numpy(), marker="o", markersize=3, label=month, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("removed day index within month")
    ax.set_ylabel("delta official score")
    ax.set_title("Leave-one-day-out influence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_post(metrics: pl.DataFrame, path: Path) -> None:
    labels = metrics["mode"].to_list()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x, metrics["official_score_proxy"].to_numpy(), color="tab:purple", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("official score (proxy)")
    ax.set_title("Independent post-window (2026-07-07 to 2026-11-17)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/phase6.yaml")
    parser.add_argument("--official-positions", default="reports/live_window_20251216_20260706/positions.csv")
    parser.add_argument("--post-positions", default="reports/live_window_20260707_20261117/positions.csv")
    parser.add_argument("--panel", default="data/lake/panel_live.parquet")
    parser.add_argument("--output", default="reports/phase6_signal_robustness")
    args = parser.parse_args()

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    phase6 = config.section("phase6")
    if not Path("docs/phase6_preregistration.md").exists():
        raise RuntimeError("Phase 6 preregistration file is missing")

    official = (
        pl.read_csv(args.official_positions)
        .filter(pl.col("mode") == "daily")
        .sort(COL_DATE_ID)
    )
    post = (
        pl.read_csv(args.post_positions)
        .filter(pl.col("mode") == "daily")
        .sort(COL_DATE_ID)
    )
    if official.height != 137 or post.height != 95:
        raise ValueError(f"Expected 137 official / 95 post days, got {official.height}/{post.height}")

    panel = pl.read_parquet(args.panel).sort(COL_DATE_ID)
    dates = panel.select([COL_DATE_ID, "date"]).sort(COL_DATE_ID)
    official = official.join(dates, on=COL_DATE_ID, how="left").with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )
    post = post.join(dates, on=COL_DATE_ID, how="left").with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )
    thresholds = regime_thresholds(panel, int(official[COL_DATE_ID][0]), model_config)

    output = Path(args.output)
    (output / "plots").mkdir(parents=True, exist_ok=True)

    # Experiment 1: signal x regime
    regime = signal_regime_rows(official, thresholds)
    pl.DataFrame(regime).write_csv(output / "signal_regime.csv")
    _plot_regime_ic(regime, output / "plots" / "01_regime_ic.png")
    _plot_regime_quantiles(regime, output / "plots" / "02_regime_quantiles.png")

    # Experiment 2: multiplier robustness
    grid = [float(v) for v in phase6.get("multiplier_grid", [])]
    multiplier = multiplier_rows(official, model_config, grid)
    pl.DataFrame(multiplier).write_csv(output / "multiplier_grid.csv")
    _plot_multiplier(multiplier, output / "plots" / "03_multiplier_curve.png")

    # Experiment 3: concentration
    months = [str(value) for value in phase6.get("concentration_months", ["2026-01", "2026-04"])]
    lengths = [int(value) for value in phase6.get("block_deletion_lengths", [3, 5])]
    daily_contrib, loo, block, concentration_summary = concentration(official, months, lengths)
    pl.DataFrame(daily_contrib).write_csv(output / "concentration_daily.csv")
    pl.DataFrame(loo).write_csv(output / "leave_one_out.csv")
    pl.DataFrame(block).write_csv(output / "block_deletion.csv")
    _plot_concentration(pl.DataFrame(daily_contrib), output / "plots" / "04_concentration.png")
    _plot_loo(pl.DataFrame(loo), output / "plots" / "05_leave_one_out.png")

    # Experiment 5: official-window inference
    official_baselines = {
        mode: (
            pl.read_csv(args.official_positions)
            .filter(pl.col("mode") == mode)
            .sort(COL_DATE_ID)
        )
        for mode in ("buy_hold", "constant_1.25", "vol_only")
    }
    official_tests, official_extra = test_family(
        official,
        official_baselines,
        phase6,
        int(phase6.get("seed", 42)),
    )
    pl.DataFrame(official_tests).write_csv(output / "statistical_tests_official.csv")

    # Experiment 4: independent post-window, same frozen protocol.
    post_baselines = {
        mode: (
            pl.read_csv(args.post_positions)
            .filter(pl.col("mode") == mode)
            .sort(COL_DATE_ID)
        )
        for mode in ("buy_hold", "constant_1.25", "vol_only")
    }
    post_tests, post_extra = test_family(
        post,
        post_baselines,
        phase6,
        int(phase6.get("seed", 42)) + 100,
    )
    pl.DataFrame(post_tests).write_csv(output / "statistical_tests_post.csv")
    post_rows = []
    for mode in ("daily", "fixed_rolling", "offline", "vol_only", "buy_hold", "constant_1.25"):
        sub = pl.read_csv(args.post_positions).filter(pl.col("mode") == mode).sort(COL_DATE_ID)
        metrics = score_positions(sub["allocation"].to_numpy(), sub[COL_TARGET_METRIC].to_numpy())
        post_rows.append(
            {
                "mode": mode,
                "n_days": sub.height,
                "official_score_proxy": float(metrics["score"]),
                "raw_sharpe": float(metrics["sharpe"]),
                "cumulative_return": float(metrics["cumulative_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "turnover": float(metrics["turnover"]),
                "mean_allocation": float(metrics["mean_allocation"]),
            }
        )
    pl.DataFrame(post_rows).write_csv(output / "post_metrics.csv")
    _plot_post(pl.DataFrame(post_rows), output / "plots" / "06_post_metrics.png")
    post_regime = signal_regime_rows(post, thresholds)
    pl.DataFrame(post_regime).write_csv(output / "post_signal_regime.csv")

    manifest = {
        "created_at": utc_now(),
        "phase": "phase6_signal_robustness",
        "preregistration": "docs/phase6_preregistration.md",
        "config": "configs/gdbt/phase6.yaml",
        "official_positions": str(args.official_positions),
        "official_sha256": sha256(Path(args.official_positions)),
        "post_positions": str(args.post_positions),
        "post_sha256": sha256(Path(args.post_positions)),
        "official_n_days": int(official.height),
        "post_n_days": int(post.height),
        "regime_thresholds": list(thresholds),
        "hac_max_lags": int(phase6.get("hac_max_lags", 10)),
        "block_length": int(phase6.get("block_length", 20)),
        "bootstrap_reps": int(phase6.get("bootstrap_reps", 2000)),
        "multiplier_grid": grid,
        "notes": [
            "Post-window run is separate and was not used to choose any parameter.",
            "Official scores are reconstructed with fwd=target_metric and rf=0.",
            "HAC inference uses a fixed 10-lag Newey-West covariance.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "manifest": manifest,
            "signal_regime": regime,
            "multiplier_grid": multiplier,
            "concentration": concentration_summary,
            "official_tests": official_tests,
            "official_extra": official_extra,
            "post_metrics": post_rows,
            "post_tests": post_tests,
            "post_extra": post_extra,
            "post_signal_regime": post_regime,
        },
    )
    print(pl.DataFrame(regime))
    print(pl.DataFrame(post_rows))
    print(pl.DataFrame(official_tests))
    print(pl.DataFrame(post_tests))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
