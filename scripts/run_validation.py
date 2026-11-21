"""GBDT incremental-value validation on the saved six-fold OOF predictions.

This script does not train or tune a model. It only re-scores, reorders or
resamples already-realized OOF predictions and returns.

Sections
--------
1. Temporal dependence: block permutation and circular shift, block-length
   sensitivity, Online vs Offline.
2. Paired block bootstrap: Online vs Buy & Hold, Constant 1.25, Vol-only, plus
   causal exposure-matched baselines.
3. Exposure-matched baselines: lag-1 allocation, expanding-mean exposure,
   volatility-matched exposure, past-only bootstrap.
4. Prediction quality: rank/Pearson IC, quantile returns, regime IC, monotonicity.
5. Allocation attribution: signal/vol/smoothing/clipping ladder, bounds,
   turnover and transaction-cost sensitivity.

Usage:
    python scripts/run_validation.py
    python scripts/run_validation.py --config configs/validation.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from common.config import (
    COL_DATE_ID,
    COL_FWD,
    COL_RF,
    COL_TARGET,
    COL_TARGET_METRIC,
    Config,
)
from common.data import date_ids_for_years, load_date_map, load_panel
from common.experiments import (
    causal_vol_estimates,
    score_positions,
    simulate_positions,
    simulate_vol_target,
)
from common.registry import git_sha, utc_now, write_json
from common.splits import rolling_window_folds
from common.validation import (
    block_permute,
    circular_shift,
    exposure_expanding_mean,
    exposure_lag1,
    exposure_vol_matched,
    moving_block_indices,
    pearson_ic,
    position_stats,
    quantile_returns,
    rank_ic,
    regime_labels,
    regime_thresholds,
    score_with_turnover_cost,
    trailing_volatility,
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


@dataclass
class FoldData:
    fold: Any
    frame: pl.DataFrame
    train_mean: float
    train_vol_median: float
    regime_thresholds: np.ndarray


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
    regime_window: int,
    n_regimes: int,
) -> list[FoldData]:
    labels = panel.select([COL_DATE_ID, COL_TARGET, COL_TARGET_METRIC, COL_FWD, COL_RF])
    joined = oof.join(labels, on=COL_DATE_ID, how="left").sort(COL_DATE_ID)
    if any(joined[column].null_count() for column in (COL_TARGET_METRIC, COL_FWD, COL_RF)):
        raise ValueError("OOF rows could not be joined to labels")

    panel_sorted = panel.sort(COL_DATE_ID)
    trailing = trailing_volatility(
        panel_sorted[COL_TARGET_METRIC].to_numpy(), window=regime_window
    )
    panel_index = {
        int(date_id): i
        for i, date_id in enumerate(panel_sorted[COL_DATE_ID].to_list())
    }

    out: list[FoldData] = []
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
        train_trailing = np.asarray(
            [trailing[panel_index[int(date_id)]] for date_id in fold.train_dates],
            dtype=float,
        )
        thresholds = regime_thresholds(train_trailing, n_regimes=n_regimes)
        test_trailing = np.asarray(
            [trailing[panel_index[int(date_id)]] for date_id in fold.test_dates],
            dtype=float,
        )
        regimes = regime_labels(test_trailing, thresholds)
        frame = frame.with_columns(pl.Series("regime", regimes))
        out.append(
            FoldData(
                fold=fold,
                frame=frame,
                train_mean=train_mean,
                train_vol_median=float(np.median(train_vol)),
                regime_thresholds=thresholds,
            )
        )
    return out


def score_frame(frame: pl.DataFrame, positions: np.ndarray) -> dict[str, Any]:
    return score_positions(
        positions,
        frame[COL_FWD].to_numpy(),
        frame[COL_RF].to_numpy(),
    )


def actual_fold_scores(items: list[FoldData], key: str) -> list[float]:
    return [
        float(score_frame(item.frame, item.frame[key].to_numpy())["score"])
        for item in items
    ]


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, values[idx] * (len(values) - rank))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# 1. Temporal dependence
# ---------------------------------------------------------------------------

def temporal_tests(
    fold_data: dict[str, list[FoldData]],
    model_config: GBDTConfig,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    block_lengths = [int(v) for v in config["block_lengths"]]
    primary = int(config["primary_block_length"])
    reps = int(config["temporal_reps"])
    summary: list[dict[str, Any]] = []
    folds_out: list[dict[str, Any]] = []
    replicates: list[dict[str, Any]] = []

    for mode, items in fold_data.items():
        actual = actual_fold_scores(items, "allocation")
        actual_mean = float(np.mean(actual))
        mode_offset = 0 if mode == "offline" else 1_000_000

        for block_length in block_lengths:
            null_means: list[float] = []
            fold_null: dict[str, list[float]] = {item.fold.name: [] for item in items}
            for rep in range(reps):
                fold_scores: list[float] = []
                for fold_index, item in enumerate(items):
                    rng = np.random.default_rng(
                        seed
                        + rep * 200_003
                        + fold_index * 1_009
                        + block_length * 17
                        + mode_offset
                    )
                    raw = block_permute(
                        item.frame["raw_pred"].to_numpy(), block_length, rng
                    )
                    positions = simulate_positions(
                        raw,
                        item.frame["vol"],
                        item.frame["multiplier"],
                        smoothing=model_config.smoothing,
                        transaction_cost=model_config.transaction_cost,
                        vol_scaling=model_config.vol_scaling,
                    )
                    score = float(score_frame(item.frame, positions)["score"])
                    fold_scores.append(score)
                    fold_null[item.fold.name].append(score)
                null_means.append(float(np.mean(fold_scores)))
            array = np.asarray(null_means)
            p_one = float(
                (1 + np.sum(array >= actual_mean)) / (len(array) + 1)
            ) if actual_mean >= 0 else float(
                (1 + np.sum(array <= actual_mean)) / (len(array) + 1)
            )
            summary.append(
                {
                    "mode": mode,
                    "test": "block_permutation",
                    "block_length": block_length,
                    "actual_mean_score": actual_mean,
                    "null_mean_score": float(np.mean(array)),
                    "null_std_score": float(np.std(array, ddof=1)),
                    "null_p05": float(np.percentile(array, 5)),
                    "null_p95": float(np.percentile(array, 95)),
                    "p_value": p_one,
                    "n_reps": reps,
                }
            )
            if block_length == primary:
                for item in items:
                    values = np.asarray(fold_null[item.fold.name], dtype=float)
                    null_mean = float(np.mean(values))
                    fold_p = float(
                        (1 + np.sum(values >= actual[items.index(item)]))
                        / (len(values) + 1)
                    )
                    folds_out.append(
                        {
                            "mode": mode,
                            "test": "block_permutation",
                            "block_length": block_length,
                            "fold": item.fold.name,
                            "actual_score": actual[items.index(item)],
                            "null_mean_score": null_mean,
                            "null_p05": float(np.percentile(values, 5)),
                            "null_p95": float(np.percentile(values, 95)),
                            "p_value": fold_p,
                        }
                    )
                replicates.extend(
                    {
                        "mode": mode,
                        "test": "block_permutation",
                        "block_length": block_length,
                        "replicate": i,
                        "mean_score": value,
                    }
                    for i, value in enumerate(array)
                )

        # Circular shifts preserve the whole sequence but destroy phase.
        circular_means: list[float] = []
        circular_fold: dict[str, list[float]] = {item.fold.name: [] for item in items}
        for rep in range(reps):
            fold_scores = []
            for fold_index, item in enumerate(items):
                rng = np.random.default_rng(
                    seed + rep * 300_007 + fold_index * 1_009 + mode_offset + 7
                )
                n = item.frame.height
                shift = int(rng.integers(1, n))
                raw = circular_shift(item.frame["raw_pred"].to_numpy(), shift)
                positions = simulate_positions(
                    raw,
                    item.frame["vol"],
                    item.frame["multiplier"],
                    smoothing=model_config.smoothing,
                    transaction_cost=model_config.transaction_cost,
                    vol_scaling=model_config.vol_scaling,
                )
                score = float(score_frame(item.frame, positions)["score"])
                fold_scores.append(score)
                circular_fold[item.fold.name].append(score)
            circular_means.append(float(np.mean(fold_scores)))
        array = np.asarray(circular_means)
        p_circular = float(
            (1 + np.sum(array >= actual_mean)) / (len(array) + 1)
        ) if actual_mean >= 0 else float(
            (1 + np.sum(array <= actual_mean)) / (len(array) + 1)
        )
        summary.append(
            {
                "mode": mode,
                "test": "circular_shift",
                "block_length": 0,
                "actual_mean_score": actual_mean,
                "null_mean_score": float(np.mean(array)),
                "null_std_score": float(np.std(array, ddof=1)),
                "null_p05": float(np.percentile(array, 5)),
                "null_p95": float(np.percentile(array, 95)),
                "p_value": p_circular,
                "n_reps": reps,
            }
        )
        for item in items:
            values = np.asarray(circular_fold[item.fold.name], dtype=float)
            actual_fold = actual[items.index(item)]
            fold_p = float(
                (1 + np.sum(values >= actual_fold)) / (len(values) + 1)
            ) if actual_fold >= 0 else float(
                (1 + np.sum(values <= actual_fold)) / (len(values) + 1)
            )
            folds_out.append(
                {
                    "mode": mode,
                    "test": "circular_shift",
                    "block_length": 0,
                    "fold": item.fold.name,
                    "actual_score": actual_fold,
                    "null_mean_score": float(np.mean(values)),
                    "null_p05": float(np.percentile(values, 5)),
                    "null_p95": float(np.percentile(values, 95)),
                    "p_value": fold_p,
                }
            )
        replicates.extend(
            {
                "mode": mode,
                "test": "circular_shift",
                "block_length": 0,
                "replicate": i,
                "mean_score": value,
            }
            for i, value in enumerate(array)
        )

    # Holm adjustment across block lengths within each mode/test.
    for mode in fold_data:
        indices = [
            i
            for i, row in enumerate(summary)
            if row["mode"] == mode and row["test"] == "block_permutation"
        ]
        adjusted = holm_adjust([summary[i]["p_value"] for i in indices])
        for i, value in zip(indices, adjusted):
            summary[i]["p_value_holm"] = value
        for i, row in enumerate(summary):
            row.setdefault("p_value_holm", float("nan"))
    return summary, folds_out, replicates


# ---------------------------------------------------------------------------
# 2. Paired block bootstrap
# ---------------------------------------------------------------------------

def build_strategies(
    items: list[FoldData],
    model_config: GBDTConfig,
) -> dict[str, list[np.ndarray]]:
    strategies: dict[str, list[np.ndarray]] = {
        "online": [],
        "buy_hold": [],
        "constant_1.25": [],
        "vol_only_1.0": [],
        "lag1": [],
        "expanding_mean": [],
        "vol_matched": [],
    }
    for item in items:
        frame = item.frame
        online = frame["allocation"].to_numpy()
        strategies["online"].append(online)
        strategies["buy_hold"].append(np.ones(frame.height))
        strategies["constant_1.25"].append(np.full(frame.height, 1.25))
        strategies["vol_only_1.0"].append(
            simulate_vol_target(
                frame["vol"],
                item.train_vol_median,
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
            )
        )
        strategies["lag1"].append(exposure_lag1(online))
        strategies["expanding_mean"].append(exposure_expanding_mean(online))
        strategies["vol_matched"].append(
            exposure_vol_matched(online, frame["vol"], item.train_vol_median)
        )
    return strategies


def score_concatenated(
    strategies: dict[str, list[np.ndarray]],
    items: list[FoldData],
) -> dict[str, float]:
    result = {}
    for name, arrays in strategies.items():
        positions = np.concatenate(arrays)
        fwd = np.concatenate([item.frame[COL_FWD].to_numpy() for item in items])
        rf = np.concatenate([item.frame[COL_RF].to_numpy() for item in items])
        result[name] = float(score_positions(positions, fwd, rf)["score"])
    return result


def score_fold_mean(
    strategies: dict[str, list[np.ndarray]],
    items: list[FoldData],
) -> dict[str, float]:
    result = {}
    for name, arrays in strategies.items():
        scores = [
            float(score_frame(item.frame, positions)["score"])
            for item, positions in zip(items, arrays)
        ]
        result[name] = float(np.mean(scores))
    return result


def bootstrap_scores(
    strategies: dict[str, list[np.ndarray]],
    items: list[FoldData],
    level: str,
    block_length: int,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    names = list(strategies)
    n_folds = len(items)
    if level == "concatenated":
        out = {name: np.empty(n_boot) for name in names}
    else:
        out = {name: np.empty((n_boot, n_folds)) for name in names}

    for b in range(n_boot):
        rng = np.random.default_rng(seed + b * 10_007)
        if level == "concatenated":
            positions_parts = {name: [] for name in names}
            fwd_parts = []
            rf_parts = []
            for fold_index, item in enumerate(items):
                indices = moving_block_indices(item.frame.height, block_length, rng)
                fwd_parts.append(item.frame[COL_FWD].to_numpy()[indices])
                rf_parts.append(item.frame[COL_RF].to_numpy()[indices])
                for name in names:
                    positions_parts[name].append(strategies[name][fold_index][indices])
            fwd = np.concatenate(fwd_parts)
            rf = np.concatenate(rf_parts)
            for name in names:
                positions = np.concatenate(positions_parts[name])
                out[name][b] = score_positions(positions, fwd, rf)["score"]
        else:
            for fold_index, item in enumerate(items):
                indices = moving_block_indices(item.frame.height, block_length, rng)
                fwd = item.frame[COL_FWD].to_numpy()[indices]
                rf = item.frame[COL_RF].to_numpy()[indices]
                for name in names:
                    out[name][b, fold_index] = score_positions(
                        strategies[name][fold_index][indices], fwd, rf
                    )["score"]
    return out


def paired_summary(
    actual_online: float,
    actual_base: float,
    boot_online: np.ndarray,
    boot_base: np.ndarray,
    n_boot: int,
) -> dict[str, Any]:
    diffs = boot_online - boot_base
    actual = actual_online - actual_base
    low, high = percentile_ci(diffs)
    p_two = min(1.0, 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))))
    p_one = (
        float(np.mean(diffs <= 0)) if actual >= 0 else float(np.mean(diffs >= 0))
    )
    return {
        "actual_online": float(actual_online),
        "actual_baseline": float(actual_base),
        "actual_diff": float(actual),
        "ci_low": low,
        "ci_high": high,
        "p_two_sided": float(min(1.0, p_two)),
        "p_one_sided": float(min(1.0, p_one)),
        "significant_5pct": bool(low > 0 or high < 0),
        "n_boot": n_boot,
    }


def paired_tests(
    fold_data: dict[str, list[FoldData]],
    model_config: GBDTConfig,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    items = fold_data["online"]
    strategies = build_strategies(items, model_config)
    baselines = list(config["primary_comparisons"]) + list(config["exposure_comparisons"])
    blocks = [
        "buy_hold",
        "constant_1.25",
        "vol_only_1.0",
        "lag1",
        "expanding_mean",
        "vol_matched",
    ]
    rows: list[dict[str, Any]] = []

    actual_concat = score_concatenated(strategies, items)
    actual_fold_mean = score_fold_mean(strategies, items)
    boot_concat = bootstrap_scores(
        strategies, items, "concatenated", int(config["primary_block_length"]),
        int(config["bootstrap_reps"]), seed + 11,
    )
    boot_fold = bootstrap_scores(
        strategies, items, "fold", int(config["primary_block_length"]),
        int(config["bootstrap_reps"]), seed + 12,
    )
    boot_fold_mean = {name: values.mean(axis=1) for name, values in boot_fold.items()}

    for block in blocks:
        if block not in baselines:
            continue
        group = "primary" if block in config["primary_comparisons"] else "exposure"
        for level, actual, boot in [
            ("concatenated", actual_concat, boot_concat),
            ("fold_mean", actual_fold_mean, boot_fold_mean),
        ]:
            summary = paired_summary(
                actual["online"], actual[block], boot["online"], boot[block],
                int(config["bootstrap_reps"]),
            )
            rows.append(
                {
                    "comparison": f"online_vs_{block}",
                    "baseline": block,
                    "group": group,
                    "level": level,
                    "block_length": int(config["primary_block_length"]),
                    **summary,
                }
            )

    # Per-fold paired differences.
    for block in blocks:
        if block not in baselines:
            continue
        group = "primary" if block in config["primary_comparisons"] else "exposure"
        for fold_index, item in enumerate(items):
            summary = paired_summary(
                float(score_frame(item.frame, strategies["online"][fold_index])["score"]),
                float(score_frame(item.frame, strategies[block][fold_index])["score"]),
                boot_fold["online"][:, fold_index],
                boot_fold[block][:, fold_index],
                int(config["bootstrap_reps"]),
            )
            rows.append(
                {
                    "comparison": f"online_vs_{block}",
                    "baseline": block,
                    "group": group,
                    "level": f"fold_{item.fold.name}",
                    "block_length": int(config["primary_block_length"]),
                    **summary,
                }
            )

    # Block-length sensitivity for the six main comparisons.
    for block in blocks:
        group = "primary" if block in config["primary_comparisons"] else "exposure"
        for block_length in [10, 20, 40]:
            boot_concat_l = bootstrap_scores(
                strategies, items, "concatenated", block_length,
                int(config["bootstrap_sensitivity_reps"]), seed + 20 + block_length,
            )
            boot_fold_l = bootstrap_scores(
                strategies, items, "fold", block_length,
                int(config["bootstrap_sensitivity_reps"]), seed + 30 + block_length,
            )
            boot_fold_l_mean = {
                name: values.mean(axis=1) for name, values in boot_fold_l.items()
            }
            for level, actual, boot in [
                ("concatenated", actual_concat, boot_concat_l),
                ("fold_mean", actual_fold_mean, boot_fold_l_mean),
            ]:
                summary = paired_summary(
                    actual["online"], actual[block], boot["online"], boot[block],
                    int(config["bootstrap_sensitivity_reps"]),
                )
                rows.append(
                    {
                        "comparison": f"online_vs_{block}",
                        "baseline": block,
                        "group": group,
                        "level": f"sensitivity_{level}",
                        "block_length": block_length,
                        **summary,
                    }
                )

    # Holm correction for the primary and exposure groups separately, at the
    # primary block length and each level.
    for group in ("primary", "exposure"):
        for level in ("concatenated", "fold_mean"):
            indices = [
                i
                for i, row in enumerate(rows)
                if row["group"] == group
                and row["level"] == level
                and row["block_length"] == int(config["primary_block_length"])
            ]
            adjusted = holm_adjust([rows[i]["p_two_sided"] for i in indices])
            for i, value in zip(indices, adjusted):
                rows[i]["p_two_holm"] = value
    for row in rows:
        row.setdefault("p_two_holm", float("nan"))
    return rows


# ---------------------------------------------------------------------------
# 3. Exposure-matched baselines
# ---------------------------------------------------------------------------

def exposure_tests(
    fold_data: dict[str, list[FoldData]],
    model_config: GBDTConfig,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = fold_data["online"]
    strategies = build_strategies(items, model_config)
    names = ["online", "lag1", "expanding_mean", "vol_matched"]
    rows: list[dict[str, Any]] = []
    for name in names:
        positions = np.concatenate(strategies[name])
        stats = position_stats(positions)
        concat_score = score_concatenated(strategies, items)[name]
        fold_mean = score_fold_mean(strategies, items)[name]
        rows.append(
            {
                "baseline": name,
                **stats,
                "score_mean": fold_mean,
                "score_continuous": concat_score,
            }
        )

    # Causal distribution-matched bootstrap: each day samples an allocation
    # observed strictly earlier in the same fold.
    n_paths = int(config["exposure_paths"])
    path_means = []
    for path in range(n_paths):
        fold_scores = []
        for fold_index, item in enumerate(items):
            rng = np.random.default_rng(seed + path * 40_009 + fold_index * 1_009)
            online = strategies["online"][fold_index]
            positions = np.zeros_like(online)
            for t in range(1, len(online)):
                positions[t] = online[rng.integers(0, t)]
            fold_scores.append(float(score_frame(item.frame, positions)["score"]))
        path_means.append(float(np.mean(fold_scores)))
    path_means_arr = np.asarray(path_means)
    actual_mean = score_fold_mean(strategies, items)["online"]
    p = float(
        (1 + np.sum(path_means_arr >= actual_mean)) / (len(path_means_arr) + 1)
    ) if actual_mean >= 0 else float(
        (1 + np.sum(path_means_arr <= actual_mean)) / (len(path_means_arr) + 1)
    )
    rows.append(
        {
            "baseline": "past_bootstrap",
            "mean": float(np.mean(path_means_arr)),
            "median": float(np.median(path_means_arr)),
            "std": float(np.std(path_means_arr, ddof=1)),
            "score_mean": float(np.mean(path_means_arr)),
            "score_continuous": float("nan"),
            "p_value_vs_online": p,
            "n_paths": n_paths,
        }
    )
    return rows, [{"path": i, "mean_score": value} for i, value in enumerate(path_means)]


# ---------------------------------------------------------------------------
# 4. Prediction quality
# ---------------------------------------------------------------------------

def prediction_quality(
    fold_data: dict[str, list[FoldData]],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    q = int(config["quantiles"])
    quality: list[dict[str, Any]] = []
    quantiles: list[dict[str, Any]] = []
    ic_summary: list[dict[str, Any]] = []
    monotonicity: list[dict[str, Any]] = []

    for mode, items in fold_data.items():
        fold_ics: list[float] = []
        for item in items:
            frame = item.frame
            pred = frame["raw_pred"].to_numpy()
            target = frame[COL_TARGET_METRIC].to_numpy()
            ric = rank_ic(pred, target)
            pic = pearson_ic(pred, target)
            fold_ics.append(ric)
            quality.append(
                {
                    "mode": mode,
                    "fold": item.fold.name,
                    "regime": "all",
                    "n": len(pred),
                    "rank_ic": ric,
                    "pearson_ic": pic,
                    "mean_target_metric": float(np.mean(target)),
                }
            )
            for regime in range(int(config["n_regimes"])):
                mask = frame["regime"].to_numpy() == regime
                if mask.sum() >= 20:
                    quality.append(
                        {
                            "mode": mode,
                            "fold": item.fold.name,
                            "regime": f"regime_{regime}",
                            "n": int(mask.sum()),
                            "rank_ic": rank_ic(pred[mask], target[mask]),
                            "pearson_ic": pearson_ic(pred[mask], target[mask]),
                            "mean_target_metric": float(np.mean(target[mask])),
                        }
                    )
            qrows = quantile_returns(pred, target, q)
            for row in qrows:
                quantiles.append({"mode": mode, "fold": item.fold.name, **row})
            means = [row["mean_target"] for row in qrows]
            if len(means) >= 2:
                rho = float(spearmanr(np.arange(1, len(means) + 1), means).statistic)
                increasing = sum(
                    1 for i in range(len(means) - 1) if means[i + 1] > means[i]
                )
                monotonicity.append(
                    {
                        "mode": mode,
                        "fold": item.fold.name,
                        "quantile_spearman": rho,
                        "n_increasing": int(increasing),
                        "monotonic": bool(increasing == len(means) - 1),
                        "top_bottom": float(means[-1] - means[0]),
                    }
                )

        # Block bootstrap of the mean rank IC.
        reps = int(config["bootstrap_sensitivity_reps"])
        ics = []
        for rep in range(reps):
            fold_values = []
            rng = np.random.default_rng(seed + rep * 50_009 + (0 if mode == "offline" else 7_000_000))
            for item in items:
                idx = moving_block_indices(
                    item.frame.height, int(config["primary_block_length"]), rng
                )
                fold_values.append(
                    rank_ic(
                        item.frame["raw_pred"].to_numpy()[idx],
                        item.frame[COL_TARGET_METRIC].to_numpy()[idx],
                    )
                )
            ics.append(float(np.nanmean(fold_values)))
        ic_ci = np.asarray(ics)
        low, high = percentile_ci(ic_ci)
        mean_ic = float(np.mean(fold_ics))
        ic_summary.append(
            {
                "mode": mode,
                "scope": "all",
                "mean_rank_ic": mean_ic,
                "median_rank_ic": float(np.median(fold_ics)),
                "std_rank_ic": float(np.std(fold_ics, ddof=1)),
                "positive_folds": int(np.sum(np.asarray(fold_ics) > 0)),
                "n_folds": len(fold_ics),
                "ci_low": low,
                "ci_high": high,
                "p_two_sided": float(
                    min(1.0, 2.0 * min(float(np.mean(ic_ci <= 0)), float(np.mean(ic_ci >= 0))))
                ),
                "status": "inconclusive" if low <= 0 <= high else "directional",
            }
        )

        # Per-regime fold average (no bootstrap; small n).
        for regime in range(int(config["n_regimes"])):
            vals = [
                row["rank_ic"]
                for row in quality
                if row["mode"] == mode
                and row["regime"] == f"regime_{regime}"
                and np.isfinite(row["rank_ic"])
            ]
            if vals:
                ic_summary.append(
                    {
                        "mode": mode,
                        "scope": f"regime_{regime}",
                        "mean_rank_ic": float(np.mean(vals)),
                        "median_rank_ic": float(np.median(vals)),
                        "std_rank_ic": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "positive_folds": int(np.sum(np.asarray(vals) > 0)),
                        "n_folds": len(vals),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "p_two_sided": float("nan"),
                        "status": "inconclusive" if len(vals) < 6 or 0 < sum(v > 0 for v in vals) < len(vals) else "directional",
                    }
                )
    return quality, quantiles, ic_summary, monotonicity


# ---------------------------------------------------------------------------
# 5. Allocation attribution
# ---------------------------------------------------------------------------

def attribution_tests(
    fold_data: dict[str, list[FoldData]],
    model_config: GBDTConfig,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    variants: list[dict[str, Any]] = []
    bounds: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    turnover_costs: list[dict[str, Any]] = []
    for mode, items in fold_data.items():
        for item in items:
            frame = item.frame
            raw = frame["raw_pred"].to_numpy()
            vol = frame["vol"].to_numpy()
            multiplier = frame["multiplier"].to_numpy()
            ones_vol = np.ones_like(vol)
            ones_mult = np.ones_like(multiplier)

            def positions_for(kind: str, transaction_cost: float | None = None) -> np.ndarray:
                cost = model_config.transaction_cost if transaction_cost is None else transaction_cost
                if kind == "full":
                    return simulate_positions(raw, vol, multiplier, smoothing=model_config.smoothing, transaction_cost=cost, vol_scaling=model_config.vol_scaling)
                if kind == "no_smoothing":
                    return simulate_positions(raw, vol, multiplier, smoothing=0.0, transaction_cost=cost, vol_scaling=model_config.vol_scaling)
                if kind == "no_vol":
                    return simulate_positions(raw, ones_vol, multiplier, smoothing=model_config.smoothing, transaction_cost=cost, vol_scaling=1.0)
                if kind == "no_multiplier":
                    return simulate_positions(raw, vol, ones_mult, smoothing=model_config.smoothing, transaction_cost=cost, vol_scaling=model_config.vol_scaling)
                if kind == "signal_only":
                    return simulate_positions(raw, ones_vol, multiplier, smoothing=0.0, transaction_cost=0.0, vol_scaling=1.0)
                if kind == "signal_vol":
                    return simulate_positions(raw, vol, multiplier, smoothing=0.0, transaction_cost=0.0, vol_scaling=model_config.vol_scaling)
                if kind == "signal_vol_smooth":
                    return simulate_positions(raw, vol, multiplier, smoothing=model_config.smoothing, transaction_cost=0.0, vol_scaling=model_config.vol_scaling)
                if kind == "vol_only":
                    return simulate_vol_target(vol, item.train_vol_median, smoothing=model_config.smoothing, transaction_cost=cost)
                raise ValueError(kind)

            for kind in [
                "full",
                "no_smoothing",
                "no_vol",
                "no_multiplier",
                "signal_only",
                "signal_vol",
                "signal_vol_smooth",
                "vol_only",
            ]:
                pos = positions_for(kind)
                metrics = score_frame(frame, pos)
                variants.append(
                    {
                        "mode": mode,
                        "fold": item.fold.name,
                        "variant": kind,
                        "score": metrics["score"],
                        "sharpe": metrics["sharpe"],
                        "mean_allocation": metrics["mean_allocation"],
                        "turnover": metrics["turnover"],
                        "n_days": metrics["n_days"],
                    }
                )

            pre = raw * multiplier / (np.maximum(vol, 1e-12) * model_config.vol_scaling)
            final = frame["allocation"].to_numpy()
            bounds.append(
                {
                    "mode": mode,
                    "fold": item.fold.name,
                    "pre_clip_below_zero": float(np.mean(pre < 0.0)),
                    "pre_clip_above_two": float(np.mean(pre > 2.0)),
                    "pre_clip_inside": float(np.mean((pre >= 0.0) & (pre <= 2.0))),
                    "final_zero_share": float(np.mean(final <= 1e-9)),
                    "final_near_zero_share": float(np.mean(final <= 0.01)),
                    "final_two_share": float(np.mean(final >= 2.0 - 1e-9)),
                    "final_near_two_share": float(np.mean(final >= 1.99)),
                    "final_mean": float(np.mean(final)),
                    "final_median": float(np.median(final)),
                }
            )

            for cost in config["cost_grid"]:
                pos = positions_for("full", transaction_cost=float(cost))
                metrics = score_frame(frame, pos)
                costs.append(
                    {
                        "mode": mode,
                        "fold": item.fold.name,
                        "transaction_cost": float(cost),
                        "score": metrics["score"],
                        "sharpe": metrics["sharpe"],
                        "mean_allocation": metrics["mean_allocation"],
                        "turnover": metrics["turnover"],
                    }
                )

            for cost_bps in config["turnover_cost_bps"]:
                metrics = score_with_turnover_cost(
                    frame["allocation"].to_numpy(),
                    frame[COL_FWD].to_numpy(),
                    frame[COL_RF].to_numpy(),
                    float(cost_bps),
                )
                turnover_costs.append(
                    {
                        "mode": mode,
                        "fold": item.fold.name,
                        "cost_bps": float(cost_bps),
                        "score": metrics["score"],
                        "sharpe": metrics["sharpe"],
                        "mean_allocation": float(frame["allocation"].mean()),
                        "turnover": float(position_stats(frame["allocation"].to_numpy())["turnover"]),
                    }
                )
    return variants, bounds, costs, turnover_costs


# ---------------------------------------------------------------------------
# Records and plots
# ---------------------------------------------------------------------------

def write_records(
    output: Path,
    config: Config,
    validation: dict[str, Any],
    config_path: str,
    offline_path: Path,
    online_path: Path,
    fold_data: dict[str, list[FoldData]],
    temporal_summary: list[dict[str, Any]],
    temporal_folds: list[dict[str, Any]],
    temporal_reps: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    exposure: list[dict[str, Any]],
    exposure_paths: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    quantiles: list[dict[str, Any]],
    ic_summary: list[dict[str, Any]],
    monotonicity: list[dict[str, Any]],
    attribution: list[dict[str, Any]],
    bounds: list[dict[str, Any]],
    costs: list[dict[str, Any]],
    turnover_costs: list[dict[str, Any]],
    parity: dict[str, float],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output / "config_used.yaml")
    tables = {
        "temporal_summary": temporal_summary,
        "temporal_folds": temporal_folds,
        "temporal_replicates": temporal_reps,
        "paired_comparison": paired,
        "exposure_matched": exposure,
        "exposure_past_bootstrap": exposure_paths,
        "prediction_quality": quality,
        "prediction_quantiles": quantiles,
        "prediction_ic_summary": ic_summary,
        "quantile_monotonicity": monotonicity,
        "allocation_attribution": attribution,
        "position_bounds": bounds,
        "cost_sensitivity": costs,
        "turnover_cost_sensitivity": turnover_costs,
    }
    for name, rows in tables.items():
        if rows:
            pl.DataFrame(rows).write_csv(output / f"{name}.csv")

    manifest = {
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "config": str(validation),
        "offline_oof": str(offline_path),
        "online_oof": str(online_path),
        "folds": {
            item.fold.name: {
                "train": [int(item.fold.train_dates[0]), int(item.fold.train_dates[-1])],
                "test": [int(item.fold.test_dates[0]), int(item.fold.test_dates[-1])],
                "n_train": len(item.fold.train_dates),
                "n_test": len(item.fold.test_dates),
            }
            for item in fold_data["online"]
        },
        "seed": int(validation["seed"]),
        "block_lengths": validation["block_lengths"],
        "primary_block_length": validation["primary_block_length"],
        "bootstrap_reps": validation["bootstrap_reps"],
        "temporal_reps": validation["temporal_reps"],
        "exposure_paths": validation["exposure_paths"],
        "cash_score_convention": "cash -> score=sharpe=0.0",
        "no_lookahead_notes": [
            "No model was retrained or tuned on OOF results.",
            "Temporal nulls permute/shift realized test predictions only.",
            "Exposure baselines use only allocations observed strictly earlier in the fold.",
            "Regime thresholds come from the training window's trailing volatility.",
            "Block bootstrap is used for uncertainty only; it does not select a model.",
        ],
        "oof_parity_max_abs_diff": parity,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "manifest": manifest,
            "temporal": temporal_summary,
            "paired": paired,
            "exposure": exposure,
            "prediction_ic": ic_summary,
            "monotonicity": monotonicity,
        },
    )
    return manifest


def plot_all(
    output: Path,
    fold_data: dict[str, list[FoldData]],
    temporal_summary: list[dict[str, Any]],
    temporal_reps: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    exposure: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    quantiles: list[dict[str, Any]],
    ic_summary: list[dict[str, Any]],
    monotonicity: list[dict[str, Any]],
    attribution: list[dict[str, Any]],
    bounds: list[dict[str, Any]],
    costs: list[dict[str, Any]],
    turnover_costs: list[dict[str, Any]],
    strategies: dict[str, list[np.ndarray]],
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 01 block sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mode in zip(axes, ["offline", "online"]):
        rows = [r for r in temporal_summary if r["mode"] == mode and r["test"] == "block_permutation"]
        rows.sort(key=lambda r: r["block_length"])
        x = [r["block_length"] for r in rows]
        ax.plot(x, [r["null_mean_score"] for r in rows], marker="o", label="null mean")
        ax.fill_between(
            x,
            [r["null_p05"] for r in rows],
            [r["null_p95"] for r in rows],
            alpha=0.2,
            label="null 5-95%",
        )
        actual = rows[0]["actual_mean_score"]
        ax.axhline(actual, color="red", linestyle="--", label="actual")
        for r in rows:
            ax.annotate(f"p={r['p_value']:.2f}", (r["block_length"], r["null_mean_score"]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.set_title(mode)
        ax.set_xlabel("block length")
        ax.set_ylabel("mean official score")
    axes[0].legend(fontsize=7)
    fig.suptitle("Block-permutation sensitivity")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_temporal_block_sensitivity.png")
    plt.close(fig)

    # 02 null distributions: primary block and circular
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mode in zip(axes, ["offline", "online"]):
        block = [r["mean_score"] for r in temporal_reps if r["mode"] == mode and r["test"] == "block_permutation"]
        circular = [r["mean_score"] for r in temporal_reps if r["mode"] == mode and r["test"] == "circular_shift"]
        ax.hist(block, bins=35, alpha=0.45, label="block permutation", color="tab:blue")
        ax.hist(circular, bins=35, alpha=0.45, label="circular shift", color="tab:orange")
        actual = next(
            r["actual_mean_score"]
            for r in temporal_summary
            if r["mode"] == mode and r["test"] == "block_permutation"
        )
        ax.axvline(actual, color="red", linestyle="--", label="actual")
        ax.set_title(mode)
        ax.set_xlabel("mean official score")
        ax.set_ylabel("replicates")
    axes[0].legend(fontsize=7)
    fig.suptitle("Temporal null distributions (block length = 20)")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_temporal_null_distributions.png")
    plt.close(fig)

    # 03 exposure-matched scores
    exposure_plot = [r for r in exposure if r["baseline"] in ("online", "lag1", "expanding_mean", "vol_matched")]
    labels = [r["baseline"] for r in exposure_plot]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    x = np.arange(len(labels))
    ax.bar(x - width / 2, [r["score_mean"] for r in exposure_plot], width, label="mean fold score")
    ax.bar(x + width / 2, [r["score_continuous"] for r in exposure_plot], width, label="continuous OOF score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("score")
    ax.set_title("Exposure-matched baselines")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_exposure_matched_scores.png")
    plt.close(fig)

    # 04 exposure distributions
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, color in [("online", "tab:blue"), ("expanding_mean", "tab:orange"), ("vol_matched", "tab:green")]:
        values = np.concatenate(strategies[name])
        ax.hist(values, bins=40, alpha=0.4, density=True, label=name, color=color)
    ax.set_xlabel("allocation")
    ax.set_ylabel("density")
    ax.set_title("Allocation distribution under exposure matching")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_exposure_distributions.png")
    plt.close(fig)

    # 05 prediction quantiles
    fig, ax = plt.subplots(figsize=(9, 4.2))
    modes = ["offline", "online"]
    width = 0.38
    for i, mode in enumerate(modes):
        rows = [r for r in quantiles if r["mode"] == mode]
        means = []
        for q in range(1, 6):
            values = [r["mean_target"] for r in rows if r["quantile"] == q]
            means.append(float(np.mean(values)))
        ax.bar(np.arange(1, 6) + (i - 0.5) * width, np.asarray(means) * 1e4, width, label=mode)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("prediction quintile (1=lowest)")
    ax.set_ylabel("mean forward excess return (bps)")
    ax.set_title("Prediction quantile returns")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_prediction_quantiles.png")
    plt.close(fig)

    # 06 IC by regime heatmap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    folds = [item.fold.name for item in fold_data["online"]]
    im = None
    for ax, mode in zip(axes, modes):
        matrix = np.full((len(folds), 3), np.nan)
        for i, fold in enumerate(folds):
            for j in range(3):
                row = next(
                    (r for r in ic_summary if r["mode"] == mode and r["scope"] == f"regime_{j}"),
                    None,
                )
                # Use per-fold quality rows for the heatmap.
                qrow = next(
                    (r for r in quality if r["mode"] == mode and r["fold"] == fold and r["regime"] == f"regime_{j}"),
                    None,
                )
                if qrow:
                    matrix[i, j] = qrow["rank_ic"]
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=0.25)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["low vol", "mid vol", "high vol"])
        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels(folds)
        ax.set_title(mode)
        for i in range(len(folds)):
            for j in range(3):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="rank IC")
    fig.suptitle("Rank IC by fold and market-vol regime")
    fig.tight_layout()
    fig.savefig(plot_dir / "06_ic_by_regime.png")
    plt.close(fig)

    # 07 paired forest
    plot_rows = [r for r in paired if r["level"] in ("concatenated", "fold_mean") and r["block_length"] == 20]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    for i, r in enumerate(plot_rows):
        color = "tab:blue" if r["group"] == "primary" else "tab:orange"
        ax.errorbar(
            r["actual_diff"],
            i,
            xerr=[[r["actual_diff"] - r["ci_low"]], [r["ci_high"] - r["actual_diff"]]],
            fmt="o",
            color=color,
            capsize=3,
        )
        labels.append(f"{r['baseline']} [{r['level']}]")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("score difference (Online - baseline)")
    ax.set_title("Paired block bootstrap, 95% CI (block=20)")
    fig.tight_layout()
    fig.savefig(plot_dir / "07_paired_forest.png")
    plt.close(fig)

    # 08 paired by fold, primary comparisons
    primary = [b for b in ("buy_hold", "constant_1.25", "vol_only_1.0")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, baseline in zip(axes, primary):
        rows = [r for r in paired if r["baseline"] == baseline and r["level"].startswith("fold_")]
        rows.sort(key=lambda r: r["level"])
        x = np.arange(len(rows))
        ax.bar(x, [r["actual_diff"] for r in rows], color="tab:blue", alpha=0.7)
        ax.errorbar(
            x,
            [r["actual_diff"] for r in rows],
            yerr=[
                [r["actual_diff"] - r["ci_low"] for r in rows],
                [r["ci_high"] - r["actual_diff"] for r in rows],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["level"].replace("fold_", "") for r in rows])
        ax.set_title(f"Online - {baseline}")
        ax.set_ylabel("score difference")
    fig.suptitle("Paired differences by fold (95% block-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(plot_dir / "08_paired_by_fold.png")
    plt.close(fig)

    # 09 allocation ladder
    ladder = ["full", "no_smoothing", "no_vol", "no_multiplier", "signal_only", "signal_vol", "signal_vol_smooth", "vol_only"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mode in zip(axes, modes):
        rows = [r for r in attribution if r["mode"] == mode]
        means = [float(np.mean([r["score"] for r in rows if r["variant"] == v])) for v in ladder]
        ax.barh(ladder, means, color="tab:blue", alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(mode)
        ax.set_xlabel("mean official score")
    axes[0].invert_yaxis()
    fig.suptitle("Allocation attribution ladder")
    fig.tight_layout()
    fig.savefig(plot_dir / "09_allocation_ladder.png")
    plt.close(fig)

    # 10 cost sensitivity
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, color in [("offline", "tab:blue"), ("online", "tab:orange")]:
        rows = [r for r in costs if r["mode"] == mode]
        xs = sorted({r["transaction_cost"] for r in rows})
        means = [
            float(np.mean([r["score"] for r in rows if r["transaction_cost"] == c]))
            for c in xs
        ]
        ax.plot(np.asarray(xs) * 1e4, means, marker="o", label=mode, color=color)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("transaction cost (bps per day)")
    ax.set_ylabel("mean official score")
    ax.set_title("Transaction-cost sensitivity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "10_cost_sensitivity.png")
    plt.close(fig)

    # 11 position bounds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    folds = [item.fold.name for item in fold_data["online"]]
    for ax, mode in zip(axes, modes):
        rows = [r for r in bounds if r["mode"] == mode]
        rows.sort(key=lambda r: r["fold"])
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r["final_near_zero_share"] for r in rows], 0.4, label="near zero (<=0.01)")
        ax.bar(x + 0.2, [r["final_near_two_share"] for r in rows], 0.4, label="near two (>=1.99)")
        ax.set_xticks(x)
        ax.set_xticklabels([r["fold"] for r in rows])
        ax.set_title(mode)
        ax.set_ylabel("share of days")
        ax.legend(fontsize=7)
    fig.suptitle("Positions at bounds")
    fig.tight_layout()
    fig.savefig(plot_dir / "11_position_bounds.png")
    plt.close(fig)

    # 12 turnover-based transaction cost
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, color in [("offline", "tab:blue"), ("online", "tab:orange")]:
        rows = [r for r in turnover_costs if r["mode"] == mode]
        xs = sorted({r["cost_bps"] for r in rows})
        means = [
            float(np.mean([r["score"] for r in rows if r["cost_bps"] == c]))
            for c in xs
        ]
        ax.plot(xs, means, marker="o", label=mode, color=color)
    ax.set_xlabel("turnover cost (bps per unit traded)")
    ax.set_ylabel("mean official score")
    ax.set_title("Turnover-proportional cost sensitivity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "12_turnover_cost.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation.yaml")
    parser.add_argument("--output", default="reports/validation")
    parser.add_argument("--offline-oof", default=None)
    parser.add_argument("--online-oof", default=None)
    parser.add_argument("--temporal-reps", type=int, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--exposure-paths", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    validation = config.section("validation")
    if args.temporal_reps is not None:
        validation["temporal_reps"] = int(args.temporal_reps)
    if args.bootstrap_reps is not None:
        validation["bootstrap_reps"] = int(args.bootstrap_reps)
        validation["bootstrap_sensitivity_reps"] = int(args.bootstrap_reps)
    if args.exposure_paths is not None:
        validation["exposure_paths"] = int(args.exposure_paths)
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel, folds = make_folds(config)
    offline_path = Path(args.offline_oof) if args.offline_oof else latest_oof(
        config.paths.models_dir, "*_gdbt_notebook_cv_offline_*/oof_predictions.csv"
    )
    online_path = Path(args.online_oof) if args.online_oof else latest_oof(
        config.paths.models_dir, "*_gdbt_notebook_cv_online_*/oof_predictions.csv"
    )
    oofs = {"offline": load_oof(offline_path), "online": load_oof(online_path)}
    fold_data = {
        mode: prepare_fold_data(
            panel, oof, folds, model_config,
            regime_window=int(validation["regime_window"]),
            n_regimes=int(validation["n_regimes"]),
        )
        for mode, oof in oofs.items()
    }
    parity = {}
    for mode, items in fold_data.items():
        diffs = []
        for item in items:
            frame = item.frame
            manual = simulate_positions(
                frame["raw_pred"], frame["vol"], frame["multiplier"],
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
            )
            diffs.append(float(np.max(np.abs(manual - frame["allocation"].to_numpy()))))
        parity[mode] = max(diffs)
        if parity[mode] > 1e-7:
            raise RuntimeError(f"{mode} OOF allocation parity failed: {parity[mode]}")

    seed = int(validation["seed"])
    temporal_summary, temporal_folds, temporal_reps = temporal_tests(
        fold_data, model_config, validation, seed
    )
    paired = paired_tests(fold_data, model_config, validation, seed)
    exposure, exposure_paths = exposure_tests(fold_data, model_config, validation, seed)
    quality, quantiles, ic_summary, monotonicity = prediction_quality(
        fold_data, validation, seed
    )
    attribution, bounds, costs, turnover_costs = attribution_tests(fold_data, model_config, validation)
    strategies = build_strategies(fold_data["online"], model_config)

    output = Path(args.output)
    manifest = write_records(
        output, config, validation, args.config, offline_path, online_path, fold_data,
        temporal_summary, temporal_folds, temporal_reps, paired, exposure,
        exposure_paths, quality, quantiles, ic_summary, monotonicity,
        attribution, bounds, costs, turnover_costs, parity,
    )
    plot_all(
        output, fold_data, temporal_summary, temporal_reps, paired, exposure,
        quality, quantiles, ic_summary, monotonicity, attribution, bounds, costs,
        turnover_costs, strategies,
    )
    print(f"wrote {output}")
    print(json.dumps({"parity": parity, "temporal": temporal_summary[:3], "paired": paired[:3]}, indent=2, default=str))


if __name__ == "__main__":
    main()
