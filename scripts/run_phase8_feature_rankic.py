"""Phase 8: feature-level RankIC analysis on the six-fold OOF dates.

Research-only diagnostics. This script does not train a model, modify the OOF
files, select features, or change any allocation parameter.

The feature values are reconstructed causally, one prediction date at a time,
with the fold-specific `NotebookFeatureBuilder`. The builder is fitted only on
that fold's training dates; `build(panel, [date])` is used so the notebook's
I-column backward fill cannot cross into a later test date. The model-input
version is therefore the same point-in-time feature path used by the saved OOF.

Outputs are written under `reports/` (gitignored) and the narrative report is
`docs/feature_rankic_analysis.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from common.config import COL_DATE_ID, COL_TARGET_METRIC, Config
from common.data import date_ids_for_years, load_date_map, load_panel
from common.registry import utc_now, write_json
from common.splits import rolling_window_folds
from common.validation import (
    moving_block_indices,
    rank_ic,
    regime_labels,
    regime_thresholds,
    trailing_volatility,
)
from gdbt.config import GBDTConfig
from gdbt.features import DERIVED_FEATURES, NotebookFeatureBuilder

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 170,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)

COLORS = {
    "positive": "#38bdf8",
    "negative": "#fb7185",
    "raw": "#38bdf8",
    "derived": "#fbbf24",
    "neutral": "#94a3b8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/phase8.yaml")
    parser.add_argument("--oof", default=None, help="override OOF CSV")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_oof(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path).sort(COL_DATE_ID)
    required = {COL_DATE_ID, "raw_pred", "allocation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF is missing columns: {sorted(missing)}")
    if frame[COL_DATE_ID].n_unique() != frame.height:
        raise ValueError("OOF has duplicate date_id values")
    return frame


def build_folds(panel: pl.DataFrame, config: Config, phase8: dict[str, Any]) -> list[Any]:
    date_map = load_date_map(config.paths.date_map_csv)
    by_year = date_ids_for_years(date_map, phase8.get("test_years", []))
    panel_dates = set(panel[COL_DATE_ID].to_list())
    by_year = {
        int(year): [date_id for date_id in dates if date_id in panel_dates]
        for year, dates in by_year.items()
    }
    by_year = {year: dates for year, dates in by_year.items() if dates}
    return rolling_window_folds(
        panel[COL_DATE_ID].to_list(),
        by_year,
        train_window=int(phase8.get("train_window", 800)),
        gap=int(phase8.get("gap", 1)),
    )


def raw_feature_values(panel: pl.DataFrame, raw_features: list[str]) -> np.ndarray:
    return panel.select(raw_features).to_numpy().astype(float)


def derived_feature_values(raw: np.ndarray, raw_features: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(raw_features)}
    out = np.full((raw.shape[0], len(DERIVED_FEATURES)), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if all(name in index for name in ("I1", "I2")):
            out[:, 0] = raw[:, index["I2"]] - raw[:, index["I1"]]
        if all(name in index for name in ("M11", "I2", "I9", "I7")):
            denominator = (raw[:, index["I2"]] + raw[:, index["I9"]] + raw[:, index["I7"]]) / 3.0
            out[:, 1] = raw[:, index["M11"]] / denominator
        if all(name in index for name in ("V1", "S1")):
            out[:, 2] = raw[:, index["V1"]] * raw[:, index["S1"]]
        if all(name in index for name in ("M11", "V1")):
            out[:, 3] = raw[:, index["M11"]] * raw[:, index["V1"]]
        if all(name in index for name in ("I9", "S1")):
            out[:, 4] = raw[:, index["I9"]] * raw[:, index["S1"]]
    out[~np.isfinite(out)] = np.nan
    return out


def feature_universe(folds: Sequence[Any], panel: pl.DataFrame, model_config: GBDTConfig) -> tuple[list[str], dict[str, set[str]]]:
    union: list[str] = []
    seen: set[str] = set()
    availability: dict[str, set[str]] = defaultdict(set)
    for fold in folds:
        builder = NotebookFeatureBuilder(model_config).fit(panel, fold.train_dates)
        for feature in builder.features:
            if feature not in seen:
                seen.add(feature)
                union.append(feature)
            availability[feature].add(str(fold.name))
    return union, dict(availability)


def build_feature_matrices(
    panel: pl.DataFrame,
    folds: Sequence[Any],
    model_config: GBDTConfig,
    universe: list[str],
    raw_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, set[str]]]:
    """Return model-input X, raw X, derived X, target, row fold names, index."""
    date_ids = [int(value) for value in panel[COL_DATE_ID].to_list()]
    row_index = {date_id: i for i, date_id in enumerate(date_ids)}
    fold_names: list[str] = []
    selected_rows: list[int] = []
    for fold in folds:
        for date_id in fold.test_dates:
            selected_rows.append(row_index[int(date_id)])
            fold_names.append(str(fold.name))
    selected = np.asarray(selected_rows, dtype=int)
    n_rows = len(selected)

    X = np.full((n_rows, len(universe)), np.nan, dtype=float)
    feature_index = {feature: i for i, feature in enumerate(universe)}
    fold_availability: dict[str, set[str]] = defaultdict(set)

    for fold in folds:
        builder = NotebookFeatureBuilder(model_config).fit(panel, fold.train_dates)
        local_features = list(builder.features)
        local_positions = [
            i for i, name in enumerate(fold_names) if name == str(fold.name)
        ]
        for local_row, date_id in zip(local_positions, fold.test_dates, strict=True):
            batch = builder.build(panel, [int(date_id)], with_target=False)
            values = np.asarray(batch.X[0], dtype=float)
            for feature, value in zip(local_features, values, strict=True):
                X[local_row, feature_index[feature]] = value
                fold_availability[feature].add(str(fold.name))

    raw = raw_feature_values(panel, raw_features)[selected]
    derived = derived_feature_values(
        raw_feature_values(panel, raw_features), raw_features
    )[selected]
    target = panel.select(COL_TARGET_METRIC).to_numpy().reshape(-1).astype(float)[selected]
    return X, raw, derived, target, fold_names, fold_availability


def safe_p_value(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan")
    return float(spearmanr(xv, yv).pvalue)


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return out
    ordered = valid_idx[np.argsort(p[valid_idx], kind="mergesort")]
    m = len(ordered)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = ordered[rank - 1]
        value = p[idx] * m / rank
        running = min(running, value)
        adjusted[rank - 1] = running
    out[ordered] = np.clip(adjusted, 0.0, 1.0)
    return out


def flag_feature(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "constant": True,
            "low_variance": True,
            "outlier_count": 0,
            "max_abs_robust_z": float("nan"),
            "feature_min": float("nan"),
            "feature_max": float("nan"),
        }
    std = float(np.std(finite, ddof=0))
    unique = int(len(np.unique(finite)))
    constant = unique <= 1 or std <= 1e-15
    scale = max(abs(float(np.mean(finite))), 1.0)
    low_variance = std <= 1e-10 * scale
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0:
        robust_z = 0.6745 * (finite - median) / mad
    elif std > 0:
        robust_z = (finite - median) / std
    else:
        robust_z = np.zeros_like(finite)
    outlier_count = int(np.sum(np.abs(robust_z) > 8.0))
    return {
        "constant": bool(constant),
        "low_variance": bool(low_variance),
        "outlier_count": outlier_count,
        "max_abs_robust_z": float(np.max(np.abs(robust_z))) if len(robust_z) else float("nan"),
        "feature_min": float(np.min(finite)),
        "feature_max": float(np.max(finite)),
    }


def stability_label(fold_ics: Sequence[float], n_available: int) -> str:
    values = np.asarray([value for value in fold_ics if np.isfinite(value)], dtype=float)
    if len(values) < 3 or n_available < 4:
        return "insufficient"
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    mean = float(np.mean(values))
    if positive >= len(values) - 1 and mean > 0:
        return "stable_positive"
    if negative >= len(values) - 1 and mean < 0:
        return "stable_negative"
    strong = np.sum(np.abs(values) >= 0.10)
    if len(values) >= 4 and strong <= 1:
        return "single_period"
    return "mixed"


def bootstrap_ic(
    x: np.ndarray,
    y: np.ndarray,
    fold_names: Sequence[str],
    block_length: int,
    reps: int,
    seed: int,
) -> dict[str, float]:
    groups: list[np.ndarray] = []
    by_fold: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(fold_names):
        by_fold[name].append(i)
    for indices in by_fold.values():
        idx = np.asarray(indices, dtype=int)
        valid = idx[np.isfinite(x[idx]) & np.isfinite(y[idx])]
        if len(valid) >= 3:
            groups.append(valid)
    actual = rank_ic(x, y)
    if len(groups) == 0 or not np.isfinite(actual):
        return {
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_p_two": float("nan"),
            "bootstrap_n": 0,
        }
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(reps), dtype=float)
    for rep in range(int(reps)):
        sampled: list[int] = []
        for group in groups:
            local = moving_block_indices(len(group), int(block_length), rng)
            sampled.extend(group[local].tolist())
        idx = np.asarray(sampled, dtype=int)
        samples[rep] = rank_ic(x[idx], y[idx])
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return {
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_p_two": float("nan"),
            "bootstrap_n": 0,
        }
    p_two = 2.0 * min(
        (1.0 + float(np.sum(finite <= 0.0))) / (len(finite) + 1.0),
        (1.0 + float(np.sum(finite >= 0.0))) / (len(finite) + 1.0),
    )
    return {
        "bootstrap_ci_low": float(np.percentile(finite, 2.5)),
        "bootstrap_ci_high": float(np.percentile(finite, 97.5)),
        "bootstrap_p_two": float(min(1.0, p_two)),
        "bootstrap_n": int(len(finite)),
    }


def summarize_matrix(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    feature_class: dict[str, str],
    fold_names: Sequence[str],
    phase8: dict[str, Any],
    seed: int,
    do_bootstrap: bool,
    representation: str = "model_input",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    by_fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    n_total = len(target)
    fold_arrays: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(fold_names):
        mask = np.asarray([value == name for value in fold_names], dtype=bool)
        fold_arrays[name] = np.flatnonzero(mask)

    for j, feature in enumerate(feature_names):
        values = matrix[:, j]
        finite_all = np.isfinite(values) & np.isfinite(target)
        n_valid = int(np.sum(finite_all))
        fold_ics: list[float] = []
        available_folds = 0
        for fold_name, indices in fold_arrays.items():
            x = values[indices]
            y = target[indices]
            mask = np.isfinite(x) & np.isfinite(y)
            count = int(np.sum(mask))
            available = count >= 3
            available_folds += int(available)
            ic = rank_ic(x[mask], y[mask]) if available else float("nan")
            p = safe_p_value(x, y) if available else float("nan")
            if np.isfinite(ic):
                fold_ics.append(ic)
            by_fold_rows.append(
                {
                    "feature": feature,
                    "feature_class": feature_class.get(feature, "raw"),
                    "fold": fold_name,
                    "n_total": int(len(indices)),
                    "n_valid": count,
                    "missing_ratio": 1.0 - count / len(indices) if len(indices) else float("nan"),
                    "rank_ic": ic,
                    "p_raw": p,
                    "available": bool(available),
                }
            )
        fold_values = np.asarray(fold_ics, dtype=float)
        mean_ic = float(np.mean(fold_values)) if len(fold_values) else float("nan")
        median_ic = float(np.median(fold_values)) if len(fold_values) else float("nan")
        std_ic = float(np.std(fold_values, ddof=0)) if len(fold_values) > 1 else float("nan")
        icir = mean_ic / std_ic if np.isfinite(std_ic) and std_ic > 1e-12 else float("nan")
        overall_ic = rank_ic(values[finite_all], target[finite_all])
        raw_p = safe_p_value(values, target)
        flags = flag_feature(values)
        row: dict[str, Any] = {
            "feature": feature,
            "feature_class": feature_class.get(feature, "raw"),
            "representation": representation,
            "n_total": n_total,
            "n_valid": n_valid,
            "missing_ratio": 1.0 - n_valid / n_total if n_total else float("nan"),
            "folds_available": available_folds,
            "overall_rank_ic": overall_ic,
            "overall_p_raw": raw_p,
            "mean_rank_ic": mean_ic,
            "median_rank_ic": median_ic,
            "std_rank_ic": std_ic,
            "icir": icir,
            "fold_icir": icir,
            "positive_fold_ratio": (
                float(np.mean(fold_values > 0)) if len(fold_values) else float("nan")
            ),
            "negative_fold_ratio": (
                float(np.mean(fold_values < 0)) if len(fold_values) else float("nan")
            ),
            "stability_label": stability_label(fold_ics, available_folds),
            **flags,
        }
        if do_bootstrap:
            row.update(
                bootstrap_ic(
                    values,
                    target,
                    fold_names,
                    block_length=int(phase8.get("bootstrap_block", 20)),
                    reps=int(phase8.get("bootstrap_reps", 2000)),
                    seed=seed + j,
                )
            )
        summary_rows.append(row)

    summary = pl.DataFrame(summary_rows)
    if "overall_p_raw" in summary.columns:
        q = bh_adjust(summary["overall_p_raw"].to_list())
        summary = summary.with_columns(pl.Series("q_bh", q))
        summary = summary.with_columns(
            (pl.col("q_bh") <= float(phase8.get("fdr_alpha", 0.05))).alias("q_bh_significant")
        )
    by_fold = pl.DataFrame(by_fold_rows)
    return summary, by_fold


def rolling_analysis(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    feature_class: dict[str, str],
    fold_names: Sequence[str],
    date_ids: Sequence[int],
    windows: Sequence[int],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    by_fold: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(fold_names):
        by_fold[name].append(i)
    for j, feature in enumerate(feature_names):
        values = matrix[:, j]
        for window in windows:
            for fold_name, indices in by_fold.items():
                x_all = values[np.asarray(indices, dtype=int)]
                y_all = target[np.asarray(indices, dtype=int)]
                dates = np.asarray([date_ids[i] for i in indices], dtype=int)
                series: list[dict[str, Any]] = []
                for end in range(int(window) - 1, len(indices)):
                    start = end - int(window) + 1
                    x = x_all[start : end + 1]
                    y = y_all[start : end + 1]
                    mask = np.isfinite(x) & np.isfinite(y)
                    if int(np.sum(mask)) < max(3, int(0.8 * int(window))):
                        continue
                    ic = rank_ic(x[mask], y[mask])
                    if not np.isfinite(ic):
                        continue
                    row = {
                        "feature": feature,
                        "feature_class": feature_class.get(feature, "raw"),
                        "representation": "model_input",
                        "fold": fold_name,
                        "date_id": int(dates[end]),
                        "window": int(window),
                        "n_obs": int(np.sum(mask)),
                        "rank_ic": float(ic),
                    }
                    rows.append(row)
                    series.append(row)
                if not series:
                    continue
                sign = 1 if series[0]["rank_ic"] > 0 else -1
                start = 0
                for i, item in enumerate(series[1:], start=1):
                    current = 1 if item["rank_ic"] > 0 else -1
                    if current != sign:
                        block = series[start:i]
                        run_rows.append(
                            {
                                "feature": feature,
                                "fold": fold_name,
                                "window": int(window),
                                "start_date_id": int(block[0]["date_id"]),
                                "end_date_id": int(block[-1]["date_id"]),
                                "sign": "positive" if sign > 0 else "negative",
                                "n_obs": len(block),
                                "mean_rank_ic": float(np.mean([row["rank_ic"] for row in block])),
                            }
                        )
                        start = i
                        sign = current
                block = series[start:]
                run_rows.append(
                    {
                        "feature": feature,
                        "fold": fold_name,
                        "window": int(window),
                        "start_date_id": int(block[0]["date_id"]),
                        "end_date_id": int(block[-1]["date_id"]),
                        "sign": "positive" if sign > 0 else "negative",
                        "n_obs": len(block),
                        "mean_rank_ic": float(np.mean([row["rank_ic"] for row in block])),
                    }
                )
    rolling = pl.DataFrame(rows)
    returns = pl.DataFrame(run_rows)
    stats = (
        rolling.group_by(["feature", "window"])
        .agg(
            pl.col("rank_ic").mean().alias("rolling_mean"),
            pl.col("rank_ic").median().alias("rolling_median"),
            pl.col("rank_ic").std().alias("rolling_std"),
            (pl.col("rank_ic") > 0).mean().alias("rolling_positive_ratio"),
            pl.len().alias("rolling_n"),
        )
        .sort(["feature", "window"])
    )
    # Count actual sign changes within each fold; a fold boundary is not a
    # sign flip. Keep the runs table for interval inspection.
    flips = (
        returns.sort(["feature", "window", "fold", "start_date_id"])
        .with_columns(
            pl.col("sign")
            .shift()
            .over(["feature", "window", "fold"])
            .alias("_previous_sign")
        )
        .filter(
            pl.col("_previous_sign").is_not_null()
            & (pl.col("sign") != pl.col("_previous_sign"))
        )
        .group_by(["feature", "window"])
        .agg(pl.len().alias("rolling_sign_flips"))
    )
    stats = stats.join(flips, on=["feature", "window"], how="left").with_columns(
        pl.col("rolling_sign_flips").fill_null(0)
    )
    return rolling, returns, stats


def regime_analysis(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    feature_class: dict[str, str],
    fold_names: Sequence[str],
    regime_names: np.ndarray,
    min_regime_obs: int,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    by_fold: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(fold_names):
        by_fold[name].append(i)
    for j, feature in enumerate(feature_names):
        values = matrix[:, j]
        for regime in ("low_vol", "mid_vol", "high_vol"):
            fold_values: list[float] = []
            n_obs = 0
            for fold_name, indices in by_fold.items():
                idx = np.asarray(indices, dtype=int)
                mask = (regime_names[idx] == regime) & np.isfinite(values[idx]) & np.isfinite(target[idx])
                count = int(np.sum(mask))
                if count < min_regime_obs:
                    continue
                n_obs += count
                fold_values.append(rank_ic(values[idx][mask], target[idx][mask]))
            finite = np.asarray([v for v in fold_values if np.isfinite(v)], dtype=float)
            if finite.size == 0:
                continue
            rows.append(
                {
                    "feature": feature,
                    "feature_class": feature_class.get(feature, "raw"),
                    "representation": "model_input",
                    "regime": regime,
                    "n_obs": int(n_obs),
                    "n_folds": int(len(finite)),
                    "mean_rank_ic": float(np.mean(finite)),
                    "median_rank_ic": float(np.median(finite)),
                    "positive_ic_ratio": float(np.mean(finite > 0)),
                    "pooled_rank_ic": float(
                        rank_ic(
                            values[
                                np.isfinite(values)
                                & np.isfinite(target)
                                & np.isin(regime_names, [regime])
                            ],
                            target[
                                np.isfinite(values)
                                & np.isfinite(target)
                                & np.isin(regime_names, [regime])
                            ],
                        )
                    ),
                    "inconclusive": bool(len(finite) < 4 or n_obs < min_regime_obs),
                }
            )
    return pl.DataFrame(rows)


def regime_counts(
    fold_names: Sequence[str],
    regime_names: np.ndarray,
) -> pl.DataFrame:
    rows = []
    for fold in dict.fromkeys(fold_names):
        mask = np.asarray([name == fold for name in fold_names], dtype=bool)
        for regime in ("low_vol", "mid_vol", "high_vol"):
            rows.append(
                {
                    "fold": fold,
                    "regime": regime,
                    "n_days": int(np.sum(mask & (regime_names == regime))),
                }
            )
    return pl.DataFrame(rows)


def correlation_analysis(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    feature_class: dict[str, str],
    threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = len(feature_names)
    corr = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(corr, 1.0)
    pair_rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            mask = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
            count = int(np.sum(mask))
            if count < 20 or np.std(matrix[mask, i]) == 0 or np.std(matrix[mask, j]) == 0:
                continue
            ri = rankdata(matrix[mask, i])
            rj = rankdata(matrix[mask, j])
            rho = float(np.corrcoef(ri, rj)[0, 1])
            corr[i, j] = corr[j, i] = rho
            pair_rows.append(
                {
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "feature_class_a": feature_class.get(feature_names[i], "raw"),
                    "feature_class_b": feature_class.get(feature_names[j], "raw"),
                    "spearman_rho": rho,
                    "n_pair": count,
                    "high_corr": bool(abs(rho) >= threshold),
                }
            )
    matrix_frame = pl.DataFrame(
        {"feature": feature_names, **{name: corr[:, j] for j, name in enumerate(feature_names)}}
    )
    pairs = pl.DataFrame(pair_rows).sort(pl.col("spearman_rho").abs(), descending=True)
    # Connected components of the |rho| >= threshold graph.
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= threshold:
                union(i, j)
    clusters: dict[int, list[str]] = defaultdict(list)
    for i, feature in enumerate(feature_names):
        clusters[find(i)].append(feature)
    cluster_rows = [
        {
            "cluster": f"cluster_{k}",
            "n_features": len(features),
            "features": "|".join(features),
            "feature_classes": "|".join(sorted({feature_class.get(f, "raw") for f in features})),
        }
        for k, features in enumerate(clusters.values(), start=1)
        if len(features) > 1
    ]
    clusters_frame = pl.DataFrame(cluster_rows) if cluster_rows else pl.DataFrame(
        {"cluster": [], "n_features": [], "features": [], "feature_classes": []}
    )
    # Group-level average absolute correlation, using the raw/derived class and
    # the first letter of raw features as the feature family.
    def family(feature: str) -> str:
        return "derived" if feature_class.get(feature) == "derived" else feature[0]

    group_rows = []
    families = sorted({family(feature) for feature in feature_names})
    for a in families:
        for b in families:
            vals = [
                abs(corr[i, j])
                for i in range(n)
                for j in range(i + 1, n)
                if family(feature_names[i]) == a
                and family(feature_names[j]) == b
                and np.isfinite(corr[i, j])
            ]
            if vals:
                group_rows.append(
                    {
                        "group_a": a,
                        "group_b": b,
                        "mean_abs_spearman": float(np.mean(vals)),
                        "n_pairs": int(len(vals)),
                    }
                )
    groups = pl.DataFrame(group_rows)
    return matrix_frame, pairs, clusters_frame, groups


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_features(summary: pl.DataFrame, path: Path) -> None:
    top = (
        summary.filter(pl.col("mean_rank_ic").is_not_null())
        .with_columns(pl.col("mean_rank_ic").abs().alias("_abs"))
        .sort("_abs", descending=True)
        .head(20)
    )
    rows = list(top.iter_rows(named=True))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    colors = [COLORS["positive"] if row["mean_rank_ic"] >= 0 else COLORS["negative"] for row in rows]
    ax.barh([row["feature"] for row in rows], [row["mean_rank_ic"] for row in rows], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mean fold RankIC")
    ax.set_title("Top 20 features by |mean fold RankIC| (color = sign)")
    limit = max(max(abs(row["mean_rank_ic"]) for row in rows) * 1.55, 0.05)
    ax.set_xlim(-limit, limit)
    for i, row in enumerate(rows):
        ax.text(
            row["mean_rank_ic"] + (0.002 if row["mean_rank_ic"] >= 0 else -0.002),
            i,
            f"{row['mean_rank_ic']:.3f} · {row['folds_available']} folds · n={row['n_valid']}",
            va="center",
            ha="left" if row["mean_rank_ic"] >= 0 else "right",
            fontsize=6.5,
            color="#334155",
            clip_on=False,
        )
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, path)


def plot_fold_heatmap(summary: pl.DataFrame, by_fold: pl.DataFrame, path: Path) -> None:
    ordered = (
        summary.with_columns(pl.col("mean_rank_ic").abs().alias("_abs"))
        .sort("_abs", descending=True)
        .select("feature")
        .to_series()
        .to_list()
    )
    folds = list(dict.fromkeys(by_fold["fold"].to_list()))
    pivot = (
        by_fold.pivot(index="feature", on="fold", values="rank_ic")
        .select(["feature", *folds])
        .with_columns(pl.col("feature").cast(pl.Categorical))
    )
    data = pivot.select(folds).to_numpy().astype(float)
    feature_labels = pivot["feature"].cast(pl.Utf8).to_list()
    order = [feature_labels.index(feature) for feature in ordered if feature in feature_labels]
    data = data[order]
    labels = [feature_labels[i] for i in order]
    fold_counts = {
        row["fold"]: int(row["n_total"])
        for row in by_fold.group_by("fold").agg(pl.col("n_total").max()).iter_rows(named=True)
    }
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e2e8f0")
    fig, ax = plt.subplots(figsize=(8.5, max(8, len(labels) * 0.23)))
    im = ax.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        cmap=cmap,
        vmin=-0.6,
        vmax=0.6,
    )
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"{fold}\n(n={fold_counts.get(fold, 0)})" for fold in folds], rotation=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title("Feature RankIC across six OOF folds (all features; grey = unavailable)")
    fig.colorbar(im, ax=ax, shrink=0.55, label="RankIC")
    save_plot(fig, path)


def plot_regime(regime: pl.DataFrame, counts: pl.DataFrame, path: Path) -> None:
    if regime.height == 0:
        return
    totals = {
        row["regime"]: int(row["n_days"])
        for row in counts.group_by("regime").agg(pl.col("n_days").sum()).iter_rows(named=True)
    }
    top = (
        regime.with_columns(pl.col("mean_rank_ic").abs().alias("_abs"))
        .group_by("feature")
        .agg(pl.col("_abs").max().alias("_max"))
        .sort("_max", descending=True)
        .head(20)["feature"]
        .to_list()
    )
    sub = regime.filter(pl.col("feature").is_in(top))
    piv = sub.pivot(index="feature", on="regime", values="mean_rank_ic").fill_null(float("nan"))
    order = {feature: i for i, feature in enumerate(top)}
    piv = piv.with_columns(pl.col("feature").replace_strict(order, default=9999).alias("_order")).sort("_order")
    labels = piv["feature"].to_list()
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for offset, regime_name, color in [
        (-width, "low_vol", COLORS["positive"]),
        (0.0, "mid_vol", COLORS["derived"]),
        (width, "high_vol", COLORS["negative"]),
    ]:
        values = piv[regime_name].to_numpy() if regime_name in piv.columns else np.full(len(labels), np.nan)
        ax.bar(
            x + offset,
            values,
            width,
            label=f"{regime_name} (days={totals.get(regime_name, 0)})",
            color=color,
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("mean fold RankIC within regime")
    ax.set_title("Top features by |regime RankIC| (colors = low / mid / high volatility)")
    ax.legend(frameon=False, fontsize=8)
    save_plot(fig, path)


def plot_correlation(corr: pl.DataFrame, pairs: pl.DataFrame, path: Path) -> None:
    names = corr["feature"].to_list()
    matrix = corr.select(names).to_numpy().astype(float)
    high = int(pairs.filter(pl.col("high_corr")).height) if pairs.height else 0
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=3.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=3.5)
    ax.set_title(
        f"Model-input feature Spearman correlation matrix\n"
        f"pairwise n<=1,488; {high} pairs with |rho|>=0.90"
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman rho")
    save_plot(fig, path)


def plot_scatter(summary: pl.DataFrame, path: Path) -> None:
    data = summary.filter(pl.col("mean_rank_ic").is_not_null() & pl.col("std_rank_ic").is_not_null())
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for cls, color in [("raw", COLORS["raw"]), ("derived", COLORS["derived"])]:
        sub = data.filter(pl.col("feature_class") == cls)
        ax.scatter(
            sub["mean_rank_ic"].to_numpy(),
            sub["std_rank_ic"].to_numpy(),
            s=18 + 0.03 * sub["n_valid"].to_numpy(),
            alpha=0.75,
            color=color,
            label=cls,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mean fold RankIC")
    ax.set_ylabel("RankIC standard deviation across folds")
    ax.set_title(
        f"RankIC level vs instability (all {data.height} model-input features; "
        f"n=1,488 OOF dates)"
    )
    for row in data.sort(pl.col("mean_rank_ic").abs(), descending=True).head(8).iter_rows(named=True):
        ax.annotate(row["feature"], (row["mean_rank_ic"], row["std_rank_ic"]), fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    save_plot(fig, path)


def plot_rolling(
    rolling: pl.DataFrame,
    summary: pl.DataFrame,
    path: Path,
) -> None:
    if rolling.height == 0:
        return
    selected = (
        summary.filter(pl.col("mean_rank_ic").is_not_null())
        .with_columns(pl.col("mean_rank_ic").abs().alias("_abs"))
        .sort("_abs", descending=True)
        .head(6)["feature"]
        .to_list()
    )
    sub = rolling.filter(
        pl.col("feature").is_in(selected) & (pl.col("window") == 120)
    )
    fold_starts = (
        sub.group_by("fold").agg(pl.col("date_id").min()).sort("date_id")
        if sub.height
        else pl.DataFrame({"fold": [], "date_id": []})
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 8.4), sharex=True)
    for ax, feature in zip(axes.flat, selected):
        data = sub.filter(pl.col("feature") == feature).sort("date_id")
        ax.plot(data["date_id"].to_numpy(), data["rank_ic"].to_numpy(), linewidth=1.1)
        ax.axhline(0, color="black", linewidth=0.7)
        for start in fold_starts["date_id"].to_list():
            ax.axvline(start, color="#94a3b8", linewidth=0.6, linestyle=":")
        ax.set_title(feature, fontsize=9)
        ax.set_ylabel("120d RankIC")
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("date_id (vertical dotted lines = fold starts)")
    fig.suptitle("Rolling 120-day RankIC — top 6 by |mean fold RankIC|", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    phase8 = config.section("phase8")
    seed = int(args.seed if args.seed is not None else phase8.get("seed", 42))
    oof_path = Path(args.oof or phase8["oof_path"])
    output = Path(args.output_dir)
    figures = output / "feature_rankic_figures"
    figures.mkdir(parents=True, exist_ok=True)

    panel = load_panel(config.paths.train_csv, config.paths.test_csv)
    folds = build_folds(panel, config, phase8)
    oof = load_oof(oof_path)
    expected_dates = [int(date) for fold in folds for date in fold.test_dates]
    if oof.height != len(expected_dates) or set(oof[COL_DATE_ID].to_list()) != set(expected_dates):
        raise ValueError(
            f"OOF dates ({oof.height}) do not match the six-fold test dates ({len(expected_dates)})"
        )
    if len(expected_dates) != 1488:
        raise ValueError(f"Expected 1,488 OOF dates, got {len(expected_dates)}")

    raw_features = [name for name in panel.columns if name[:1] in {"D", "E", "I", "M", "P", "S", "V"}]
    if len(raw_features) != 94:
        raise ValueError(f"Expected 94 raw features, got {len(raw_features)}")
    universe, availability = feature_universe(folds, panel, model_config)
    X, raw, derived, target, fold_names, fold_availability = build_feature_matrices(
        panel, folds, model_config, universe, raw_features
    )
    if not np.isfinite(target).all():
        raise ValueError("Non-finite target_metric in OOF dates")
    feature_class = {name: ("derived" if name in DERIVED_FEATURES else "raw") for name in universe}

    # Causal checks: single-row feature construction must equal the same call
    # on a panel truncated at that date. This is the same contract as the
    # model's one-date-at-a-time prediction path.
    leakage_checks = []
    for fold in folds[:2]:
        builder = NotebookFeatureBuilder(model_config).fit(panel, fold.train_dates)
        date = int(fold.test_dates[len(fold.test_dates) // 2])
        full = builder.build(panel, [date], with_target=False).X
        truncated = builder.build(
            panel.filter(pl.col(COL_DATE_ID) <= date), [date], with_target=False
        ).X
        leakage_checks.append(float(np.max(np.abs(full - truncated))))

    summary, by_fold = summarize_matrix(
        X,
        target,
        universe,
        feature_class,
        fold_names,
        phase8,
        seed,
        do_bootstrap=True,
        representation="model_input",
    )
    raw_summary, raw_by_fold = summarize_matrix(
        raw,
        target,
        raw_features,
        {name: "raw" for name in raw_features},
        fold_names,
        phase8,
        seed + 500,
        do_bootstrap=False,
        representation="raw",
    )
    derived_summary, derived_by_fold = summarize_matrix(
        derived,
        target,
        DERIVED_FEATURES,
        {name: "derived" for name in DERIVED_FEATURES},
        fold_names,
        phase8,
        seed + 700,
        do_bootstrap=False,
        representation="derived",
    )
    rolling, rolling_runs, rolling_stats = rolling_analysis(
        X,
        target,
        universe,
        feature_class,
        fold_names,
        expected_dates,
        phase8.get("rolling_windows", [60, 120]),
    )
    if rolling_stats.height:
        summary = summary.join(
            rolling_stats.pivot(index="feature", on="window", values=["rolling_mean", "rolling_std", "rolling_positive_ratio", "rolling_sign_flips"]),
            on="feature",
            how="left",
        )
    regime_names = np.full(len(target), "unknown", dtype=object)
    regime_counts_rows = []
    panel_sorted = panel.sort(COL_DATE_ID)
    panel_index = {int(date): i for i, date in enumerate(panel_sorted[COL_DATE_ID].to_list())}
    trailing = trailing_volatility(
        panel_sorted[COL_TARGET_METRIC].to_numpy(),
        window=int(phase8.get("regime_window", 20)),
    )
    for fold in folds:
        train_idx = [panel_index[int(date)] for date in fold.train_dates]
        train_trailing = np.asarray([trailing[i] for i in train_idx], dtype=float)
        thresholds = regime_thresholds(
            train_trailing, n_regimes=int(phase8.get("n_regimes", 3))
        )
        local_positions = np.asarray(
            [i for i, name in enumerate(fold_names) if name == str(fold.name)], dtype=int
        )
        test_trailing = np.asarray(
            [trailing[panel_index[int(date)]] for date in fold.test_dates], dtype=float
        )
        labels = regime_labels(test_trailing, thresholds)
        regime_names[local_positions] = [
            "low_vol" if value == 0 else "mid_vol" if value == 1 else "high_vol"
            for value in labels
        ]
        regime_counts_rows.extend(
            [
                {
                    "fold": str(fold.name),
                    "regime": name,
                    "n_days": int(np.sum(labels == idx)),
                    "threshold_low": float(thresholds[0]),
                    "threshold_high": float(thresholds[1]),
                }
                for idx, name in enumerate(("low_vol", "mid_vol", "high_vol"))
            ]
        )
    regime = regime_analysis(
        X,
        target,
        universe,
        feature_class,
        fold_names,
        regime_names,
        min_regime_obs=int(phase8.get("min_regime_obs", 20)),
    )
    regime_counts_frame = pl.DataFrame(regime_counts_rows)
    corr_matrix, corr_pairs, corr_clusters, group_corr = correlation_analysis(
        X, universe, feature_class, float(phase8.get("high_corr_threshold", 0.90))
    )

    # Files explicitly requested by the phase brief.
    summary.write_csv(output / "feature_rankic_summary.csv")
    by_fold.write_csv(output / "feature_rankic_by_fold.csv")
    raw_summary.write_csv(output / "feature_rankic_raw_summary.csv")
    raw_by_fold.write_csv(output / "feature_rankic_raw_by_fold.csv")
    derived_summary.write_csv(output / "feature_rankic_derived_summary.csv")
    derived_by_fold.write_csv(output / "feature_rankic_derived_by_fold.csv")
    rolling.write_parquet(output / "feature_rankic_rolling.parquet")
    rolling_runs.write_csv(output / "feature_rankic_rolling_runs.csv")
    corr_matrix.write_csv(output / "feature_correlation_matrix.csv")
    corr_pairs.write_csv(output / "feature_correlation_pairs.csv")
    corr_clusters.write_csv(output / "feature_correlation_clusters.csv")
    group_corr.write_csv(output / "feature_group_correlation.csv")
    regime.write_csv(output / "feature_rankic_regime.csv")
    regime_counts_frame.write_csv(output / "feature_rankic_regime_counts.csv")

    plot_top_features(summary, figures / "01_top_features_abs_mean_rankic.png")
    plot_fold_heatmap(summary, by_fold, figures / "02_feature_fold_heatmap.png")
    plot_rolling(rolling, summary, figures / "03_rolling_rankic_selected.png")
    plot_regime(regime, regime_counts_frame, figures / "04_regime_rankic.png")
    plot_correlation(corr_matrix, corr_pairs, figures / "05_feature_correlation_heatmap.png")
    plot_scatter(summary, figures / "06_mean_vs_std_scatter.png")

    # Compact, human-readable feature-set audit.
    feature_sets = []
    for fold in folds:
        builder = NotebookFeatureBuilder(model_config).fit(panel, fold.train_dates)
        feature_sets.append(
            {
                "fold": str(fold.name),
                "n_train": len(fold.train_dates),
                "n_test": len(fold.test_dates),
                "n_features": len(builder.features),
                "missing_from_union": "|".join(sorted(set(universe) - set(builder.features))),
                "features": "|".join(builder.features),
            }
        )
    feature_sets_frame = pl.DataFrame(feature_sets)
    feature_sets_frame.write_csv(output / "feature_rankic_feature_sets.csv")

    manifest = {
        "created_at": utc_now(),
        "phase": "phase8_feature_rankic",
        "oof": str(oof_path),
        "oof_sha256": sha256(oof_path),
        "panel": str(config.paths.train_csv),
        "panel_sha256": sha256(Path(config.paths.train_csv)),
        "n_oof_dates": len(target),
        "n_features_union": len(universe),
        "n_raw_features": len(raw_features),
        "n_derived_features": len(DERIVED_FEATURES),
        "n_folds": len(folds),
        "folds": [
            {
                "fold": fold.name,
                "n_train": len(fold.train_dates),
                "n_test": len(fold.test_dates),
                "train_start": int(fold.train_dates[0]),
                "train_end": int(fold.train_dates[-1]),
                "test_start": int(fold.test_dates[0]),
                "test_end": int(fold.test_dates[-1]),
                "n_features": len(NotebookFeatureBuilder(model_config).fit(panel, fold.train_dates).features),
            }
            for fold in folds
        ],
        "feature_sets": feature_sets,
        "regime_counts": regime_counts_rows,
        "leakage_checks_max_abs_diff": leakage_checks,
        "leakage_check_passed": bool(max(leakage_checks) < 1e-12) if leakage_checks else False,
        "rolling_windows": phase8.get("rolling_windows", [60, 120]),
        "bootstrap_reps": int(phase8.get("bootstrap_reps", 2000)),
        "bootstrap_block": int(phase8.get("bootstrap_block", 20)),
        "fdr_alpha": float(phase8.get("fdr_alpha", 0.05)),
        "notes": [
            "No model was trained; OOF file was read but not modified.",
            "Feature values are rebuilt one date at a time from fold-specific training windows.",
            "Unavailable features in a fold are kept as NaN and reported, not dropped silently.",
            "BH FDR is applied to the 99 model-input features separately from raw features.",
            "RankIC is time-series Spearman across OOF dates, not daily cross-sectional IC.",
            "Rolling and regime statistics are descriptive and are not used for model or feature selection.",
        ],
    }
    write_json(output / "feature_rankic_manifest.json", manifest)
    print(summary.head(20))
    print(regime_counts_frame)
    print(f"wrote {output} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
