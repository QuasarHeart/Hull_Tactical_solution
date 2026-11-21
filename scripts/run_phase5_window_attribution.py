"""Phase 5: attribution of the 137-day official competition window.

All experiments start from the saved Daily-Retrain position path; no model is
retrained and no OOF is modified. The original path is reproduced first, then
signal/value ablations are evaluated with the same multiplier, volatility,
smoothing, clipping and transaction-cost rules.
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

from common.config import COL_DATE, COL_DATE_ID, COL_TARGET, COL_TARGET_METRIC, Config
from common.evaluation import summarize_positions, summarize_turnover_adjusted
from common.experiments import simulate_positions
from common.metrics import ParticipantVisibleError
from common.registry import utc_now, write_json
from common.validation import block_permute, rank_ic, pearson_ic, position_stats
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


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(positions: np.ndarray, target_metric: np.ndarray) -> dict[str, Any]:
    if np.allclose(positions, 0.0):
        return {
            "score": 0.0,
            "sharpe": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "mean_allocation": 0.0,
            "strategy_vol": 0.0,
            "market_vol": 0.0,
            "vol_penalty": 1.0,
            "return_penalty": 1.0,
            "strategy_mean_excess": 0.0,
            "market_mean_excess": 0.0,
            "n_days": len(positions),
            "cash": True,
        }
    try:
        report = summarize_positions(positions, target_metric, np.zeros_like(target_metric))
        report["cash"] = False
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
            "market_vol": float("nan"),
            "vol_penalty": float("nan"),
            "return_penalty": float("nan"),
            "strategy_mean_excess": float("nan"),
            "market_mean_excess": float("nan"),
            "n_days": len(positions),
            "cash": False,
        }


def _variant_row(
    name: str,
    positions: np.ndarray,
    target_metric: np.ndarray,
    *,
    description: str,
    cost_bps: float,
) -> dict[str, Any]:
    metrics = _score(positions, target_metric)
    stats = position_stats(positions)
    turnover_adjusted = summarize_turnover_adjusted(
        positions, target_metric, np.zeros_like(target_metric), cost_bps=cost_bps
    )
    return {
        "variant": name,
        "description": description,
        "score": metrics["score"],
        "raw_sharpe": metrics["sharpe"],
        "cumulative_return": metrics["cumulative_return"],
        "max_drawdown": metrics["max_drawdown"],
        "turnover": metrics["turnover"],
        "mean_allocation": metrics["mean_allocation"],
        "zero_share": stats.get("zero_share", float("nan")),
        "two_share": stats.get("two_share", float("nan")),
        "strategy_mean_excess": metrics["strategy_mean_excess"],
        "market_mean_excess": metrics["market_mean_excess"],
        "vol_penalty": metrics["vol_penalty"],
        "return_penalty": metrics["return_penalty"],
        "net_sharpe_10bps": turnover_adjusted["net_sharpe"],
        "net_cumulative_10bps": turnover_adjusted["net_cumulative_return"],
        "n_days": len(positions),
    }


def _month_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for month in sorted(set(frame["month"].to_list())):
        sub = frame.filter(pl.col("month") == month).sort(COL_DATE_ID)
        metrics = _score(sub["allocation"].to_numpy(), sub[COL_TARGET_METRIC].to_numpy())
        rows.append(
            {
                "month": month,
                "n_days": sub.height,
                "score": metrics["score"],
                "raw_sharpe": metrics["sharpe"],
                "cumulative_return": metrics["cumulative_return"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
                "mean_allocation": metrics["mean_allocation"],
                "mean_target": float(sub[COL_TARGET_METRIC].mean()),
                "small_sample": sub.height < 15,
            }
        )
    return rows


def _regime_rows(frame: pl.DataFrame, thresholds: tuple[float, float]) -> list[dict[str, Any]]:
    low, high = thresholds
    labels = np.where(
        frame["vol"].to_numpy() <= low,
        "low_vol",
        np.where(frame["vol"].to_numpy() >= high, "high_vol", "mid_vol"),
    )
    frame = frame.with_columns(pl.Series("regime", labels))
    rows = []
    for regime in ("low_vol", "mid_vol", "high_vol"):
        sub = frame.filter(pl.col("regime") == regime).sort(COL_DATE_ID)
        if sub.height == 0:
            continue
        metrics = _score(sub["allocation"].to_numpy(), sub[COL_TARGET_METRIC].to_numpy())
        rows.append(
            {
                "regime": regime,
                "n_days": sub.height,
                "score": metrics["score"],
                "raw_sharpe": metrics["sharpe"],
                "cumulative_return": metrics["cumulative_return"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
                "mean_allocation": metrics["mean_allocation"],
                "mean_target": float(sub[COL_TARGET_METRIC].mean()),
                "mean_vol": float(sub["vol"].mean()),
            }
        )
    return rows


def _signal_bins(frame: pl.DataFrame, n_bins: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred = frame["raw_pred"].to_numpy()
    target = frame[COL_TARGET_METRIC].to_numpy()
    order = np.argsort(pred, kind="mergesort")
    groups = np.array_split(order, n_bins)
    rows = []
    for index, group in enumerate(groups):
        rows.append(
            {
                "bin": index + 1,
                "n": int(len(group)),
                "pred_min": float(np.min(pred[group])),
                "pred_max": float(np.max(pred[group])),
                "mean_pred": float(np.mean(pred[group])),
                "mean_target": float(np.mean(target[group])),
                "median_target": float(np.median(target[group])),
                "positive_share": float(np.mean(target[group] > 0)),
                "mean_allocation": float(np.mean(frame["allocation"].to_numpy()[group])),
            }
        )
    means = np.asarray([row["mean_target"] for row in rows])
    monotone_up = bool(np.all(np.diff(means) >= 0.0))
    monotone_down = bool(np.all(np.diff(means) <= 0.0))
    summary = {
        "n_bins": n_bins,
        "pearson_ic": float(pearson_ic(pred, target)),
        "rank_ic": float(rank_ic(pred, target)),
        "bin_mean_targets": means.tolist(),
        "monotone_up": monotone_up,
        "monotone_down": monotone_down,
        "top_minus_bottom": float(means[-1] - means[0]),
    }
    return rows, summary


def _shuffle(
    frame: pl.DataFrame,
    model_config: GBDTConfig,
    reps: int,
    block_length: int,
    seed: int,
    block: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    raw = frame["raw_pred"].to_numpy()
    vol = frame["vol"].to_numpy()
    multiplier = frame["multiplier"].to_numpy()
    target = frame[COL_TARGET_METRIC].to_numpy()
    actual = float(_score(frame["allocation"].to_numpy(), target)["score"])
    rng = np.random.default_rng(seed)
    scores = np.empty(int(reps), dtype=float)
    for i in range(int(reps)):
        shuffled = (
            block_permute(raw, block_length, rng)
            if block
            else rng.permutation(raw)
        )
        positions = simulate_positions(
            shuffled,
            vol,
            multiplier,
            smoothing=model_config.smoothing,
            transaction_cost=model_config.transaction_cost,
            vol_scaling=model_config.vol_scaling,
            max_position=model_config.max_signal,
        )
        scores[i] = float(_score(positions, target)["score"])
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        p = float("nan")
    else:
        p = (1.0 + float(np.sum(finite >= actual))) / (len(finite) + 1.0)
    summary = {
        "label": "block_shuffle" if block else "iid_shuffle",
        "actual_score": actual,
        "n_reps": int(reps),
        "block_length": int(block_length) if block else None,
        "null_mean": float(np.nanmean(finite)) if len(finite) else float("nan"),
        "null_sd": float(np.nanstd(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "null_p05": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "null_median": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "null_p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        "p_value": p,
        "finite_reps": int(len(finite)),
    }
    rows = [
        {"label": summary["label"], "rep": i, "score": float(scores[i])}
        for i in range(len(scores))
    ]
    return rows, summary, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/phase5.yaml")
    parser.add_argument("--positions", default="reports/live_window_20251216_20260706/positions.csv")
    parser.add_argument("--panel", default="data/lake/panel_live.parquet")
    parser.add_argument("--output", default="reports/phase5_window_attribution")
    parser.add_argument("--shuffle-reps", type=int, default=None)
    parser.add_argument("--block-shuffle-reps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    phase5 = config.section("phase5")
    shuffle_reps = int(
        args.shuffle_reps if args.shuffle_reps is not None else phase5.get("shuffle_reps", 2000)
    )
    block_reps = int(
        args.block_shuffle_reps
        if args.block_shuffle_reps is not None
        else phase5.get("block_shuffle_reps", 1000)
    )
    block_length = int(
        args.block_length if args.block_length is not None else phase5.get("block_length", 20)
    )
    seed = int(phase5.get("seed", 42))
    n_bins = int(phase5.get("n_signal_bins", 5))
    cost_bps = float(phase5.get("cost_bps", 10.0))

    positions_path = Path(args.positions)
    positions = (
        pl.read_csv(positions_path)
        .filter(pl.col("mode") == "daily")
        .sort(COL_DATE_ID)
    )
    if positions.height != 137:
        raise ValueError(f"Expected 137 Daily Retrain days, got {positions.height}")
    panel = pl.read_parquet(args.panel).select([COL_DATE_ID, COL_DATE]).sort(COL_DATE_ID)
    frame = positions.join(panel, on=COL_DATE_ID, how="left", suffix="_panel")
    frame = frame.with_columns(pl.col(COL_DATE).dt.strftime("%Y-%m").alias("month"))
    raw = frame["raw_pred"].to_numpy()
    target = frame[COL_TARGET_METRIC].to_numpy()
    vol = frame["vol"].to_numpy()
    multiplier = frame["multiplier"].to_numpy()
    original = frame["allocation"].to_numpy()

    # Verify the saved path reproduces the same allocation formula.
    reproduced = simulate_positions(
        raw,
        vol,
        multiplier,
        smoothing=model_config.smoothing,
        transaction_cost=model_config.transaction_cost,
        vol_scaling=model_config.vol_scaling,
        max_position=model_config.max_signal,
    )
    allocation_parity_max_diff = float(np.max(np.abs(reproduced - original)))

    output = Path(args.output)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    variant_rows = [
        _variant_row(
            "original",
            original,
            target,
            description="original signal + original allocation",
            cost_bps=cost_bps,
        ),
        _variant_row(
            "reverse_signal",
            simulate_positions(
                -raw,
                vol,
                multiplier,
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
                max_position=model_config.max_signal,
            ),
            target,
            description="raw signal sign reversed, same allocation rules",
            cost_bps=cost_bps,
        ),
        _variant_row(
            "no_multiplier",
            simulate_positions(
                raw,
                vol,
                np.ones_like(multiplier),
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
                max_position=model_config.max_signal,
            ),
            target,
            description="multiplier set to 1",
            cost_bps=cost_bps,
        ),
        _variant_row(
            "fixed_600",
            simulate_positions(
                raw,
                vol,
                np.full_like(multiplier, 600.0),
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
                max_position=model_config.max_signal,
            ),
            target,
            description="fixed 600 multiplier, no 600/400 switch",
            cost_bps=cost_bps,
        ),
        _variant_row(
            "fixed_400",
            simulate_positions(
                raw,
                vol,
                np.full_like(multiplier, 400.0),
                smoothing=model_config.smoothing,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
                max_position=model_config.max_signal,
            ),
            target,
            description="fixed 400 multiplier, no 600/400 switch",
            cost_bps=cost_bps,
        ),
        _variant_row(
            "no_smoothing",
            simulate_positions(
                raw,
                vol,
                multiplier,
                smoothing=0.0,
                transaction_cost=model_config.transaction_cost,
                vol_scaling=model_config.vol_scaling,
                max_position=model_config.max_signal,
            ),
            target,
            description="smoothing=0, same cost",
            cost_bps=cost_bps,
        ),
    ]

    iid_rows, iid_summary, iid_scores = _shuffle(
        frame, model_config, shuffle_reps, block_length, seed, block=False
    )
    block_rows, block_summary, block_scores = _shuffle(
        frame, model_config, block_reps, block_length, seed + 1, block=True
    )
    variant_rows.extend(
        []
    )
    variant_df = pl.DataFrame(variant_rows)
    variant_df.write_csv(output / "variant_metrics.csv")
    pl.DataFrame([*iid_rows, *block_rows]).write_csv(output / "shuffle_scores.csv")
    write_json(
        output / "shuffle_summary.json",
        {"iid": iid_summary, "block": block_summary},
    )

    bins, bin_summary = _signal_bins(frame, n_bins)
    pl.DataFrame(bins).write_csv(output / "signal_bins.csv")
    write_json(output / "signal_bins_summary.json", bin_summary)

    monthly = _month_rows(frame)
    pl.DataFrame(monthly).write_csv(output / "monthly.csv")

    initial = (
        panel.join(
            pl.read_parquet(args.panel).select([COL_DATE_ID, "target"]),
            on=COL_DATE_ID,
            how="left",
        )
        .filter(
            (pl.col(COL_DATE_ID) < int(frame[COL_DATE_ID][0]))
            & pl.col("target").is_not_null()
        )
        .sort(COL_DATE_ID)
        .tail(model_config.max_train_rows)
    )
    # Recompute the causal vol on the initial window using the same panel target.
    from common.experiments import causal_vol_estimates

    initial_vol = causal_vol_estimates(
        pl.read_parquet(args.panel), initial[COL_DATE_ID].to_list(), vol_window=model_config.vol_window
    )
    thresholds = tuple(
        float(value)
        for value in np.nanquantile(initial_vol, [0.33, 0.67])
    )
    regime = _regime_rows(frame, thresholds)  # type: ignore[arg-type]
    pl.DataFrame(regime).write_csv(output / "regime.csv")

    _plot_variants(variant_df, output / "plots" / "01_variant_scores.png")
    _plot_shuffle(iid_scores, block_scores, iid_summary, block_summary, output / "plots" / "02_shuffle_null.png")
    _plot_bins(pl.DataFrame(bins), output / "plots" / "03_signal_bins.png")
    _plot_monthly(pl.DataFrame(monthly), output / "plots" / "04_monthly.png")
    _plot_regime(pl.DataFrame(regime), output / "plots" / "05_regime.png")
    _plot_paths(frame, variant_df, output / "plots" / "06_allocation_paths.png")

    manifest = {
        "created_at": utc_now(),
        "phase": "phase5_window_attribution",
        "positions": str(positions_path),
        "positions_sha256": _hash(positions_path),
        "panel": str(args.panel),
        "n_days": int(frame.height),
        "date_id_start": int(frame[COL_DATE_ID][0]),
        "date_id_end": int(frame[COL_DATE_ID][-1]),
        "allocation_parity_max_diff": allocation_parity_max_diff,
        "shuffle": {"iid": iid_summary, "block": block_summary},
        "signal_bins": bin_summary,
        "regime_thresholds": list(thresholds),
        "cost_bps": cost_bps,
        "notes": [
            "No model was retrained; all ablations reuse the saved signal/vol/multiplier.",
            "Official score uses target_metric as fwd and rf=0; exact hidden fwd/rf are unavailable for live rows.",
            "Shuffle preserves each day's vol and multiplier; only the signal order changes.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "manifest": manifest,
            "variants": variant_rows,
            "shuffle": {"iid": iid_summary, "block": block_summary},
            "signal_bins": bin_summary,
            "monthly": monthly,
            "regime": regime,
        },
    )

    print(variant_df.select(
        ["variant", "score", "raw_sharpe", "cumulative_return", "max_drawdown", "turnover", "mean_allocation"]
    ))
    print(json.dumps({"iid": iid_summary, "block": block_summary, "bins": bin_summary}, indent=2))
    print(f"wrote {output}")


def _plot_variants(frame: pl.DataFrame, path: Path) -> None:
    labels = frame["variant"].to_list()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, frame["score"].to_numpy(), color="tab:blue", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("official score (proxy)")
    ax.set_title("Signal and allocation ablations, 137-day window")
    for index, value in enumerate(frame["score"].to_numpy()):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_shuffle(iid: np.ndarray, block: np.ndarray, iid_summary: dict[str, Any], block_summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, values, summary, title in zip(
        axes,
        [iid, block],
        [iid_summary, block_summary],
        ["IID signal shuffle", "Block signal shuffle (L=20)"],
    ):
        finite = values[np.isfinite(values)]
        ax.hist(finite, bins=45, color="tab:gray", alpha=0.7)
        ax.axvline(summary["actual_score"], color="tab:red", linewidth=2, label="actual")
        ax.axvline(summary["null_p95"], color="black", linestyle="--", linewidth=0.8, label="null p95")
        ax.set_title(title)
        ax.set_xlabel("official score")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("repetitions")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_bins(frame: pl.DataFrame, path: Path) -> None:
    x = frame["bin"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x, frame["mean_target"].to_numpy(), color="tab:green", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("prediction quintile (low to high)")
    ax.set_ylabel("mean target_metric")
    ax.set_title("Prediction bins: high signal vs future return")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_monthly(frame: pl.DataFrame, path: Path) -> None:
    x = np.arange(frame.height)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x, frame["cumulative_return"].to_numpy() * 100.0, color="tab:orange", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["month"].to_list(), rotation=45, ha="right")
    ax.set_ylabel("monthly cumulative return (%)")
    ax.set_title("Monthly decomposition")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_regime(frame: pl.DataFrame, path: Path) -> None:
    x = np.arange(frame.height)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x, frame["score"].to_numpy(), color="tab:purple", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["regime"].to_list())
    ax.set_ylabel("official score (proxy)")
    ax.set_title("Volatility-regime decomposition")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_paths(frame: pl.DataFrame, variants: pl.DataFrame, path: Path) -> None:
    # Reconstruct variant paths for the most interpretable four.
    # The saved frame contains only the original allocation; use the variant
    # scores for labels and the original/allocation path for a direct visual.
    # A separate allocation path is already in the Phase 3/4 reports; this plot
    # focuses on the original signal's allocation and the raw signal bins.
    x = frame[COL_DATE_ID].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, frame["allocation"].to_numpy(), color="tab:blue", linewidth=0.8, label="original")
    ax.plot(x, frame["raw_pred"].to_numpy() * 600.0, color="tab:gray", linewidth=0.6, alpha=0.6, label="raw × 600")
    ax.set_xlabel("date_id")
    ax.set_ylabel("allocation / signal")
    ax.set_title("Original allocation path and raw signal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
