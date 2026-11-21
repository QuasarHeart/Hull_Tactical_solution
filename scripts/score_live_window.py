"""Score a live competition window with a causal warm-up path.

Default window: 2026-06-25 through 2026-07-06 (inclusive). The model runs from
an earlier warm-up date so the allocation policy has a realistic prior path;
only the requested window is scored.

Modes:
  daily           dynamic feature builder + full daily refit (current D)
  fixed_rolling   fixed feature builder + rolling model update (B)
  offline         fixed feature builder + fixed model (A)
  vol_only        volatility target using a causal 20-day window
  buy_hold        constant 1.0

The live source has no forward_returns/risk_free_rate, so the official-style
score treats target_metric as forward returns and sets rf=0. This is a proxy,
not the exact hidden competition score.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from common.allocation import NotebookAllocationPolicy
from common.config import COL_DATE_ID, COL_FWD, COL_RF, COL_TARGET, COL_TARGET_METRIC, Config
from common.evaluation import summarize_positions
from common.experiments import causal_vol_estimates, simulate_vol_target
from common.metrics import sharpe as annualized_sharpe
from common.registry import utc_now, write_json
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder
from gdbt.model import SklearnEnsemble


def make_policy(model_config: GBDTConfig) -> NotebookAllocationPolicy:
    return NotebookAllocationPolicy(
        vol_window=model_config.vol_window,
        multiplier_low=model_config.signal_multiplier_low_vol,
        multiplier_high=model_config.signal_multiplier_high_vol,
        min_signal=model_config.min_signal,
        max_signal=model_config.max_signal,
        vol_scaling=model_config.vol_scaling,
        smoothing=model_config.smoothing,
        transaction_cost=model_config.transaction_cost,
    )


def initial_window(panel: pl.DataFrame, first_date: int, max_rows: int) -> pl.DataFrame:
    return (
        panel.filter(pl.col(COL_DATE_ID) < first_date)
        .filter(pl.col(COL_TARGET).is_not_null())
        .sort(COL_DATE_ID)
        .tail(max_rows)
    )


def predict_row(
    panel: pl.DataFrame,
    date_id: int,
    builder: NotebookFeatureBuilder,
    model: SklearnEnsemble,
    policy: NotebookAllocationPolicy,
) -> dict[str, Any]:
    batch = builder.build(panel, [date_id], with_target=False)
    components = model.predict_components(batch)
    raw = float(components["ensemble"][0])
    context = builder.context(panel, date_id)
    allocation = float(policy.transform(raw, context))
    row = panel.filter(pl.col(COL_DATE_ID) == date_id).row(0, named=True)
    return {
        "date_id": int(date_id),
        "raw_pred": raw,
        "raw_pred_enet": float(components["enet"][0]),
        "raw_pred_xgb": float(components["xgb"][0]),
        "raw_pred_lgb": float(components["lgb"][0]),
        "allocation": allocation,
        "vol": float(context.vol),
        "v1": float(context.v1) if context.v1 is not None else float("nan"),
        "v1_median": float(context.v1_median),
        "multiplier": float(context.multiplier),
        "target": float(row[COL_TARGET]),
        "target_metric": float(row[COL_TARGET_METRIC]),
        "forward_returns": float(row[COL_FWD]) if row[COL_FWD] is not None else float("nan"),
        "risk_free_rate": float(row[COL_RF]) if row[COL_RF] is not None else float("nan"),
    }


def run_daily(
    panel: pl.DataFrame,
    dates: list[int],
    model_config: GBDTConfig,
    initial_dates: list[int],
    initial_batch: Any,
) -> dict[str, Any]:
    policy = make_policy(model_config)
    policy.fit(initial_batch)
    rows = []
    for date_id in dates:
        train = (
            panel.filter(pl.col(COL_DATE_ID) < date_id)
            .filter(pl.col(COL_TARGET).is_not_null())
            .sort(COL_DATE_ID)
            .tail(model_config.max_train_rows)
        )
        builder = NotebookFeatureBuilder(model_config).fit(panel, train[COL_DATE_ID].to_list())
        model = SklearnEnsemble(model_config).fit(
            builder.build(panel, train[COL_DATE_ID].to_list(), with_target=True)
        )
        row = predict_row(panel, date_id, builder, model, policy)
        rows.append(row)
        policy.update(float(row["target"]))
    return {"mode": "daily", "rows": rows}


def run_fixed_rolling(
    panel: pl.DataFrame,
    dates: list[int],
    model_config: GBDTConfig,
    initial_dates: list[int],
    initial_batch: Any,
) -> dict[str, Any]:
    builder = NotebookFeatureBuilder(model_config).fit(panel, initial_dates)
    model = SklearnEnsemble(model_config).fit(initial_batch)
    policy = make_policy(model_config)
    policy.fit(initial_batch)
    rows = []
    previous: int | None = None
    for date_id in dates:
        if previous is not None:
            previous_target = float(
                panel.filter(pl.col(COL_DATE_ID) == previous)[COL_TARGET][0]
            )
            update_batch = builder.build(panel, [previous], with_target=True)
            model.update(update_batch)
            policy.update(previous_target)
        rows.append(predict_row(panel, date_id, builder, model, policy))
        previous = date_id
    return {"mode": "fixed_rolling", "rows": rows}


def run_offline(
    panel: pl.DataFrame,
    dates: list[int],
    model_config: GBDTConfig,
    initial_dates: list[int],
    initial_batch: Any,
) -> dict[str, Any]:
    builder = NotebookFeatureBuilder(model_config).fit(panel, initial_dates)
    model = SklearnEnsemble(model_config).fit(initial_batch)
    policy = make_policy(model_config)
    policy.fit(initial_batch)
    rows = []
    for date_id in dates:
        row = predict_row(panel, date_id, builder, model, policy)
        rows.append(row)
        policy.update(float(row["target"]))
    return {"mode": "offline", "rows": rows}


def run_vol_only(
    panel: pl.DataFrame,
    dates: list[int],
    model_config: GBDTConfig,
    initial_dates: list[int],
) -> dict[str, Any]:
    target_vol = causal_vol_estimates(
        panel, initial_dates, vol_window=model_config.vol_window
    )
    vol = causal_vol_estimates(panel, dates, vol_window=model_config.vol_window)
    positions = simulate_vol_target(
        vol,
        float(np.median(target_vol)),
        smoothing=model_config.smoothing,
        transaction_cost=model_config.transaction_cost,
    )
    rows = []
    for date_id, position in zip(dates, positions):
        row = panel.filter(pl.col(COL_DATE_ID) == date_id).row(0, named=True)
        rows.append(
            {
                "date_id": int(date_id),
                "raw_pred": 0.0,
                "allocation": float(position),
                "vol": float(
                    vol[dates.index(date_id)]
                ),
                "v1": float(row["V1"]) if row["V1"] is not None else float("nan"),
                "v1_median": float(np.nanmedian(panel.filter(pl.col(COL_DATE_ID).is_in(initial_dates))["V1"].to_numpy())),
                "multiplier": float("nan"),
                "target": float(row[COL_TARGET]),
                "target_metric": float(row[COL_TARGET_METRIC]),
            }
        )
    return {"mode": "vol_only", "rows": rows}


def run_buy_hold(panel: pl.DataFrame, dates: list[int], position: float = 1.0) -> dict[str, Any]:
    rows = []
    for date_id in dates:
        row = panel.filter(pl.col(COL_DATE_ID) == date_id).row(0, named=True)
        rows.append(
            {
                "date_id": int(date_id),
                "raw_pred": 0.0,
                "allocation": float(position),
                "vol": float("nan"),
                "v1": float(row["V1"]) if row["V1"] is not None else float("nan"),
                "v1_median": float("nan"),
                "multiplier": float("nan"),
                "target": float(row[COL_TARGET]),
                "target_metric": float(row[COL_TARGET_METRIC]),
            }
        )
    return {"mode": "buy_hold", "rows": rows}


def score_rows(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    window = [row for row in rows if start <= row["date_id"] <= end]
    if not window:
        raise ValueError("No rows in requested window")
    positions = np.asarray([row["allocation"] for row in window], dtype=float)
    target_metric = np.asarray([row["target_metric"] for row in window], dtype=float)
    target = np.asarray([row["target"] for row in window], dtype=float)
    # Live source has no fwd/rf: proxy with fwd=target_metric and rf=0.
    report = summarize_positions(positions, target_metric, np.zeros_like(target_metric))
    strategy_returns = positions * target_metric
    raw_excess = positions * target
    return {
        "n_days": len(window),
        "first_date_id": int(window[0]["date_id"]),
        "last_date_id": int(window[-1]["date_id"]),
        "proxy_official_score": float(report["score"]),
        "proxy_raw_sharpe": float(report["sharpe"]),
        "proxy_report": {
            key: report[key]
            for key in (
                "strategy_vol",
                "market_vol",
                "vol_penalty",
                "return_penalty",
                "strategy_mean_excess",
                "market_mean_excess",
            )
        },
        "strategy_mean_excess": float(np.mean(strategy_returns)),
        "target_mean": float(np.mean(target_metric)),
        "target_vol_daily": float(np.std(target_metric, ddof=1)),
        "target_sharpe": float(annualized_sharpe(target_metric)),
        "raw_excess_mean": float(np.mean(raw_excess)),
        "cumulative_return_proxy": float(report["cumulative_return"]),
        "max_drawdown_proxy": float(report["max_drawdown"]),
        "turnover": float(report["turnover"]),
        "mean_allocation": float(report["mean_allocation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/phase4.yaml")
    parser.add_argument("--panel", default="data/lake/panel_live.parquet")
    parser.add_argument("--start", default="2026-06-25")
    parser.add_argument("--end", default="2026-07-06")
    parser.add_argument("--warmup-start", default="2026-01-02")
    parser.add_argument("--modes", default="daily,fixed_rolling,offline,vol_only,buy_hold,constant_1.25")
    parser.add_argument("--output", default="reports/live_window_20260625_20260706")
    args = parser.parse_args()

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel = pl.read_parquet(args.panel).sort(COL_DATE_ID)
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    warmup_date = date.fromisoformat(args.warmup_start)

    def date_ids_between(low: date, high: date) -> list[int]:
        return (
            panel.filter(
                (pl.col("date") >= low)
                & (pl.col("date") <= high)
                & pl.col(COL_TARGET).is_not_null()
            )
            .sort(COL_DATE_ID)[COL_DATE_ID]
            .to_list()
        )

    dates = date_ids_between(warmup_date, end_date)
    if not dates:
        raise ValueError("No live dates in requested range")
    initial = initial_window(panel, dates[0], model_config.max_train_rows)
    initial_dates = initial[COL_DATE_ID].to_list()
    initial_builder = NotebookFeatureBuilder(model_config).fit(panel, initial_dates)
    initial_batch = initial_builder.build(panel, initial_dates, with_target=True)
    start_id = int(
        panel.filter(pl.col("date") >= start_date).sort(COL_DATE_ID)[COL_DATE_ID][0]
    )
    end_id = int(
        panel.filter(pl.col("date") <= end_date).sort(COL_DATE_ID)[COL_DATE_ID][-1]
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    all_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "daily":
            result = run_daily(panel, dates, model_config, initial_dates, initial_batch)
        elif mode == "fixed_rolling":
            result = run_fixed_rolling(panel, dates, model_config, initial_dates, initial_batch)
        elif mode == "offline":
            result = run_offline(panel, dates, model_config, initial_dates, initial_batch)
        elif mode == "vol_only":
            result = run_vol_only(panel, dates, model_config, initial_dates)
        elif mode == "buy_hold":
            result = run_buy_hold(panel, dates, 1.0)
        elif mode == "constant_1.25":
            result = run_buy_hold(panel, dates, 1.25)
        else:
            raise ValueError(f"Unknown mode {mode}")
        for row in result["rows"]:
            row["mode"] = mode
        all_rows.extend(result["rows"])
        result_metrics = score_rows(result["rows"], start_id, end_id)
        result_metrics["mode"] = mode
        metrics.append(result_metrics)
        print(f"{mode}: {json.dumps(result_metrics, default=str)}")

    pl.DataFrame(all_rows).sort(["mode", COL_DATE_ID]).write_csv(output / "positions.csv")
    flat_metrics: list[dict[str, Any]] = []
    for row in metrics:
        flat = {key: value for key, value in row.items() if key != "proxy_report"}
        for key, value in row.get("proxy_report", {}).items():
            flat[f"proxy_{key}"] = value
        flat_metrics.append(flat)
    pl.DataFrame(flat_metrics).write_csv(output / "metrics.csv")
    write_json(
        output / "summary.json",
        {
            "created_at": utc_now(),
            "start": args.start,
            "end": args.end,
            "warmup_start": args.warmup_start,
            "start_date_id": start_id,
            "end_date_id": end_id,
            "n_scored_days": int(sum(1 for value in dates if start_id <= value <= end_id)),
            "modes": modes,
            "metrics": metrics,
            "proxy_note": (
                "Live source has target and target_metric but no forward_returns/"
                "risk_free_rate. Proxy uses fwd=target_metric and rf=0. Exact hidden "
                "competition score is unavailable."
            ),
        },
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
