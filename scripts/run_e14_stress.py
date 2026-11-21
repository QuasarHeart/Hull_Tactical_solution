"""Post-competition E14 stress test on the live panel.

The six-fold OOF ends on 2025-12-03; the recent E14 outage starts on 2026-06-16,
so it cannot be evaluated inside the six-fold CV. This script evaluates three
feature treatments on the 108 realized days where E14 is missing:

    current_median : the production builder (median fills E14)
    drop_e14       : remove E14 from the model
    e14_indicator  : median fill + binary E14_MISSING feature

Every prediction trains on the last 800 realized rows before that date. Scores
are proxy scores (fwd = target_metric, rf=0), because the live source does not
publish forward_returns/risk_free_rate. This is an exploratory holdout, not a
replacement for the six-fold official CV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common.allocation import NotebookAllocationPolicy
from common.config import COL_DATE_ID, COL_FWD, COL_RF, COL_TARGET, COL_TARGET_METRIC
from common.evaluation import summarize_positions
from common.registry import git_sha, utc_now, write_json
from gdbt.config import GBDTConfig
from gdbt.features import NotebookFeatureBuilder
from gdbt.model import SklearnEnsemble

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


class DropE14Builder(NotebookFeatureBuilder):
    """Notebook builder with E14 removed after feature selection."""

    def fit(self, panel: pl.DataFrame, dates: Sequence[Any]) -> "DropE14Builder":
        super().fit(panel, dates)
        self.features = [feature for feature in self.features if feature != "E14"]
        self.base_features = [feature for feature in self.base_features if feature != "E14"]
        self.raw_feature_columns = [
            feature for feature in self.raw_feature_columns if feature != "E14"
        ]
        self._train_batch = self._make_batch(self.train_frame, with_target=True)
        return self


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/lake/panel_live.parquet")
    parser.add_argument("--config", default="configs/gdbt/cv.yaml")
    parser.add_argument("--output", default="reports/deployment_eval")
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=1)
    args = parser.parse_args()

    from common.config import Config

    config = Config.load(args.config)
    model_config = GBDTConfig.from_dict(config.section("model"))
    panel = pl.read_parquet(args.panel).sort(COL_DATE_ID)
    recent = panel.filter(
        pl.col("E14").is_null() & (pl.col("date") >= pl.date(2026, 1, 1))
    )
    test_dates = recent.filter(pl.col(COL_TARGET).is_not_null())[COL_DATE_ID].to_list()
    if args.limit_days:
        test_dates = test_dates[: int(args.limit_days)]

    # E14 is absent from every six-fold test day; only the post-competition
    # live window can expose it.
    overlap_days = recent.filter(pl.col(COL_DATE_ID) <= 9047).height
    missing_summary = {
        "live_rows": panel.height,
        "e14_missing_all": int(panel["E14"].null_count()),
        "e14_missing_2026": int(recent.height),
        "e14_missing_2026_realized": int(recent[COL_TARGET].is_not_null().sum()),
        "e14_missing_2026_first": str(recent["date"].min()),
        "e14_missing_2026_last": str(recent["date"].max()),
        "six_fold_test_end_date_id": 9047,
        "e14_missing_in_six_fold_test": int(overlap_days),
        "note": (
            "The six-fold OOF cannot measure the 2026-06-16 outage; this script "
            "uses the post-competition live holdout with proxy scores."
        ),
    }

    variants = {
        "current_median": panel,
        "drop_e14": panel,
        "e14_indicator": panel.with_columns(
            pl.col("E14").is_null().cast(pl.Float64).alias("E14_MISSING")
        ),
    }
    rows: list[dict[str, Any]] = []
    for variant, frame in variants.items():
        first_test = test_dates[0]
        initial = (
            frame.filter(pl.col(COL_DATE_ID) < first_test)
            .filter(pl.col(COL_TARGET).is_not_null())
            .sort(COL_DATE_ID)
            .tail(model_config.max_train_rows)
        )
        initial_dates = initial[COL_DATE_ID].to_list()
        builder_cls = DropE14Builder if variant == "drop_e14" else NotebookFeatureBuilder
        initial_builder = builder_cls(model_config)
        initial_builder.fit(frame, initial_dates)
        policy = make_policy(model_config)
        policy.fit(initial_builder.build(frame, initial_dates, with_target=True))

        positions: list[float] = []
        raw_values: list[float] = []
        missing_shares: list[float] = []
        last_model: SklearnEnsemble | None = None
        builder = None
        model = None
        for day_index, date_id in enumerate(test_dates):
            if model is None or day_index % int(args.retrain_every) == 0:
                train = (
                    frame.filter(pl.col(COL_DATE_ID) < date_id)
                    .filter(pl.col(COL_TARGET).is_not_null())
                    .sort(COL_DATE_ID)
                    .tail(model_config.max_train_rows)
                )
                train_dates = train[COL_DATE_ID].to_list()
                builder = builder_cls(model_config)
                builder.fit(frame, train_dates)
                model = SklearnEnsemble(model_config).fit(
                    builder.build(frame, train_dates, with_target=True)
                )
                missing_shares.append(
                    float(train["E14"].is_null().mean()) if "E14" in train.columns else 0.0
                )
                last_model = model
            test_batch = builder.build(frame, [date_id], with_target=False)
            raw = float(model.predict(test_batch)[0])
            context = builder.context(frame, date_id)
            allocation = policy.transform(raw, context)
            target = float(
                frame.filter(pl.col(COL_DATE_ID) == date_id)[COL_TARGET][0]
            )
            policy.update(target)
            positions.append(float(allocation))
            raw_values.append(raw)

        test_frame = frame.filter(pl.col(COL_DATE_ID).is_in(test_dates)).sort(COL_DATE_ID)
        proxy_fwd = test_frame[COL_TARGET_METRIC].to_numpy()
        proxy_rf = np.zeros_like(proxy_fwd)
        metrics = summarize_positions(np.asarray(positions), proxy_fwd, proxy_rf)
        rows.append(
            {
                "variant": variant,
                "n_days": len(positions),
                "score_proxy": metrics["score"],
                "sharpe_proxy": metrics["sharpe"],
                "cumulative_return_proxy": metrics["cumulative_return"],
                "max_drawdown_proxy": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
                "mean_allocation": metrics["mean_allocation"],
                "mean_raw_pred": float(np.mean(raw_values)),
                "mean_train_e14_missing_share": float(np.mean(missing_shares)),
                "final_model_features": len(last_model.features) if last_model else 0,
            }
        )

        # Save final model importance for diagnostics.
        if last_model is not None:
            importance = getattr(last_model.xgb, "feature_importances_", None)
            if importance is not None:
                pairs = sorted(
                    zip(last_model.features, np.asarray(importance, dtype=float)),
                    key=lambda item: item[1],
                    reverse=True,
                )
                (Path(args.output) / "plots").mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    [{"feature": name, "importance": float(value)} for name, value in pairs]
                ).write_csv(Path(args.output) / f"e14_importance_{variant}.csv")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output / "e14_stress.csv")
    write_json(
        output / "e14_stress.json",
        {
            "created_at": utc_now(),
            "git_sha": git_sha(),
            "missing_summary": missing_summary,
            "variants": rows,
            "proxy": "official score with fwd=target_metric and rf=0; not the leaderboard score",
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r["variant"] for r in rows]
    ax.bar(labels, [r["score_proxy"] for r in rows], color="tab:blue", alpha=0.75)
    for i, r in enumerate(rows):
        ax.text(i, r["score_proxy"], f"{r['score_proxy']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("proxy official score")
    ax.set_title("E14 stress test (2026-06-16 to 2026-11-17)")
    fig.tight_layout()
    fig.savefig(output / "plots" / "e14_stress.png")
    plt.close(fig)

    print(pl.DataFrame(rows))
    print(json.dumps(missing_summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
