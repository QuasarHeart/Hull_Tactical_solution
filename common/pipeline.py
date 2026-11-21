"""Shared orchestration: fit -> walk-forward predict -> score.

The date loop, the online-update protocol and the allocation policy live here.
Model families only implement `Predictor` (fit/predict/update) and a
`DatasetBuilder` (panel -> Batch). This keeps GBDT and DL scores comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from .allocation import AllocationContext
from .config import COL_FWD, COL_RF, COL_TARGET_METRIC
from .data import attach_labels
from .metrics import ScoreReport, ic_spearman, official_score, proxy_score
from .model import Batch, DatasetBuilder, Predictor
from .splits import Fold


@dataclass
class FoldResult:
    name: str
    predictions: pl.DataFrame
    scored: pl.DataFrame
    report: dict[str, Any]


@dataclass
class CVResult:
    folds: list[FoldResult]
    oof: pl.DataFrame
    summary: dict[str, Any]


def score_predictions(scored: pl.DataFrame, allocation_col: str = "allocation") -> ScoreReport:
    """Official score when fwd/rf exist, else the target-only proxy."""
    pos = scored[allocation_col].to_numpy()
    if COL_FWD in scored.columns and COL_RF in scored.columns:
        fwd = scored[COL_FWD].to_numpy()
        rf = scored[COL_RF].to_numpy()
        if np.isfinite(fwd).all() and np.isfinite(rf).all():
            return official_score(pos, fwd, rf)
    if COL_TARGET_METRIC not in scored.columns:
        raise ValueError("Cannot score: neither fwd/rf nor target_metric is available")
    return proxy_score(pos, scored[COL_TARGET_METRIC].to_numpy())


class FullPipeline:
    def __init__(
        self,
        predictor: Predictor,
        builder: DatasetBuilder,
        policy: Any,
        date_col: str = "date_id",
        online_update: bool = False,
        update_lag: int = 1,
    ) -> None:
        self.predictor = predictor
        self.builder = builder
        self.policy = policy
        self.date_col = date_col
        self.online_update = bool(online_update)
        self.update_lag = int(update_lag)
        self.train_batch: Batch | None = None

    def fit(
        self,
        panel: pl.DataFrame,
        train_dates: Sequence[Any],
        tune_dates: Sequence[Any] | None = None,
    ) -> "FullPipeline":
        train_dates = list(train_dates)
        self.builder.fit(panel, train_dates)
        train_batch = self.builder.build(panel, train_dates, with_target=True)
        valid_batch = None
        if tune_dates:
            valid_batch = self.builder.build(panel, list(tune_dates), with_target=True)
        self.predictor.fit(train_batch, valid_batch)
        self.policy.fit(train_batch)
        self.train_batch = train_batch
        return self

    def predict_raw_one(self, panel: pl.DataFrame, date: Any) -> float:
        batch = self.builder.build(panel, [date], with_target=False)
        raw = np.asarray(self.predictor.predict(batch), dtype=float).reshape(-1)
        if raw.size == 0:
            raise ValueError(f"Predictor returned no prediction for {date}")
        return float(raw[0])

    def _predict_one(self, panel: pl.DataFrame, date: Any) -> dict[str, Any]:
        batch = self.builder.build(panel, [date], with_target=False)
        raw = np.asarray(self.predictor.predict(batch), dtype=float).reshape(-1)
        raw_pred = float(raw[0])
        context = self.builder.context(panel, date)
        allocation = float(self.policy.transform(raw_pred, context))

        row: dict[str, Any] = {
            self.date_col: date,
            "raw_pred": raw_pred,
            "allocation": allocation,
        }
        if hasattr(self.predictor, "predict_components"):
            components_fn = getattr(self.predictor, "predict_components")
            components = components_fn(batch)
            for key, value in components.items():
                row[f"raw_pred_{key}"] = float(np.asarray(value, dtype=float).reshape(-1)[0])
        if isinstance(context, AllocationContext):
            row["vol"] = context.vol
            row["v1"] = context.v1
            row["v1_median"] = context.v1_median
            row["multiplier"] = context.multiplier
        elif isinstance(context, dict):
            row.update(context)
        return row

    def predict_dates(
        self,
        panel: pl.DataFrame,
        dates: Sequence[Any],
        online_update: bool | None = None,
        initial_allocation: float = 0.0,
    ) -> pl.DataFrame:
        online = self.online_update if online_update is None else bool(online_update)
        self.policy.reset(initial_allocation)
        rows: list[dict[str, Any]] = []
        previous: Any | None = None

        for date in list(dates):
            if online and previous is not None:
                update_batch = self.builder.build(panel, [previous], with_target=True)
                if update_batch.y is not None and np.isfinite(update_batch.y).all():
                    self.predictor.update(update_batch)
                    self.policy.update(float(update_batch.y[0]))
            rows.append(self._predict_one(panel, date))
            previous = date

        if not rows:
            raise ValueError("predict_dates received no dates")
        return pl.DataFrame(rows)


class PipelineCV:
    def __init__(
        self,
        pipeline_factory: Callable[[], FullPipeline],
        date_col: str = "date_id",
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.date_col = date_col

    def fit(self, panel: pl.DataFrame, folds: Sequence[Fold]) -> CVResult:
        results: list[FoldResult] = []
        for fold in folds:
            pipeline = self.pipeline_factory()
            pipeline.fit(panel, fold.train_dates, fold.tune_dates or None)
            predictions = pipeline.predict_dates(panel, fold.test_dates)
            scored = attach_labels(panel, predictions, date_col=self.date_col)
            report = score_predictions(scored)
            report_dict = report.as_dict()
            report_dict["ic"] = ic_spearman(
                scored["target_metric"].to_numpy(), scored["raw_pred"].to_numpy()
            )
            report_dict["direction_accuracy"] = float(
                np.mean(
                    np.sign(scored["target_metric"].to_numpy())
                    == np.sign(scored["raw_pred"].to_numpy())
                )
            )
            results.append(
                FoldResult(
                    name=fold.name,
                    predictions=predictions,
                    scored=scored,
                    report=report_dict,
                )
            )

        oof = pl.concat([r.predictions for r in results], how="vertical")
        scores = [r.report["score"] for r in results]
        ics = [r.report["ic"] for r in results]
        directions = [r.report["direction_accuracy"] for r in results]
        summary = {
            "score_mean": float(np.mean(scores)),
            "score_worst": float(np.min(scores)),
            "score_best": float(np.max(scores)),
            "ic_mean": float(np.nanmean(ics)),
            "direction_mean": float(np.nanmean(directions)),
            "folds": {r.name: r.report for r in results},
        }
        return CVResult(folds=results, oof=oof, summary=summary)


class PipelineEnsemble:
    """Average raw predictions of several fitted pipelines, then size once."""

    def __init__(
        self,
        pipelines: Sequence[FullPipeline],
        weights: Sequence[float] | None = None,
        date_col: str = "date_id",
    ) -> None:
        if not pipelines:
            raise ValueError("PipelineEnsemble requires at least one pipeline")
        self.pipelines = list(pipelines)
        self.weights = (
            np.asarray(weights, dtype=float)
            if weights is not None
            else np.ones(len(self.pipelines), dtype=float)
        )
        if len(self.weights) != len(self.pipelines):
            raise ValueError("weights length must match pipelines")
        self.weights = self.weights / self.weights.sum()
        self.date_col = date_col
        self.builder = self.pipelines[0].builder
        self.policy = self.pipelines[0].policy

    def fit(
        self,
        panel: pl.DataFrame,
        train_dates: Sequence[Any],
        tune_dates: Sequence[Any] | None = None,
    ) -> "PipelineEnsemble":
        for pipeline in self.pipelines:
            pipeline.fit(panel, train_dates, tune_dates)
        return self

    def predict_dates(self, panel: pl.DataFrame, dates: Sequence[Any]) -> pl.DataFrame:
        self.policy.reset()
        rows: list[dict[str, Any]] = []
        for date in list(dates):
            raws = np.asarray(
                [p.predict_raw_one(panel, date) for p in self.pipelines], dtype=float
            )
            raw_pred = float(np.dot(self.weights, raws))
            context = self.builder.context(panel, date)
            allocation = float(self.policy.transform(raw_pred, context))
            row: dict[str, Any] = {
                self.date_col: date,
                "raw_pred": raw_pred,
                "allocation": allocation,
            }
            for index, value in enumerate(raws):
                row[f"raw_pred_model{index}"] = float(value)
            if isinstance(context, AllocationContext):
                row["vol"] = context.vol
                row["v1"] = context.v1
                row["v1_median"] = context.v1_median
                row["multiplier"] = context.multiplier
            rows.append(row)
        return pl.DataFrame(rows)
