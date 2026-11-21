"""Daily Hull Tactical dataset ingestion.

Source:
    https://www.hulltactical.com/kaggle-dataset-download/
      -> 302 -> https://www.hulltactical.com/wp-content/uploads/kaggle/YYYYMMDD_data.csv

The file is a full history: 96 columns = an empty-name date column + `target` +
94 features. Missing values are the string `NA`; the last few target values are
pending (not yet realized). The file's row index is the same `date_id` used by
the competition train set.

The ingest is deliberately boring: download to an immutable raw file, validate,
attach exact competition `fwd/rf` labels for the overlap, and atomically replace
a live panel parquet. A state file makes reruns idempotent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    COL_DATE,
    COL_DATE_ID,
    COL_FWD,
    COL_RF,
    COL_TARGET,
    COL_TARGET_METRIC,
    TARGET_OFFSET,
    Config,
    DataSourceSettings,
)
from .schema import N_FEATURES, feature_columns

SOURCE_DATE_RE = re.compile(r"/(\d{8})_data\.csv(?:$|[?#])")
MIN_FILE_BYTES = 1_000_000


@dataclass
class DailySource:
    data_date: date
    url: str


@dataclass
class IngestResult:
    status: str  # downloaded | skipped | dry_run | checked
    data_date: date
    url: str
    raw_path: str | None = None
    panel_path: str | None = None
    rows: int = 0
    target_rows: int = 0
    sha256: str | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "data_date": self.data_date.isoformat(),
            "url": self.url,
            "raw_path": self.raw_path,
            "panel_path": self.panel_path,
            "rows": self.rows,
            "target_rows": self.target_rows,
            "sha256": self.sha256,
            "message": self.message,
            "warnings": self.warnings,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_session(settings: DataSourceSettings) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=settings.retries,
        connect=settings.retries,
        read=settings.retries,
        status=settings.retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": settings.user_agent,
            "Accept": "text/csv,text/plain,*/*",
        }
    )
    return session


def resolve_latest(
    settings: DataSourceSettings,
    session: requests.Session | None = None,
) -> DailySource:
    """Resolve the latest dated CSV without downloading its body."""
    session = session or make_session(settings)
    with session.get(
        settings.base_url,
        timeout=settings.timeout_seconds,
        allow_redirects=True,
        stream=True,
    ) as response:
        response.raise_for_status()
        final_url = response.url
    match = SOURCE_DATE_RE.search(final_url)
    if not match:
        raise RuntimeError(
            f"Could not find a YYYYMMDD_data.csv date in the resolved URL: {final_url}"
        )
    return DailySource(
        data_date=datetime.strptime(match.group(1), "%Y%m%d").date(),
        url=final_url,
    )


def resolve_for_date(
    settings: DataSourceSettings,
    data_date: date,
) -> DailySource:
    base = settings.base_url
    if "/kaggle-dataset-download" in base:
        directory = base.rsplit("/kaggle-dataset-download", 1)[0]
    else:
        directory = base.rstrip("/")
    url = f"{directory}/wp-content/uploads/kaggle/{data_date:%Y%m%d}_data.csv"
    return DailySource(data_date=data_date, url=url)


def download(
    source: DailySource,
    raw_dir: str | Path,
    settings: DataSourceSettings,
    session: requests.Session | None = None,
    force: bool = False,
) -> tuple[Path, str]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{source.data_date:%Y%m%d}_data.csv"
    if destination.exists() and not force:
        return destination, sha256_file(destination)

    session = session or make_session(settings)
    temp = destination.with_suffix(".csv.part")
    try:
        with session.get(
            source.url,
            timeout=settings.timeout_seconds,
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            with temp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        size = temp.stat().st_size
        if size < MIN_FILE_BYTES:
            head = temp.read_text(encoding="utf-8", errors="ignore")[:200]
            raise RuntimeError(
                f"Downloaded file is too small ({size} bytes); response starts with: {head!r}"
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)
    return destination, sha256_file(destination)


def expected_feature_columns(train_csv: str | Path) -> list[str]:
    header = pl.read_csv(train_csv, n_rows=0)
    features = feature_columns(header)
    if len(features) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features in {train_csv}, found {len(features)}")
    return features


def read_daily_csv(
    path: str | Path,
    expected_features: Iterable[str] | None = None,
) -> pl.DataFrame:
    raw = pl.read_csv(
        path,
        null_values=["NA"],
        try_parse_dates=False,
        infer_schema_length=10_000,
    )
    if raw.width != 96:
        raise ValueError(f"Expected 96 columns in daily CSV, found {raw.width}")
    first = raw.columns[0]
    second = raw.columns[1]
    raw = raw.rename({first: COL_DATE, second: COL_TARGET})
    if COL_DATE not in raw.columns or COL_TARGET not in raw.columns:
        raise ValueError("Daily CSV must start with date, target columns")

    raw = raw.with_columns(pl.exclude(COL_DATE).cast(pl.Float64, strict=False))
    raw = raw.with_columns(pl.col(COL_DATE).cast(pl.Utf8).str.to_date("%Y-%m-%d"))
    raw = raw.with_columns(
        pl.int_range(0, pl.len(), dtype=pl.Int64).alias(COL_DATE_ID)
    )
    raw = raw.with_columns(
        pl.when(pl.col(COL_TARGET).is_not_null())
        .then(pl.col(COL_TARGET) + TARGET_OFFSET)
        .otherwise(None)
        .alias(COL_TARGET_METRIC),
        pl.lit(None, dtype=pl.Float64).alias(COL_FWD),
        pl.lit(None, dtype=pl.Float64).alias(COL_RF),
    )

    features = list(expected_features) if expected_features is not None else feature_columns(raw)
    if len(features) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, found {len(features)}")
    missing = sorted(set(features) - set(raw.columns))
    extra = sorted(set(feature_columns(raw)) - set(features))
    if missing or extra:
        raise ValueError(
            f"Daily feature columns mismatch. Missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )
    ordered = [COL_DATE, COL_DATE_ID, COL_TARGET, COL_TARGET_METRIC, COL_FWD, COL_RF]
    return raw.select(ordered + features)


def attach_competition_labels(
    daily: pl.DataFrame,
    train_csv: str | Path,
) -> pl.DataFrame:
    """Use exact fwd/rf labels for the competition overlap (date_id <= 9047)."""
    competition = pl.read_csv(train_csv)
    competition = competition.rename({"market_forward_excess_returns": COL_TARGET})
    competition = competition.with_columns(
        pl.exclude(COL_DATE_ID).cast(pl.Float64, strict=False)
    )
    competition = competition.with_columns(
        (pl.col(COL_FWD) - pl.col(COL_RF)).alias("_comp_target_metric")
    )
    competition = competition.select(
        [COL_DATE_ID, "_comp_target_metric", COL_FWD, COL_RF]
    )
    joined = daily.join(competition, on=COL_DATE_ID, how="left", suffix="_comp")
    joined = joined.with_columns(
        pl.when(pl.col("_comp_target_metric").is_not_null())
        .then(pl.col("_comp_target_metric"))
        .otherwise(pl.col(COL_TARGET_METRIC))
        .alias(COL_TARGET_METRIC),
        pl.coalesce([pl.col(f"{COL_FWD}_comp"), pl.col(COL_FWD)]).alias(COL_FWD),
        pl.coalesce([pl.col(f"{COL_RF}_comp"), pl.col(COL_RF)]).alias(COL_RF),
    )
    return joined.drop(["_comp_target_metric", f"{COL_FWD}_comp", f"{COL_RF}_comp"])


def validate_daily(
    df: pl.DataFrame,
    expected_features: Iterable[str],
    date_map: pl.DataFrame | None = None,
    require_alignment: bool = True,
    min_rows: int = 1000,
    min_targets: int = 9000,
) -> list[str]:
    expected = list(expected_features)
    warnings: list[str] = []
    if df.height < min_rows:
        raise ValueError(f"Daily CSV has only {df.height} rows; expected a full history")
    if df[COL_DATE].n_unique() != df.height:
        raise ValueError("Daily CSV contains duplicate dates")
    if not df[COL_DATE].is_sorted():
        raise ValueError("Daily CSV is not sorted by date")
    if list(df[COL_DATE_ID].to_list()) != list(range(df.height)):
        raise ValueError("date_id is not the contiguous row index")
    features = feature_columns(df)
    if set(features) != set(expected) or len(features) != N_FEATURES:
        raise ValueError("Daily feature columns do not match the training schema")

    target = df[COL_TARGET]
    null_count = target.null_count()
    if null_count > 5:
        raise ValueError(f"Too many null targets: {null_count}")
    null_tail = target.is_null().to_list()[-max(null_count, 1) :]
    if null_count and not all(null_tail):
        raise ValueError("Null targets are not confined to the end of the file")
    if df.height - null_count < min_targets:
        raise ValueError(f"Only {df.height - null_count} realized targets; file looks truncated")

    recent = df.tail(250)
    high_missing = {
        column: float(recent[column].is_null().mean())
        for column in expected
        if recent[column].is_null().mean() > 0.10
    }
    if high_missing:
        warnings.append(
            "features with >10% missing in the last 250 rows: "
            + ", ".join(f"{k}={v:.1%}" for k, v in sorted(high_missing.items()))
        )
    severe = {column: rate for column, rate in high_missing.items() if rate > 0.90}
    if severe:
        raise ValueError(f"Features missing for almost the whole recent window: {severe}")

    latest_missing = [column for column in expected if df.tail(1)[column].is_null()[0]]
    if latest_missing:
        warnings.append(
            f"latest row is missing {len(latest_missing)} features: {latest_missing[:10]}"
        )
    if len(latest_missing) > 20:
        raise ValueError(
            f"Latest row is missing {len(latest_missing)} features; source looks broken"
        )

    live = df.filter(pl.col(COL_DATE_ID) > 9047)
    if live.height:
        if live[COL_FWD].null_count() != live.height:
            raise ValueError("Live rows unexpectedly have forward_returns")
        if live[COL_RF].null_count() != live.height:
            raise ValueError("Live rows unexpectedly have risk_free_rate")

    if require_alignment and date_map is not None and df.height >= 9048:
        expected_dates = date_map.head(9048)[COL_DATE].to_list()
        actual_dates = df.head(9048)[COL_DATE].to_list()
        if actual_dates != expected_dates:
            first_bad = next(
                (i for i, (a, b) in enumerate(zip(actual_dates, expected_dates)) if a != b),
                None,
            )
            raise ValueError(
                "Daily dates do not align with the competition date map; "
                f"first mismatch at index {first_bad}"
            )
    return warnings


def upsert_panel(
    panel_path: str | Path,
    daily: pl.DataFrame,
) -> pl.DataFrame:
    """Atomically replace the live panel with the full daily history."""
    panel_path = Path(panel_path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    if panel_path.exists():
        previous = pl.read_parquet(panel_path)
        if daily.height < previous.height:
            raise ValueError(
                f"Refusing to shrink panel from {previous.height} to {daily.height} rows"
            )
    temp = panel_path.with_suffix(".parquet.part")
    daily.write_parquet(temp)
    os.replace(temp, panel_path)
    return daily


def load_panel_live(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort(COL_DATE_ID)


def upsert_prediction_history(
    path: str | Path,
    record: dict[str, Any],
) -> pl.DataFrame:
    """Append/replace one daily prediction by data_date (idempotent reruns)."""
    if "data_date" not in record:
        raise ValueError("Prediction record must include data_date")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pl.DataFrame([record])
    if path.exists():
        previous = pl.read_parquet(path)
        combined = pl.concat([previous, new_row], how="diagonal_relaxed")
    else:
        combined = new_row
    combined = combined.unique(subset=["data_date"], keep="last").sort("data_date")
    temp = path.with_suffix(".parquet.part")
    combined.write_parquet(temp)
    os.replace(temp, path)
    return combined


def load_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.part")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, default=str)
    os.replace(temp, path)


def append_log(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def write_health(
    path: str | Path,
    panel: pl.DataFrame,
    state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last_date = panel[COL_DATE].max() if panel.height else None
    target_rows = int(panel[COL_TARGET].drop_nulls().len())
    health = {
        "updated_at": utc_now(),
        "last_data_date": last_date.isoformat() if last_date else None,
        "lag_days": (date.today() - last_date).days if last_date else None,
        "rows": panel.height,
        "target_rows": target_rows,
        "pending_targets": panel.height - target_rows,
        "last_ingest_at": state.get("last_ingested_at"),
        "last_ingested_date": state.get("last_data_date"),
        "last_prediction": state.get("last_prediction"),
        "warnings": state.get("warnings", []),
    }
    if extra:
        health.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.part")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(health, handle, indent=2, sort_keys=True, default=str)
    os.replace(temp, path)
    return health


def ingest_latest(
    config: Config,
    force: bool = False,
    dry_run: bool = False,
    session: requests.Session | None = None,
) -> IngestResult:
    settings = config.data_source
    paths = config.paths
    paths.ensure()

    session = session or make_session(settings)
    source = resolve_latest(settings, session)
    state = load_state(paths.state_path)
    raw_path = paths.raw_dir / f"{source.data_date:%Y%m%d}_data.csv"

    if (
        not force
        and state.get("last_data_date") == source.data_date.isoformat()
        and raw_path.exists()
        and paths.panel_path.exists()
    ):
        panel = load_panel_live(paths.panel_path)
        return IngestResult(
            status="skipped",
            data_date=source.data_date,
            url=source.url,
            raw_path=str(raw_path),
            panel_path=str(paths.panel_path),
            rows=panel.height,
            target_rows=int(panel[COL_TARGET].drop_nulls().len()),
            sha256=state.get("sha256"),
            message="Latest source already ingested",
            warnings=state.get("warnings", []),
        )

    raw_path, digest = download(
        source, paths.raw_dir, settings, session=session, force=force
    )
    expected = expected_feature_columns(paths.train_csv)
    daily = read_daily_csv(raw_path, expected)
    daily = attach_competition_labels(daily, paths.train_csv)
    date_map = None
    if paths.date_map_csv.exists():
        from .data import load_date_map

        date_map = load_date_map(paths.date_map_csv)
    warnings: list[str] = validate_daily(
        daily,
        expected,
        date_map=date_map,
        require_alignment=settings.require_competition_alignment,
    )

    target_rows = int(daily[COL_TARGET].drop_nulls().len())
    if dry_run:
        return IngestResult(
            status="dry_run",
            data_date=source.data_date,
            url=source.url,
            raw_path=str(raw_path),
            panel_path=None,
            rows=daily.height,
            target_rows=target_rows,
            sha256=digest,
            message="Validated; panel/state not written",
            warnings=warnings,
        )

    panel = upsert_panel(paths.panel_path, daily)
    state = {
        "last_data_date": source.data_date.isoformat(),
        "last_ingested_at": utc_now(),
        "sha256": digest,
        "rows": panel.height,
        "target_rows": target_rows,
        "source_url": source.url,
        "last_status": "ok",
        "warnings": warnings,
    }
    save_state(paths.state_path, state)
    write_health(paths.health_path, panel, state, extra={"warnings": warnings})
    append_log(
        paths.log_path,
        {
            "ts": utc_now(),
            "status": "ok",
            "data_date": source.data_date.isoformat(),
            "url": source.url,
            "raw_path": str(raw_path),
            "sha256": digest,
            "rows": panel.height,
            "target_rows": target_rows,
            "warnings": warnings,
        },
    )
    return IngestResult(
        status="downloaded",
        data_date=source.data_date,
        url=source.url,
        raw_path=str(raw_path),
        panel_path=str(paths.panel_path),
        rows=panel.height,
        target_rows=target_rows,
        sha256=digest,
        warnings=warnings,
    )
