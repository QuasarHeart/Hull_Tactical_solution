"""Project configuration: paths, column names, YAML loading.

This module is deliberately dependency-light: it is imported by every other
package (including the dashboard), so it must not import polars/torch/wandb.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical column names. Competition data uses date_id; live data uses date.
COL_DATE = "date"
COL_DATE_ID = "date_id"
COL_TARGET = "target"
COL_TARGET_METRIC = "target_metric"
COL_FWD = "forward_returns"
COL_RF = "risk_free_rate"
COL_IS_SCORED = "is_scored"
COL_LAGGED_TARGET = "lagged_market_forward_excess_returns"

# train.csv's market_forward_excess_returns = fwd - rf - TARGET_OFFSET.
# Stable across 1990-2025 (mean 0.0003112, sd 7.1e-5). Used only to reconstruct
# a metric-aligned target for live rows; competition rows use fwd - rf directly.
TARGET_OFFSET = 0.0003112

WANDB_PROJECT = "hull_tactical"
RANDOM_SEED = 42

# Backwards-compatible constants for the old modules that still import them.
DATA_PATHS = {
    "train": str(PROJECT_ROOT / "data" / "train.csv"),
    "test": str(PROJECT_ROOT / "data" / "test.csv"),
}
base_path = PROJECT_ROOT


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _load_with_base(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = _read_yaml(path)
    base = raw.pop("_base_", None)
    if not base:
        return raw
    base_path = (path.parent / base).resolve()
    return _deep_merge(_load_with_base(base_path), raw)


def _resolve_path(value: str | None, default: str) -> Path:
    if value is None or value == "":
        return PROJECT_ROOT / default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass
class Paths:
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    train_csv: Path = PROJECT_ROOT / "data" / "train.csv"
    test_csv: Path = PROJECT_ROOT / "data" / "test.csv"
    date_map_csv: Path = PROJECT_ROOT / "data" / "date_map.csv"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "hull"
    panel_path: Path = PROJECT_ROOT / "data" / "lake" / "panel_live.parquet"
    predictions_path: Path = PROJECT_ROOT / "data" / "lake" / "predictions.parquet"
    state_path: Path = PROJECT_ROOT / "data" / "lake" / "ingest_state.json"
    log_path: Path = PROJECT_ROOT / "data" / "lake" / "ingest_log.jsonl"
    health_path: Path = PROJECT_ROOT / "data" / "lake" / "health.json"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.models_dir,
            self.reports_dir,
            self.raw_dir,
            self.panel_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataSourceSettings:
    base_url: str = "https://www.hulltactical.com/kaggle-dataset-download/"
    timeout_seconds: int = 60
    retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    max_lookback_days: int = 5
    require_competition_alignment: bool = True


@dataclass
class WandbSettings:
    enabled: bool = True
    project: str = WANDB_PROJECT
    entity: str | None = None
    mode: str | None = None
    base_url: str = "https://api.forge.coreweave.com"
    tags: list[str] = field(default_factory=list)
    log_artifacts: bool = True


@dataclass
class Config:
    seed: int = RANDOM_SEED
    run_name: str = "run"
    paths: Paths = field(default_factory=Paths)
    wandb: WandbSettings = field(default_factory=WandbSettings)
    data_source: DataSourceSettings = field(default_factory=DataSourceSettings)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "Config":
        raw = _load_with_base(Path(path))
        if overrides:
            raw = _deep_merge(raw, overrides)

        paths_raw = raw.get("paths", {}) or {}
        paths = Paths(
            data_dir=_resolve_path(paths_raw.get("data_dir"), "data"),
            models_dir=_resolve_path(paths_raw.get("models_dir"), "models"),
            reports_dir=_resolve_path(paths_raw.get("reports_dir"), "reports"),
            train_csv=_resolve_path(paths_raw.get("train_csv"), "data/train.csv"),
            test_csv=_resolve_path(paths_raw.get("test_csv"), "data/test.csv"),
            date_map_csv=_resolve_path(paths_raw.get("date_map_csv"), "data/date_map.csv"),
            raw_dir=_resolve_path(paths_raw.get("raw_dir"), "data/raw/hull"),
            panel_path=_resolve_path(
                paths_raw.get("panel_path"), "data/lake/panel_live.parquet"
            ),
            predictions_path=_resolve_path(
                paths_raw.get("predictions_path"), "data/lake/predictions.parquet"
            ),
            state_path=_resolve_path(
                paths_raw.get("state_path"), "data/lake/ingest_state.json"
            ),
            log_path=_resolve_path(
                paths_raw.get("log_path"), "data/lake/ingest_log.jsonl"
            ),
            health_path=_resolve_path(
                paths_raw.get("health_path"), "data/lake/health.json"
            ),
        )

        data_raw = raw.get("data_source", {}) or {}
        data_source = DataSourceSettings(
            base_url=str(
                data_raw.get("base_url")
                or "https://www.hulltactical.com/kaggle-dataset-download/"
            ),
            timeout_seconds=int(data_raw.get("timeout_seconds", 60)),
            retries=int(data_raw.get("retries", 3)),
            user_agent=str(
                data_raw.get("user_agent")
                or DataSourceSettings().user_agent
            ),
            max_lookback_days=int(data_raw.get("max_lookback_days", 5)),
            require_competition_alignment=bool(
                data_raw.get("require_competition_alignment", True)
            ),
        )

        wandb_raw = raw.get("wandb", {}) or {}
        wandb = WandbSettings(
            enabled=bool(wandb_raw.get("enabled", True)),
            project=str(
                wandb_raw.get("project")
                or os.environ.get("WANDB_PROJECT")
                or WANDB_PROJECT
            ),
            entity=(
                wandb_raw.get("entity")
                or os.environ.get("WANDB_ENTITY")
                or None
            ),
            mode=(
                wandb_raw.get("mode")
                or os.environ.get("WANDB_MODE")
                or None
            ),
            base_url=str(
                wandb_raw.get("base_url")
                or os.environ.get("WANDB_BASE_URL")
                or "https://api.forge.coreweave.com"
            ),
            tags=list(wandb_raw.get("tags", []) or []),
            log_artifacts=bool(wandb_raw.get("log_artifacts", True)),
        )
        return cls(
            seed=int(raw.get("seed", RANDOM_SEED)),
            run_name=str(raw.get("run_name", "run")),
            paths=paths,
            wandb=wandb,
            data_source=data_source,
            raw=raw,
        )

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {}) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{name}' must be a mapping")
        return value


def load_env(path: str | Path | None = None) -> None:
    """Load .env without printing or overriding existing environment variables."""
    from dotenv import load_dotenv

    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def wandb_api_key_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    if not netrc.exists():
        return False
    try:
        return "api.wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
