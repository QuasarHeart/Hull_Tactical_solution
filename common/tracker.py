"""Experiment tracking: Weights & Biases with a no-op fallback.

W&B is initialised explicitly by the entry point (not on import), so a run can
be built and verified locally before the API key is configured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, WandbSettings, wandb_api_key_available


class BaseTracker:
    url: str | None = None

    def init(self) -> bool:
        return False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_table(self, name: str, data: Any, step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str | Path, name: str, artifact_type: str = "model") -> None:
        return None

    def set_summary(self, values: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class NullTracker(BaseTracker):
    """Used for `--no-wandb` verification runs and unit tests."""


class WandbTracker(BaseTracker):
    def __init__(
        self,
        settings: WandbSettings,
        run_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.run_name = run_name
        self.config = config or {}
        self._wandb = None
        self._initialised = False

    def init(self) -> bool:
        if not self.settings.enabled:
            return False
        mode = self.settings.mode or os.environ.get("WANDB_MODE")
        if mode == "disabled":
            return False
        if mode in (None, "online") and not wandb_api_key_available():
            raise RuntimeError(
                "WANDB_API_KEY is not set. Put it in .env (or export it), then rerun. "
                "Use --no-wandb to verify the pipeline without tracking."
            )
        import wandb

        self._wandb = wandb
        settings = wandb.Settings(base_url=self.settings.base_url)
        wandb.init(
            project=self.settings.project,
            entity=self.settings.entity,
            name=self.run_name,
            config=self.config,
            tags=self.settings.tags or None,
            mode=mode,
            settings=settings,
            dir=str(PROJECT_ROOT),
            save_code=True,
        )
        self._initialised = True
        self.url = getattr(wandb.run, "url", None)
        return True

    def _require(self) -> Any:
        if not self._initialised or self._wandb is None:
            raise RuntimeError("WandbTracker.init() must be called before logging")
        return self._wandb

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        wandb = self._require()
        wandb.log(metrics, step=step)

    def log_table(self, name: str, data: Any, step: int | None = None) -> None:
        wandb = self._require()
        frame = data.to_pandas() if hasattr(data, "to_pandas") else pd.DataFrame(data)
        table = wandb.Table(dataframe=frame)
        wandb.log({name: table}, step=step)

    def log_artifact(self, path: str | Path, name: str, artifact_type: str = "model") -> None:
        if not self.settings.log_artifacts:
            return
        wandb = self._require()
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)

    def set_summary(self, values: dict[str, Any]) -> None:
        wandb = self._require()
        wandb.summary.update(values)

    def finish(self) -> None:
        if self._initialised and self._wandb is not None:
            self._wandb.finish()
            self._initialised = False


def make_tracker(
    settings: WandbSettings,
    run_name: str,
    config: dict[str, Any] | None = None,
    enabled: bool = True,
) -> BaseTracker:
    if not enabled or not settings.enabled:
        return NullTracker()
    if settings.mode == "disabled" or os.environ.get("WANDB_MODE") == "disabled":
        return NullTracker()
    return WandbTracker(settings=settings, run_name=run_name, config=config)
