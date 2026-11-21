"""Run directories, manifests and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: list[str] | None = None) -> dict[str, str]:
    names = names or [
        "numpy",
        "pandas",
        "polars",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "torch",
        "wandb",
        "pyarrow",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return slug[:40] or "run"


def make_run_id(run_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slug(run_name)}_{uuid.uuid4().hex[:6]}"


def create_run_dir(models_dir: str | Path, run_id: str) -> Path:
    path = Path(models_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_manifest(
    run_id: str,
    run_name: str,
    config: dict[str, Any],
    data_paths: dict[str, str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, path in (data_paths or {}).items():
        path = Path(path)
        data[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    manifest = {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "data": data,
        "config": config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(run_dir) / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    return path


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
    return path
