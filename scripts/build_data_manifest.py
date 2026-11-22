"""Build a reproducible manifest for tracked and local-only data artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_csv(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    frame = pl.read_csv(path)
    row: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": frame.height,
        "columns": frame.width,
        "column_names": frame.columns,
    }
    if "date_id" in frame.columns:
        row["date_id_min"] = int(frame["date_id"].min())
        row["date_id_max"] = int(frame["date_id"].max())
    if "date" in frame.columns:
        row["date_min"] = str(frame["date"].min())
        row["date_max"] = str(frame["date"].max())
    if "target" in frame.columns:
        row["target_nulls"] = int(frame["target"].null_count())
    return row


def summarize_parquet(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    frame = pl.read_parquet(path)
    row: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": frame.height,
        "columns": frame.width,
        "column_names": frame.columns,
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "target_nulls": int(frame["target"].null_count()),
        "git_tracked": False,
        "note": "generated live panel; data/lake/ is gitignored",
    }
    return row


def main() -> None:
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "Hull Tactical research pipeline",
        "data_contract": {
            "date_key": "date_id",
            "feature_count": 94,
            "target": "market_forward_excess_returns (train)",
            "scoring_proxy": "target_metric with rf=0 when forward_returns/risk_free_rate are unavailable",
        },
        "tracked": [
            summarize_csv("data/train.csv"),
            summarize_csv("data/test.csv"),
            summarize_csv("data/date_map.csv"),
        ],
        "generated": [summarize_parquet("data/lake/panel_live.parquet")],
        "raw_source": None,
    }

    raw_path = ROOT / "data" / "raw" / "hull" / "20261120_data.csv"
    if raw_path.exists():
        manifest["raw_source"] = {
            "path": "data/raw/hull/20261120_data.csv",
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "rows": 9291,
            "columns": 96,
            "source_url": "https://www.hulltactical.com/wp-content/uploads/kaggle/20261120_data.csv",
            "git_tracked": False,
            "note": "raw downloads and data/lake are gitignored",
        }

    state_path = ROOT / "data" / "lake" / "ingest_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        manifest["ingest_state"] = {
            "path": "data/lake/ingest_state.json",
            "sha256": sha256(state_path),
            "last_data_date": state.get("last_data_date"),
            "last_ingested_at": state.get("last_ingested_at"),
            "source_url": state.get("source_url"),
            "warnings": state.get("warnings", []),
            "git_tracked": False,
        }

    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(manifest["tracked"], indent=2)[:2000])


if __name__ == "__main__":
    main()
