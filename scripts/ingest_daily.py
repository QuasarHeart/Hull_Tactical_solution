"""Download and validate the latest daily Hull Tactical dataset.

Usage:
    python scripts/ingest_daily.py --check       # resolve the latest URL only
    python scripts/ingest_daily.py --dry-run     # download + validate, no panel write
    python scripts/ingest_daily.py               # ingest into data/lake/panel_live.parquet
    python scripts/ingest_daily.py --force       # re-download and re-ingest the same date
"""

from __future__ import annotations

import argparse
import json
import sys

from common.config import Config, load_env
from common.ingest import (
    ingest_latest,
    make_session,
    resolve_latest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/notebook.yaml")
    parser.add_argument("--check", action="store_true", help="resolve the latest URL only")
    parser.add_argument("--dry-run", action="store_true", help="download/validate, no panel write")
    parser.add_argument("--force", action="store_true", help="re-download the latest date")
    args = parser.parse_args()

    load_env()
    config = Config.load(args.config)
    config.paths.ensure()

    try:
        if args.check:
            session = make_session(config.data_source)
            source = resolve_latest(config.data_source, session)
            print(f"latest source date: {source.data_date.isoformat()}")
            print(f"latest source url : {source.url}")
            if config.paths.panel_path.exists():
                print(f"panel path        : {config.paths.panel_path}")
                print(f"state path        : {config.paths.state_path}")
            return 0

        result = ingest_latest(config, force=args.force, dry_run=args.dry_run)
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - entry point should print a clear error
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
