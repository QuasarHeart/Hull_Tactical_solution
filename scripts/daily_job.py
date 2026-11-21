"""Daily deployment job: ingest the latest CSV, then predict it.

This is the entry point a systemd timer or cron should call. It is intentionally
network-first: if the source is unavailable it exits non-zero and leaves the
previous panel/prediction in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import Config, load_env  # noqa: E402
from common.ingest import ingest_latest  # noqa: E402
from scripts.predict_latest import run as run_prediction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gdbt/notebook.yaml")
    parser.add_argument("--force", action="store_true", help="force re-download")
    parser.add_argument("--wandb", action="store_true", help="log the prediction to W&B")
    args = parser.parse_args()

    load_env()
    config = Config.load(args.config)
    config.paths.ensure()

    try:
        ingest = ingest_latest(config, force=args.force)
        print(json.dumps(ingest.as_dict(), indent=2, ensure_ascii=False))

        report = run_prediction(config, log_wandb=args.wandb)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - cron should see a clear error
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
