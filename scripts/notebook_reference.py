"""Execute the submitted notebook's code as a reference implementation.

The notebook's `kaggle_evaluation` import is stubbed because the gateway is not
part of the research pipeline. The notebook cell is otherwise executed verbatim
(with DATA_PATH pointed at the local data directory), and its `predict` function
is called once per row of data/test.csv.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import polars as pl

from common.config import PROJECT_ROOT


def _notebook_source(notebook_path: Path) -> str:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    if not cells:
        raise ValueError(f"No cells found in {notebook_path}")
    return "".join(cells[0].get("source", []))


def _stub_kaggle_evaluation() -> None:
    package = types.ModuleType("kaggle_evaluation")
    server = types.ModuleType("kaggle_evaluation.default_inference_server")

    class DefaultInferenceServer:
        def __init__(self, predict: Any) -> None:
            self.predict = predict

        def serve(self) -> None:
            return None

        def run_local_gateway(self, paths: Any) -> None:
            return None

    server.DefaultInferenceServer = DefaultInferenceServer  # type: ignore[attr-defined]
    package.default_inference_server = server  # type: ignore[attr-defined]
    sys.modules["kaggle_evaluation"] = package
    sys.modules["kaggle_evaluation.default_inference_server"] = server


def run_reference(
    notebook_path: str | Path = PROJECT_ROOT / "gdbt/notebooks/final_submission.ipynb",
    data_dir: str | Path = PROJECT_ROOT / "data",
) -> pl.DataFrame:
    notebook_path = Path(notebook_path)
    data_dir = Path(data_dir).resolve()
    source = _notebook_source(notebook_path)
    source = source.replace(
        "DATA_PATH = Path('/kaggle/input/hull-tactical-market-prediction/')",
        f"DATA_PATH = Path(r'{data_dir}')",
    )
    if "/kaggle/input" in source:
        raise RuntimeError("Failed to redirect DATA_PATH in the notebook source")

    _stub_kaggle_evaluation()
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(notebook_path)}
    exec(compile(source, str(notebook_path), "exec"), namespace)

    predict = namespace["predict"]
    test = pl.read_csv(data_dir / "test.csv")
    allocations = [float(predict(test.slice(i, 1))) for i in range(test.height)]
    return pl.DataFrame(
        {"date_id": test["date_id"].to_list(), "allocation": allocations}
    )


if __name__ == "__main__":
    result = run_reference()
    print(result)
    out = PROJECT_ROOT / "reports" / "notebook_reference.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(out)
    print(f"saved {out}")
