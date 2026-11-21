"""Plot the E14 static-versus-rolling stress comparison from saved CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", default="reports/deployment_eval/e14_static/e14_stress.csv")
    parser.add_argument("--rolling", default="reports/deployment_eval/e14_rolling20/e14_stress.csv")
    parser.add_argument("--output", default="reports/deployment_eval/e14_static_vs_rolling.png")
    args = parser.parse_args()

    static = pl.read_csv(args.static)
    rolling = pl.read_csv(args.rolling)
    variants = ["current_median", "drop_e14", "e14_indicator"]
    static_values = [
        float(static.filter(pl.col("variant") == variant)["score_proxy"][0])
        for variant in variants
    ]
    rolling_values = [
        float(rolling.filter(pl.col("variant") == variant)["score_proxy"][0])
        for variant in variants
    ]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(variants))
    width = 0.36
    ax.bar(x - width / 2, static_values, width, label="static pre-outage model")
    ax.bar(x + width / 2, rolling_values, width, label="rolling 20-day retrain")
    for index, (static_value, rolling_value) in enumerate(zip(static_values, rolling_values)):
        ax.text(
            index - width / 2,
            static_value,
            f"{static_value:.3f}",
            ha="center",
            va="bottom" if static_value >= 0 else "top",
            fontsize=8,
        )
        ax.text(
            index + width / 2,
            rolling_value,
            f"{rolling_value:.3f}",
            ha="center",
            va="bottom" if rolling_value >= 0 else "top",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("proxy official score")
    ax.set_title("E14 treatment stress test (108 realized days, proxy score)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
