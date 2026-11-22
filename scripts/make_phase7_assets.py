"""Build the Phase 7 presentation assets from the saved research reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BG = "#0b1220"
FG = "#e2e8f0"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
PURPLE = "#a78bfa"
ROSE = "#fb7185"


def load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path)


def main() -> None:
    cv = load_csv(ROOT / "reports" / "experiments" / "summary.csv")
    deployment = load_csv(ROOT / "reports" / "deployment_eval" / "summary.csv")
    official = load_csv(ROOT / "reports" / "live_window_20251216_20260706" / "metrics.csv")
    post = load_csv(ROOT / "reports" / "phase6_signal_robustness" / "post_metrics.csv")
    multiplier = load_csv(ROOT / "reports" / "phase6_signal_robustness" / "multiplier_grid.csv")
    with (ROOT / "reports" / "phase6_signal_robustness" / "summary.json").open() as handle:
        phase6 = json.load(handle)
    tests = load_csv(ROOT / "reports" / "phase6_signal_robustness" / "statistical_tests_official.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), facecolor=BG)
    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.title.set_color(FG)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # A. CV: six-fold mean score, the honest cross-validation view.
    ax = axes[0, 0]
    order = [
        ("buy_hold", "Buy & Hold", AMBER),
        ("constant_1.25", "Const 1.25", PURPLE),
        ("vol_only", "Vol-only", GREEN),
        ("online", "Online", BLUE),
        ("daily_retrain", "Daily retrain", ROSE),
        ("offline", "Offline", "#f472b6"),
    ]
    cv_map = {row["strategy"]: row for row in cv.iter_rows(named=True)} if "strategy" in cv.columns else {}
    dep_map = {row["strategy"]: row for row in deployment.iter_rows(named=True)} if "strategy" in deployment.columns else {}
    experiment_rows = list(cv.iter_rows(named=True))

    def cv_score(key: str) -> float | None:
        if key in dep_map:
            return float(dep_map[key]["score_mean"])
        if key == "buy_hold":
            matches = [row for row in experiment_rows if row["experiment"] == "buy_hold"]
        elif key == "constant_1.25":
            matches = [
                row for row in experiment_rows
                if row["experiment"] == "constant" and row["variant"] == "constant_1.25"
            ]
        elif key == "vol_only":
            matches = [
                row for row in experiment_rows
                if row["experiment"] == "vol_only" and row["variant"] == "scale_1.0" and row["mode"] == "offline"
            ]
        elif key == "online":
            matches = [
                row for row in experiment_rows
                if row["experiment"] == "model" and row["variant"] == "notebook_gbdt" and row["mode"] == "online"
            ]
        elif key == "offline":
            matches = [
                row for row in experiment_rows
                if row["experiment"] == "model" and row["variant"] == "notebook_gbdt" and row["mode"] == "offline"
            ]
        else:
            return None
        return float(matches[0]["score_mean"]) if matches else None

    labels, values, colors = [], [], []
    for key, label, color in order:
        value = cv_score(key)
        if value is None:
            continue
        labels.append(label)
        values.append(value)
        colors.append(color)
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("A. Six-fold walk-forward: fold-mean official score", loc="left", fontsize=10, pad=8)
    ax.set_ylabel("official score")
    ax.tick_params(axis="x", rotation=25)
    for index, value in enumerate(values):
        ax.text(index, value + 0.03, f"{value:.2f}", ha="center", color=FG, fontsize=8)

    # B. Official vs post window: score alone is not a stable statistic.
    ax = axes[0, 1]
    protocols = ["daily", "fixed_rolling", "offline", "vol_only", "buy_hold", "constant_1.25"]
    display = ["Daily", "Fixed+roll", "Offline", "Vol-only", "B&H", "Const 1.25"]
    off_map = {row["mode"]: row for row in official.iter_rows(named=True)}
    post_map = {row["mode"]: row for row in post.iter_rows(named=True)}
    x = np.arange(len(protocols))
    width = 0.36
    off_values = [float(off_map[p]["proxy_official_score"]) for p in protocols]
    post_values = [float(post_map[p]["official_score_proxy"]) for p in protocols]
    ax.bar(x - width / 2, off_values, width, label="official 137d", color=BLUE, alpha=0.85)
    ax.bar(x + width / 2, post_values, width, label="locked 95d", color=AMBER, alpha=0.85)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=25, ha="right")
    ax.set_title("B. Official window vs independent post-window", loc="left", fontsize=10, pad=8)
    ax.set_ylabel("official score (proxy)")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    # C. Multiplier curve: broad plateau, original switch marked.
    ax = axes[1, 0]
    numeric = [row for row in multiplier.iter_rows(named=True) if row["multiplier"] not in (None, "original_switch")]
    x = np.asarray([float(row["multiplier"]) for row in numeric])
    ax.plot(x, [float(row["score"]) for row in numeric], marker="o", color=GREEN, linewidth=1.8)
    original = next(row for row in multiplier.iter_rows(named=True) if row["multiplier"] == "original_switch")
    ax.axhline(float(original["score"]), color=ROSE, linestyle="--", linewidth=1.2,
               label=f"original 600/400: {float(original['score']):.3f}")
    ax.scatter([400, 600], [float(row["score"]) for row in numeric if float(row["multiplier"]) in (400, 600)],
               color=AMBER, zorder=5, label="fixed 400 / 600")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 400, 600, 1200])
    ax.set_xticklabels(["1", "10", "100", "400", "600", "1200"])
    ax.set_title("C. Multiplier robustness: no single stable optimum", loc="left", fontsize=10, pad=8)
    ax.set_xlabel("constant multiplier (log scale)")
    ax.set_ylabel("official score")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    # D. Concentration: what drives the result.
    ax = axes[1, 1]
    concentration = phase6["concentration"]
    months = ["2026-01", "2026-04"]
    metrics = ["top1_share", "top3_share", "top5_share"]
    labels = ["top 1", "top 3", "top 5"]
    x = np.arange(len(labels))
    width = 0.34
    for index, month in enumerate(months):
        values = [100.0 * float(concentration[month][metric]) for metric in metrics]
        ax.bar(x + (index - 0.5) * width, values, width, label=month,
               color=BLUE if index == 0 else AMBER, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of positive contribution")
    ax.set_title("D. January / April: positive contribution concentration", loc="left", fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    # Footer with the inferential caveat, not just the raw score.
    daily_p = float(tests.filter(pl.col("hypothesis") == "daily_mean_excess_positive")["p_positive"][0])
    holm_p = float(tests.filter(pl.col("hypothesis") == "daily_mean_excess_positive")["p_holm"][0])
    fig.text(
        0.01,
        0.012,
        "Read this as evidence, not a headline: official score = 2.192 in 137 days, but daily excess HAC p="
        f"{daily_p:.3f} (Holm p={holm_p:.3f}); the locked 95-day window is descriptive and regime-imbalanced.",
        color=MUTED,
        fontsize=8,
    )
    fig.suptitle("Hull Tactical GBDT: honest evidence map", color=FG, fontsize=15, x=0.01, ha="left", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.12, hspace=0.42, wspace=0.25)
    fig.savefig(OUT / "core_results.png", facecolor=BG)
    fig.savefig(OUT / "core_results.svg", facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'core_results.png'}")
    print(f"wrote {OUT / 'core_results.svg'}")


if __name__ == "__main__":
    main()
