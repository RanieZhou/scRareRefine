"""[DEPRECATED] Stage 8: Visualization.

Visualization is now integrated into 07_evaluate.py, which generates
method_comparison.png and rescue_effect.png alongside final_metrics.csv.

This file is kept for reference only. Use 07_evaluate.py instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from utils import load_config, make_run_dir, parse_rare_train_size, read_table


METHOD_ORDER = ["baseline", "prototype", "prototype_gate", "prototype_gate_best", "prototype_gate_marker", "fusion", "fusion_gated"]
METHOD_LABELS = {
    "baseline":              "Baseline\n(scANVI)",
    "prototype":             "Prototype\nRescue",
    "prototype_gate":        "Proto\nGate",
    "prototype_gate_best":   "Gate\n(best)",
    "prototype_gate_marker": "Gate+\nMarker",
    "fusion":                "Fusion\n(global)",
    "fusion_gated":          "Fusion\n(gated)",
}
METHOD_COLORS = {
    "baseline":              "#8da0cb",
    "prototype":             "#66c2a5",
    "prototype_gate":        "#fc8d62",
    "prototype_gate_best":   "#ff6b35",
    "prototype_gate_marker": "#e78ac3",
    "fusion":                "#a6d854",
    "fusion_gated":          "#ffd92f",
}


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in METHOD_ORDER if m in df["method"].values]
    return df.set_index("method").loc[present].reset_index()


def _bar_panel(
    ax: plt.Axes,
    methods: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    fmt: str = ".3f",
    baseline_value: float | None = None,
) -> None:
    colors = [METHOD_COLORS[m] for m in methods]
    bars = ax.bar(range(len(methods)), values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    if baseline_value is not None and "baseline" in methods:
        ax.axhline(baseline_value, color="#555555", linewidth=1.2, linestyle="--", alpha=0.7, zorder=0)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01 if ax.get_ylim()[1] > 0 else 0.005,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_method_comparison(df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    df = _ordered(df)
    methods = df["method"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        f"Method Comparison  |  rare class: {rare_class}  |  seed: {df['seed'].iloc[0]}  |  {df['split_mode'].iloc[0]}",
        fontsize=11, fontweight="bold", y=1.01,
    )

    baseline_row = df[df["method"] == "baseline"].iloc[0] if "baseline" in df["method"].values else None

    metrics = [
        ("rare_f1",         "Rare-class F1",        "F1",        (0, 1.05)),
        ("rare_recall",     "Rare-class Recall",    "Recall",    (0, 1.05)),
        ("rare_precision",  "Rare-class Precision", "Precision", (0, 1.05)),
        ("overall_accuracy","Overall Accuracy",     "Accuracy",  None),
    ]
    for ax, (col, title, ylabel, ylim) in zip(axes.flat, metrics):
        vals = [float(df.loc[df["method"] == m, col].iloc[0]) if col in df.columns and m in df["method"].values else 0.0
                for m in methods]
        baseline_val = float(baseline_row[col]) if baseline_row is not None and col in baseline_row else None
        _bar_panel(ax, methods, vals, title=title, ylabel=ylabel, ylim=ylim, baseline_value=baseline_val)
        if ylim is None:
            lo = min(v for v in vals if np.isfinite(v)) * 0.98 if vals else 0
            hi = max(v for v in vals if np.isfinite(v)) * 1.04 if vals else 1
            ax.set_ylim(lo, hi)
        # re-annotate after ylim is set
        ax.cla()
        _bar_panel(ax, methods, vals, title=title, ylabel=ylabel, ylim=ylim, baseline_value=baseline_val)
        if ylim is None:
            lo = min(v for v in vals if np.isfinite(v)) * 0.98 if vals else 0
            hi = max(v for v in vals if np.isfinite(v)) * 1.05 if vals else 1
            ax.set_ylim(lo, hi)

    legend_patches = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS.get(m, m).replace("\n", " "))
        for m in methods
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(methods),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_rescue_effect(df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    df = _ordered(df)
    rescue_methods = [m for m in df["method"].tolist() if m != "baseline"]
    if not rescue_methods:
        print("  No rescue methods found — skipping rescue_effect plot.")
        return

    rdf = df[df["method"].isin(rescue_methods)].set_index("method")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        f"Rescue Effect  |  rare class: {rare_class}  |  seed: {df['seed'].iloc[0]}",
        fontsize=11, fontweight="bold",
    )

    def _count_bar(ax, col, title, ylabel):
        vals = [float(rdf.loc[m, col]) if col in rdf.columns and m in rdf.index else 0.0
                for m in rescue_methods]
        _bar_panel(ax, rescue_methods, vals, title=title, ylabel=ylabel, fmt=".0f")
        if vals:
            ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
        ax.cla()
        _bar_panel(ax, rescue_methods, vals, title=title, ylabel=ylabel, fmt=".0f")
        if vals:
            ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

    _count_bar(axes[0], "rescued_rare_errors", "Rescued Rare Errors", "# cells")
    _count_bar(axes[1], "false_rescues",        "False Rescues",       "# cells")

    # false rescue rate (%)
    col = "major_to_rare_false_rescue_rate"
    vals = [float(rdf.loc[m, col]) * 100 if col in rdf.columns and m in rdf.index else 0.0
            for m in rescue_methods]
    _bar_panel(axes[2], rescue_methods, vals, title="False Rescue Rate (%)", ylabel="%", fmt=".3f")
    axes[2].cla()
    _bar_panel(axes[2], rescue_methods, vals, title="False Rescue Rate (%)", ylabel="%", fmt=".3f")
    if vals:
        axes[2].set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 0.01)

    legend_patches = [
        mpatches.Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS.get(m, m).replace("\n", " "))
        for m in rescue_methods
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(rescue_methods),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_heatmap(df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    df = _ordered(df)
    cols = [
        ("rare_f1",         "Rare F1"),
        ("rare_recall",     "Rare Recall"),
        ("rare_precision",  "Rare Precision"),
        ("overall_accuracy","Overall Acc"),
        ("major_to_rare_false_rescue_rate", "False Rescue\nRate"),
    ]
    cols = [(c, label) for c, label in cols if c in df.columns]
    if not cols:
        return

    methods = df["method"].tolist()
    data = np.array([
        [float(df.loc[df["method"] == m, c].iloc[0]) if m in df["method"].values else np.nan
         for c, _ in cols]
        for m in methods
    ])

    fig, ax = plt.subplots(figsize=(len(cols) * 1.8, len(methods) * 0.9 + 1.2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([label for _, label in cols], fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods], fontsize=9)

    for i in range(len(methods)):
        for j in range(len(cols)):
            val = data[i, j]
            if np.isfinite(val):
                text_color = "black" if 0.3 < val < 0.85 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7, label="metric value")
    ax.set_title(
        f"All Metrics Heatmap  |  rare: {rare_class}  |  seed: {df['seed'].iloc[0]}",
        fontsize=10, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8: visualize method comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified | lobo_<batch>")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)

    metrics_path = run_dir / "metrics" / "final_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"final_metrics.csv not found at {metrics_path}. Run 07_evaluate.py first.")

    df = read_table(metrics_path)
    out_dir = run_dir / "metrics"

    print(f"Generating plots for {rare_class} / seed={args.seed} / {args.split_mode} ...")
    plot_method_comparison(df, out_dir / "method_comparison.png", rare_class=rare_class)
    plot_rescue_effect(df, out_dir / "rescue_effect.png", rare_class=rare_class)
    plot_metrics_heatmap(df, out_dir / "metrics_heatmap.png", rare_class=rare_class)
    print(f"Done. Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
