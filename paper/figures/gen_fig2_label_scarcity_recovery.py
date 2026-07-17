from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import GRAY, GREEN, LIGHT_GRAY, panel_label, save_figure

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "comparison" / "comparison_summary.csv"
SPLITS = ROOT / "results" / "comparison" / "split_rare_nonrare_by_rts_long_train_label_ratio.csv"

ORDER = ["0.01", "0.05", "0.10", "all"]
XLABELS = ["1%", "5%", "10%", "all"]
METHODS = ["scANVI", "scRareRefine"]


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(sem)


def main() -> None:
    summary = pd.read_csv(SUMMARY, dtype={"rare_train_size": str})
    summary = summary[summary["status"] == "ok"].copy()
    split = pd.read_csv(SPLITS, dtype={"rare_train_size": str})
    train = split[split["split"] == "train"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})

    # Panel A: operational rare-label availability.
    box_data = [
        train.loc[train["rare_train_size"] == r, "train_labeled_rare_over_train_total_pct"].dropna().astype(float).values
        for r in ORDER
    ]
    bp = axes[0].boxplot(
        box_data,
        positions=np.arange(len(ORDER)),
        widths=0.48,
        patch_artist=True,
        medianprops={"color": GREEN, "linewidth": 1.2},
        boxprops={"facecolor": "#f5f5f5", "edgecolor": GRAY, "linewidth": 0.8},
        whiskerprops={"color": GRAY, "linewidth": 0.8},
        capprops={"color": GRAY, "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": GRAY, "markeredgecolor": GRAY, "alpha": 0.55},
    )
    rng = np.random.default_rng(0)
    for i, values in enumerate(box_data):
        jitter = rng.normal(0, 0.035, size=len(values))
        axes[0].scatter(np.full(len(values), i) + jitter, values, s=6, color=GRAY, alpha=0.45, linewidths=0)
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(len(ORDER)), XLABELS)
    axes[0].set_ylabel("Labeled rare cells in train (%)")
    axes[0].set_xlabel("Rare-label budget")
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.7)
    panel_label(axes[0], "A")

    # Panels B and C: rare F1 and recall recovery.
    for ax, metric, ylabel, label in [
        (axes[1], "rare_f1", "Rare-cell F1", "B"),
        (axes[2], "rare_recall", "Rare-cell recall", "C"),
    ]:
        for method, color, marker in [("scANVI", GRAY, "o"), ("scRareRefine", GREEN, "s")]:
            means, sems = [], []
            for r in ORDER:
                vals = summary.loc[
                    (summary["rare_train_size"] == r) & (summary["method"] == method),
                    metric,
                ].astype(float)
                mean, sem = mean_sem(vals.to_numpy())
                means.append(mean)
                sems.append(sem)
            ax.errorbar(
                np.arange(len(ORDER)),
                means,
                yerr=sems,
                marker=marker,
                color=color,
                capsize=2,
                markersize=4,
                label=method,
            )
        ax.set_ylim(-0.02, 1.03)
        ax.set_xticks(np.arange(len(ORDER)), XLABELS)
        ax.set_xlabel("Rare-label budget")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.7)
        panel_label(ax, label)

    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, "fig2_label_scarcity_recovery")


if __name__ == "__main__":
    main()
