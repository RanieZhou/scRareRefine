"""Plot 3-seed label-scarcity recovery curves from multiseed aggregates.

Inputs:
  results/multiseed/core_agg.csv

Outputs:
  results/multiseed/fig2_recovery_curves.png
  results/multiseed/figS_recall_recovery_panel.png
  paper/figures/fig2_recovery_curves.png
  paper/figures/figS_recall_recovery_panel.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
IN_CSV = ROOT / "results" / "multiseed" / "core_agg.csv"
OUT_RESULTS = ROOT / "results" / "multiseed"
OUT_PAPER = ROOT / "paper" / "figures"

RTS_ORDER = ["0.01", "0.05", "0.10", "all"]
X = np.arange(len(RTS_ORDER))
DATASETS = [
    ("immune_dc", "immune_dc\nASDC"),
    ("pancreas_baron", "pancreas_baron\ngamma"),
    ("pancreas_integrated", "pancreas_integrated\nendothelial"),
    ("tabula_lung_endo", "tabula_lung_endo\nlymphatic endothelial"),
    ("tabula_sapiens_stomach", "tabula_sapiens_stomach\nmast cell"),
    ("tabula_small_intestine", "tabula_small_intestine\ntuft cell"),
]
METHODS = [
    ("scANVI", "#747474", "o", 1.7),
    ("scRareRefine", "#197245", "s", 2.6),
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
    }
)


def _values(sub: pd.DataFrame, method: str, mean_col: str, std_col: str | None) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for rts in RTS_ORDER:
        row = sub[(sub["method"] == method) & (sub["rare_train_size"] == rts)]
        if row.empty:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(float(row[mean_col].iloc[0]))
            stds.append(float(row[std_col].iloc[0]) if std_col and std_col in row else 0.0)
    return means, stds


def _plot(metric: str, mean_col: str, std_col: str | None, out_name: str, ylabel: str) -> None:
    df = pd.read_csv(IN_CSV, dtype={"rare_train_size": str})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharey=True)
    axes = axes.ravel()

    for ax, (dataset, title) in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset]
        for method, color, marker, lw in METHODS:
            means, stds = _values(sub, method, mean_col, std_col)
            ax.errorbar(
                X,
                means,
                yerr=stds if std_col else None,
                color=color,
                marker=marker,
                lw=lw,
                ms=5.5,
                capsize=2.5 if std_col else 0,
                elinewidth=0.8,
                label=method,
                zorder=3 if method == "scRareRefine" else 2,
            )
        ax.set_title(title)
        ax.set_xticks(X)
        ax.set_xticklabels(RTS_ORDER)
        ax.set_xlim(-0.25, len(RTS_ORDER) - 0.75)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xlabel("rare_train_size")

    axes[0].set_ylabel(ylabel)
    axes[3].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(metric, y=0.95, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for out_dir in [OUT_RESULTS, OUT_PAPER]:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / out_name
        fig.savefig(path, bbox_inches="tight")
        print(f"[saved] {path.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    _plot(
        "Rare-cell F1 recovery under label scarcity (3 seeds, mean +/- SD)",
        "f1_mean",
        "f1_std",
        "fig2_recovery_curves.png",
        "Rare-cell F1",
    )
    _plot(
        "Rare-cell recall recovery under label scarcity (3 seeds, mean)",
        "rec_mean",
        None,
        "figS_recall_recovery_panel.png",
        "Rare-cell recall",
    )


if __name__ == "__main__":
    main()
