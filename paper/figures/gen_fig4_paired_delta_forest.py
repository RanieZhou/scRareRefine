from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import BLUE, GRAY, GREEN, LIGHT_GRAY, save_figure

ROOT = Path(__file__).resolve().parents[2]
SIG = ROOT / "results" / "comparison" / "significance_test.csv"


def main() -> None:
    df = pd.read_csv(SIG)
    scarce = df[
        df["region"].str.startswith("SCARCE") & (df["analysis"] == "nominal_budget")
    ].copy()
    scarce = scarce.sort_values("dataset_mean_delta", ascending=True).reset_index(
        drop=True
    )
    labels = [
        m + (r"$^\dagger$" if bool(t) else "")
        for m, t in zip(scarce["baseline"], scarce["transductive"])
    ]
    y = np.arange(len(scarce))

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.9))
    x = scarce["dataset_mean_delta"].astype(float).to_numpy()
    lo = scarce["cluster_boot_ci_lo"].astype(float).to_numpy()
    hi = scarce["cluster_boot_ci_hi"].astype(float).to_numpy()
    colors = [GREEN if m == "scANVI" else BLUE for m in scarce["baseline"]]

    ax.axvline(0, color=GRAY, linewidth=0.8)
    for yi, xi, l, h, c in zip(y, x, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, linewidth=1.4)
        ax.scatter([xi], [yi], color=c, s=22, zorder=3)

    for yi, row in scarce.iterrows():
        ax.text(
            0.835,
            yi,
            f"{int(row.run_wins)}/{int(row.run_ties)}/{int(row.run_losses)}",
            va="center",
            ha="right",
            fontsize=7.2,
            color=GRAY,
        )

    ax.set_yticks(y, labels)
    ax.set_xlabel(r"Dataset-equal mean $\Delta$ rare-cell F1")
    ax.set_xlim(-0.02, 0.86)
    ax.set_ylim(-0.6, len(scarce) - 0.4)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.7)
    ax.text(
        0.835,
        len(scarce) - 0.2,
        "W/T/L",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=GRAY,
    )

    fig.tight_layout()
    save_figure(fig, "fig4_paired_delta_forest")


if __name__ == "__main__":
    main()
