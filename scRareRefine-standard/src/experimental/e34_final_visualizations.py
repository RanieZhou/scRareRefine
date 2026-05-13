"""E34: Final visualizations for Round 5 experiments."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import read_table

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs" / "_experimental" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "scANVI":        "#8da0cb",
    "Euclidean":     "#66c2a5",
    "Mahal-pooled":  "#fc8d62",
    "Logit Adj":     "#e78ac3",
    "Combined":      "#a6d854",
    "S-Adaptive":    "#ffd92f",
}


def fig_e33_regime_bars():
    """Grouped bar chart: all methods × high-sep vs low-sep regimes."""
    path = ROOT / "outputs" / "_experimental" / "e33_logit_adj_plus_mahal" / "aggregated_results.csv"
    if not path.exists():
        return
    agg = read_table(path)

    high_sep = ["cDC1", "ASDC", "gamma", "innate lymphoid cell"]
    low_sep  = ["epsilon", "non-classical monocyte", "endothelial cell"]
    methods  = ["scANVI", "Euclidean", "Mahal-pooled", "Logit Adj", "Combined", "S-Adaptive"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (regime, classes) in zip(axes, [("High-sep", high_sep), ("Low-sep", low_sep)]):
        sub = agg[agg["rare_class"].isin(classes)]
        means, stds = [], []
        for m in methods:
            vals = sub[sub["method"] == m]["mean_f1"].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            stds.append(float(vals.std()) if len(vals) > 0 else 0.0)
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[COLORS.get(m, "#aaa") for m in methods],
                      alpha=0.85, width=0.6)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean rare-class F1", fontsize=10)
        ax.set_title(f"{regime} cases", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("E33: Logit Adj + Mahal Combined — regime analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig_e33_regime_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_e32_logit_adj_vs_mahal():
    """Scatter: Logit Adj F1 vs Mahal-pooled F1 per run."""
    path = ROOT / "outputs" / "_experimental" / "e32_logit_adj_full_sweep" / "per_run_results.csv"
    if not path.exists():
        return
    df = read_table(path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["mahal_f1"], df["logit_adj_f1"], alpha=0.6, s=40, c="#e78ac3")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Mahal-pooled rare_f1", fontsize=11)
    ax.set_ylabel("Logit Adjustment rare_f1", fontsize=11)
    ax.set_title("E32: Logit Adj vs Mahal-pooled (each point = one run)", fontsize=11)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    n_la_wins = (df["logit_adj_f1"] > df["mahal_f1"]).sum()
    ax.text(0.05, 0.95, f"LA wins: {n_la_wins}/{len(df)}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig_e32_logit_adj_vs_mahal_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_final_summary():
    """Final summary: all methods across all paradigms."""
    # Collect from E33 aggregated
    path = ROOT / "outputs" / "_experimental" / "e33_logit_adj_plus_mahal" / "aggregated_results.csv"
    if not path.exists():
        return
    agg = read_table(path)

    methods = ["scANVI", "Euclidean", "Mahal-pooled", "Logit Adj", "Combined", "S-Adaptive"]
    means, stds = [], []
    for m in methods:
        vals = agg[agg["method"] == m]["mean_f1"].dropna()
        means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        stds.append(float(vals.std()) if len(vals) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS.get(m, "#aaa") for m in methods],
                  alpha=0.85, width=0.6)
    for bar, v, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Mean rare-class F1 (all datasets × rts × seeds)", fontsize=10)
    ax.set_title("Final summary: all methods — mean ± std across 21 configurations",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add paradigm labels
    paradigms = ["baseline", "geometric", "geometric", "probabilistic", "ensemble", "adaptive"]
    for i, (bar, p) in enumerate(zip(bars, paradigms)):
        ax.text(bar.get_x() + bar.get_width()/2, 0.02, p,
                ha="center", va="bottom", fontsize=7, color="gray", rotation=90)

    fig.tight_layout()
    out = FIG_DIR / "fig_final_summary_all_methods.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    print("Generating Round 5 figures ...")
    fig_e33_regime_bars()
    fig_e32_logit_adj_vs_mahal()
    fig_final_summary()
    print(f"All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
