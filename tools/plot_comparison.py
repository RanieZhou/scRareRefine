"""绘制对比实验柱状图（scRareRefine vs 4 baselines，3 数据集）。

读取 results/comparison/comparison_summary.csv，绘制 2×3 子图：
  上排：rare-cell F1（mean ± SD，3 seeds）
  下排：rare-cell Recall（mean ± SD，3 seeds）

输出：results/comparison/comparison_bars.png / .pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.linewidth":   0.9,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "savefig.dpi":      300,
})

DETAIL  = Path("results/comparison/comparison_summary.csv")
OUT_PNG = Path("results/comparison/comparison_bars.png")
OUT_PDF = Path("results/comparison/comparison_bars.pdf")

METHODS = [
    ("scANVI",       "#7f7f7f"),
    ("kNN",          "#1f77b4"),
    ("CellTypist",   "#ff7f0e"),
    ("scBalance",    "#9467bd"),
    ("ProtoCloud",   "#e377c2"),   # 粉色，Cell Genomics 2026
    ("HiCat",        "#17becf"),   # 青色，Briefings in Bioinformatics 2025
    ("scCAD",        "#d62728"),   # 红色，Nature Commun 2024
    ("TOSICA",       "#8c564b"),   # 棕色，Nature Commun 2023
    ("scRareRefine", "#2ca02c"),
]

DATASETS = [
    ("immune_dc",        "immune_dc\n(ASDC, high sep >2)"),
    ("pancreas_baron",   "pancreas_baron\n(gamma, borderline sep~1.1–1.6)"),
    ("tabula_lung_endo", "tabula_lung_endo\n(lymphatic, mid sep~1.7)"),
]


def main():
    raw = pd.read_csv(DETAIL)
    raw = raw[raw["status"] == "ok"]

    # 按 dataset × method 聚合
    agg = (raw.groupby(["dataset", "method"])
              .agg(f1_mean=("rare_f1", "mean"),
                   f1_std=("rare_f1", "std"),
                   rec_mean=("rare_recall", "mean"),
                   rec_std=("rare_recall", "std"))
              .reset_index())

    # 只绘制数据中实际存在的方法
    present = set(agg["method"].unique())
    active_methods = [(m, c) for m, c in METHODS if m in present]

    x = np.arange(len(active_methods))
    colors  = [c for _, c in active_methods]
    xlabels = [m for m, _ in active_methods]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True)

    for col, (ds, title) in enumerate(DATASETS):
        sub = agg[agg["dataset"] == ds].set_index("method")

        sub = sub.reindex([m for m, _ in active_methods])  # 缺失方法填 NaN
        f1s      = sub["f1_mean"].fillna(0).tolist()
        f1_sds   = sub["f1_std"].fillna(0).tolist()
        recs     = sub["rec_mean"].fillna(0).tolist()
        rec_sds  = sub["rec_std"].fillna(0).tolist()

        # ── 上排：F1 ─────────────────────────────────────────────────
        ax = axes[0, col]
        bars = ax.bar(x, f1s, yerr=f1_sds, color=colors, edgecolor="k",
                      linewidth=0.7, capsize=4, error_kw={"elinewidth": 1.0})
        ours_idx = next((i for i, (m, _) in enumerate(active_methods) if m == "scRareRefine"), len(active_methods)-1)
        bars[ours_idx].set_linewidth(2.0)
        for xi, f1, sd in zip(x, f1s, f1_sds):
            is_ours = (xi == x[ours_idx])
            ax.text(xi, f1 + sd + 0.022, f"{f1:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if is_ours else "normal",
                    color="#1a6e1a" if is_ours else "black")
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, fontsize=11, pad=6)
        ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)
        if col == 0:
            ax.set_ylabel("Rare-cell F1\n(mean ± SD, 3 seeds)")

        # ── 下排：Recall ──────────────────────────────────────────────
        ax = axes[1, col]
        bars2 = ax.bar(x, recs, yerr=rec_sds, color=colors, edgecolor="k",
                       linewidth=0.7, capsize=4, error_kw={"elinewidth": 1.0})
        bars2[ours_idx].set_linewidth(2.0)
        for xi, r, sd in zip(x, recs, rec_sds):
            is_ours = (xi == x[ours_idx])
            ax.text(xi, r + sd + 0.022, f"{r:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if is_ours else "normal",
                    color="#1a6e1a" if is_ours else "black")
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9, rotation=15, ha="right")
        if col == 0:
            ax.set_ylabel("Rare-cell Recall\n(mean ± SD, 3 seeds)")

    # 图例
    legend_handles = [Patch(facecolor=c, edgecolor="k",
                            label=("scRareRefine (ours)" if m == "scRareRefine" else m))
                      for m, c in active_methods]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(active_methods),
               frameon=True, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(
        "Comparison of rare-cell identification methods across 3 datasets (3-seed average)\n"
        "scRareRefine achieves top F1 on tabula_lung_endo; TOSICA leads on immune_dc and pancreas_baron",
        fontsize=11.5, y=1.10)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
