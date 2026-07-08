"""对比柱状图（网格版）：5 数据集(行) × 4 比例(列) = 20 子图。

每个子图为某 (数据集, rare_train_size) 下各方法的 rare-cell F1 柱状图（均值 ± SD），
scRareRefine 绿色加粗高亮。数据源：results/comparison/comparison_summary_agg.csv
（3 seeds: 42/43/44 全格齐备，误差棒=跨 seed SD）。

输出：results/comparison/comparison_bars_grid.png / .pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.linewidth":  0.9,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "savefig.dpi":     300,
})

AGG     = Path("results/comparison/comparison_summary_agg.csv")
OUT_PNG = Path("results/comparison/comparison_bars_grid.png")

RTS_ORDER = ["0.01", "0.05", "0.10", "all"]
# Low-saturation science palette (colorblind-aware, muted/Morandi style)
METHODS = [
    ("scANVI",       "#888888"),  # neutral gray (backbone reference)
    ("kNN",          "#5B7FA6"),  # muted slate blue
    ("CellTypist",   "#C97A50"),  # muted terracotta/amber
    ("scBalance",    "#7D6A9E"),  # muted indigo
    ("ProtoCloud",   "#C47BAB"),  # muted dusty mauve
    ("HiCat",        "#6BADB5"),  # muted steel teal (transductive)
    ("scCAD",        "#B55D5A"),  # muted brick rose
    ("TOSICA",       "#906B5A"),  # warm sand brown
    ("scRareRefine", "#1A7A4A"),  # deep emerald (our method)
]
DATASETS = [
    ("immune_dc",              "immune_dc\n(ASDC)"),
    ("pancreas_baron",         "pancreas_baron\n(gamma)"),
    ("tabula_lung_endo",       "tabula_lung_endo\n(lymph EC)"),
    ("tabula_small_intestine", "tabula_small_intestine\n(tuft cell)"),
    ("tabula_sapiens_stomach", "tabula_sapiens_stomach\n(mast cell)"),
    ("pancreas_integrated",    "pancreas_integrated\n(endothelial)"),
    ("mouse_lung_tms_10x",     "mouse_lung_tms\n(vein EC)"),
    ("mouse_pancreas_tms_10x", "mouse_pancreas_tms\n(D cell)"),
]


def main():
    df = pd.read_csv(AGG, dtype={"rare_train_size": str})
    x = np.arange(len(METHODS))
    colors  = [c for _, c in METHODS]
    xlabels = [m + "*" if m == "HiCat" else m for m, _ in METHODS]
    ours_idx = [m for m, _ in METHODS].index("scRareRefine")

    n_row, n_col = len(DATASETS), len(RTS_ORDER)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.4 * n_col, 3.0 * n_row),
                             sharey=True)

    for r, (ds, ds_title) in enumerate(DATASETS):
        for c, rts in enumerate(RTS_ORDER):
            ax = axes[r, c]
            sub = df[(df["dataset"] == ds) & (df["rare_train_size"] == rts)].set_index("method")
            sub = sub.reindex([m for m, _ in METHODS])
            f1s = sub["f1_mean"].tolist()
            f1_sds = [v if pd.notna(v) else 0.0 for v in sub["f1_std"].tolist()]

            f1s_plot = [v if pd.notna(v) else 0.0 for v in f1s]
            bars = ax.bar(x, f1s_plot, yerr=f1_sds, color=colors, edgecolor="k", linewidth=0.6,
                          capsize=2.5, error_kw={"elinewidth": 0.8})
            for i, v in enumerate(f1s):
                if pd.isna(v):
                    bars[i].set_hatch("//"); bars[i].set_facecolor("#eeeeee")
                    ax.text(i, 0.03, "NA", ha="center", va="bottom",
                            fontsize=6.5, color="#666666")
            bars[ours_idx].set_linewidth(2.0); bars[ours_idx].set_edgecolor("#1A7A4A")

            # 每个方法柱子都标数值（横排，scRareRefine 加粗绿色；相邻交错上下高度防重叠）
            for xi, v, sd in zip(x, f1s, f1_sds):
                if pd.isna(v):
                    continue
                is_ours = (xi == ours_idx)
                offset = 0.02 if xi % 2 == 0 else 0.085   # 奇偶柱交错抬高
                ax.text(xi, v + sd + offset, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5,
                        fontweight="bold" if is_ours else "normal",
                        color="#1A7A4A" if is_ours else "black")
            ax.set_ylim(0, 1.22)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
            ax.tick_params(length=3, width=0.7)
            ax.set_xticks(x)
            if r == n_row - 1:
                ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(f"rare_train_size = {rts}", fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(f"{ds_title}\nrare-cell F1", fontsize=9.5)

    legend_handles = [Patch(facecolor=c, edgecolor="k",
                            label=("scRareRefine (ours)" if m == "scRareRefine" else
                                   ("HiCat* (transductive)" if m == "HiCat" else m)))
                      for m, c in METHODS]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(METHODS),
               frameon=True, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Rare-cell F1 by dataset (rows) x rare_train_size (cols) - scRareRefine vs. 8 baselines (mean +/- SD over seeds 42/43/44)",
                 fontsize=13, y=1.035)
    fig.text(0.5, -0.008,
             f"* HiCat is transductive (test-aware dimensionality reduction), shown as an upper-bound reference. "
             f"Error bars = SD across seeds 42/43/44; hatched bars mark missing runs. "
             f"Grid scope: 9 methods x {len(DATASETS)} datasets x {len(RTS_ORDER)} rare_train_size values.",
             ha="center", va="top", fontsize=8, color="#444")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
