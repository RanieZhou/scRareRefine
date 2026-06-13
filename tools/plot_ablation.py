"""绘制消融实验对比图（scRareRefine Conformal 方案，逐组件拆解）。

读取 results/ablation/ablation_summary_agg.csv（3 数据集 × 4 变体，3-seed 均值±σ），
绘制 2×3 子图：
  上排：rare-cell F1（带 SD 误差棒）
  下排：FFR_max（最坏情况假救率）+ 1% 约束红线

叙事：V1（去 rank-1 候选）/ V2（去 conformal 阈值）会让 FFR 冲破 1% 约束；
      只有 V4（完整方法）在所有数据集 FFR≤1% 且 F1 最高。

输出：results/ablation/ablation_bars.png / .pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 论文级样式（与 plot_sweep_rts.py 对齐）
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.linewidth":   0.9,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "savefig.dpi":      300,
})

AGG     = Path("results/ablation/ablation_summary_agg.csv")
OUT_PNG = Path("results/ablation/ablation_bars.png")
OUT_PDF = Path("results/ablation/ablation_bars.pdf")

FFR_LIMIT = 0.01   # 1% 假救率约束

# 变体绘制顺序 + 简短标签 + 颜色（V4 完整方法用绿色强调）
VARIANTS = [
    ("v1_no_rank1",       "V1\nno rank-1\ngate",      "#9e9e9e"),
    ("v2_rank1_nofilter", "V2\nno conformal\n(rescue all)", "#6baed6"),
    ("v3_isotropic",      "V3\nisotropic\nscore",     "#fd8d3c"),
    ("v4_full",           "V4 (full)\nrank-1 +\naniso + conformal", "#2ca02c"),
]
DATASETS = [
    ("immune_dc",        "immune_dc (ASDC)\nhigh sep (>2)"),
    ("pancreas_baron",   "pancreas_baron (gamma)\nborderline (sep~1.1-1.6)"),
    ("tabula_lung_endo", "tabula_lung_endo (lymphatic)\nmid sep (~1.7)"),
]


def main():
    df = pd.read_csv(AGG)
    x = np.arange(len(VARIANTS))
    colors = [c for _, _, c in VARIANTS]
    labels = [lab for _, lab, _ in VARIANTS]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True)

    for col, (ds, title) in enumerate(DATASETS):
        sub = df[df["dataset"] == ds].set_index("variant")

        f1s  = [sub.loc[v, "f1_mean"]  for v, _, _ in VARIANTS]
        sds  = [sub.loc[v, "f1_std"]   for v, _, _ in VARIANTS]
        ffrs = [sub.loc[v, "ffr_max"]  for v, _, _ in VARIANTS]

        # ── 上排：F1 ──────────────────────────────────────────────
        ax = axes[0, col]
        bars = ax.bar(x, f1s, yerr=sds, color=colors, edgecolor="k",
                      linewidth=0.7, capsize=4, error_kw={"elinewidth": 1.0})
        # V4 加粗描边突出
        bars[-1].set_linewidth(1.8)
        for xi, f1, sd in zip(x, f1s, sds):
            ax.text(xi, f1 + sd + 0.025, f"{f1:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold" if xi == x[-1] else "normal")
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, fontsize=11.5)
        ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)
        if col == 0:
            ax.set_ylabel("Rare-cell F1\n(mean ± SD, 3 seeds)")

        # ── 下排：FFR_max + 1% 约束 ───────────────────────────────
        ax = axes[1, col]
        bar_colors = [c if f <= FFR_LIMIT else c for c, f in zip(colors, ffrs)]
        bars = ax.bar(x, ffrs, color=bar_colors, edgecolor="k", linewidth=0.7)
        bars[-1].set_linewidth(1.8)
        # 约束线
        ax.axhline(FFR_LIMIT, color="#d62728", ls="--", lw=1.6, zorder=5)
        if col == len(DATASETS) - 1:
            ax.text(x[-1] + 0.45, FFR_LIMIT, "  FFR = 1%\n  target", color="#d62728",
                    va="center", ha="left", fontsize=9, fontweight="bold")
        # 柱顶标注：违反约束标红 ✗，通过的标 ✓
        ymax = max(max(ffrs), FFR_LIMIT) * 1.18
        for xi, f in zip(x, ffrs):
            viol = f > FFR_LIMIT
            ax.text(xi, f + ymax * 0.02,
                    f"{f*100:.2f}%\n{'✗' if viol else '✓'}",
                    ha="center", va="bottom", fontsize=8.5,
                    color="#d62728" if viol else "#2ca02c",
                    fontweight="bold")
        ax.set_ylim(0, ymax * 1.25)
        ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        if col == 0:
            ax.set_ylabel("FFR (worst of 3 seeds)\nfalse-rescue rate")

    # 统一图例
    legend_handles = [Patch(facecolor=c, edgecolor="k", label=lab.replace("\n", " "))
                      for _, lab, c in VARIANTS]
    legend_handles.append(plt.Line2D([0], [0], color="#d62728", ls="--", lw=1.6,
                                     label="FFR = 1% constraint"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               frameon=True, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(
        "Ablation of scRareRefine components: rank-1 candidate gate + anisotropic score + conformal threshold\n"
        "Only the full method (V4) keeps FFR ≤ 1% across all datasets while maximizing F1",
        fontsize=12.5, y=1.10)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
