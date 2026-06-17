"""绘制 rare_train_size 稳健性扫描对比图。

读取 results/sweep_rts/sweep_rts_agg.csv（3 数据集 × 4 比例 × 5 方法，3-seed 均值±σ），
绘制 1×3 子图：横轴 rare_train_size，纵轴 rare-cell F1，5 条方法曲线带 σ 误差带。
scRareRefine 用红色粗线突出。

输出：results/sweep_rts/sweep_rts_curves.png / .pdf
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 论文级样式
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.linewidth":   0.9,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10.5,
    "savefig.dpi":      300,
})

AGG     = Path("results/sweep_rts/sweep_rts_agg.csv")
OUT_PNG = Path("results/sweep_rts/sweep_rts_curves.png")
OUT_PDF = Path("results/sweep_rts/sweep_rts_curves.pdf")

RTS_ORDER = ["0.01", "0.05", "0.10", "all"]
DATASETS  = [
    ("immune_dc",        "immune_dc (ASDC)\nhigh separability (sep>2)"),
    ("pancreas_baron",   "pancreas_baron (gamma)\nborderline (sep~1.1-1.6)"),
    ("tabula_lung_endo", "tabula_lung_endo (lymphatic)\nmid separability (sep~1.7)"),
]
# 方法绘制顺序与样式（scRareRefine 最后画，置于最上层并突出）
# (name, color, marker, linewidth, markersize, alpha)
METHOD_STYLE = [
    ("scANVI",       "#7f7f7f", "o", 1.7, 5.5, 0.9),
    ("kNN",          "#1f77b4", "s", 1.7, 5.5, 0.9),
    ("CellTypist",   "#2ca02c", "^", 1.7, 5.5, 0.9),
    ("scBalance",    "#ff7f0e", "D", 1.7, 5.5, 0.9),
    ("scRareRefine", "#d62728", "o", 3.0, 8.5, 1.0),
]


def main():
    df = pd.read_csv(AGG)
    x = list(range(len(RTS_ORDER)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)

    for ax, (ds, title) in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        # x 轴刻度标签：rts + 实际标注数
        xticklabels = []
        for rts in RTS_ORDER:
            r = sub[sub["rare_train_size"] == rts]
            lab = int(r["lab_rare"].iloc[0]) if len(r) else "-"
            xticklabels.append(f"{rts}\n(n={lab})")

        for mname, color, marker, lw, ms, alpha in METHOD_STYLE:
            ys, errs = [], []
            for rts in RTS_ORDER:
                r = sub[(sub["method"] == mname) & (sub["rare_train_size"] == rts)]
                ys.append(r["f1_mean"].iloc[0] if len(r) else float("nan"))
                errs.append(r["f1_std"].iloc[0] if len(r) else 0.0)
            z = 6 if mname == "scRareRefine" else 2
            mew = 0.6 if mname == "scRareRefine" else 0.0
            ax.plot(x, ys, marker=marker, color=color, lw=lw, ms=ms,
                    label=mname, alpha=alpha, zorder=z,
                    markeredgecolor="k", markeredgewidth=mew)
            lo = [y - e for y, e in zip(ys, errs)]
            hi = [y + e for y, e in zip(ys, errs)]
            ax.fill_between(x, lo, hi, color=color, alpha=0.13, zorder=z - 1,
                            linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("rare_train_size (labeled rare cells)")
        ax.set_title(title)
        ax.set_xlim(-0.25, len(RTS_ORDER) - 0.75)
        ax.set_ylim(-0.02, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)

    axes[0].set_ylabel("Rare-cell F1  (mean ± SD, 3 seeds)")
    # 统一图例放在顶部
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("rare_train_size robustness: scRareRefine vs. baselines (scANVI / kNN / CellTypist / scBalance)",
                 fontsize=13, y=1.11)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
