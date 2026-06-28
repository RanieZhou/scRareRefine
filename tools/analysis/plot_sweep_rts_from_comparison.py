"""绘制 rare_train_size 稳健性扫描曲线（数据源：results/comparison/comparison_summary_agg.csv）。

comparison_summary_agg.csv 本身即一份完整的 rare_train_size 扫描：
5 数据集 × 4 比例(0.01/0.05/0.10/all) × 9 方法（seed=42 单种子）。
本脚本直接消费该 live 数据，横轴 rare_train_size、纵轴 rare-cell F1，每数据集一子图，
scRareRefine 红色粗线突出。

输出：results/sweep_rts/sweep_rts_curves.png / .pdf
注：当前为单 seed(=42)，无 σ 误差带；补 seed=43/44 后可加误差带。
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.linewidth":  0.9,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "savefig.dpi":     300,
})

AGG     = Path("results/comparison/comparison_summary_agg.csv")
OUT_PNG = Path("results/sweep_rts/sweep_rts_curves.png")

RTS_ORDER = ["0.01", "0.05", "0.10", "all"]
DATASETS = [
    ("immune_dc",              "immune_dc (ASDC)\nhigh sep (~2.1)"),
    ("pancreas_baron",         "pancreas_baron (gamma)\nborderline (~1.2-1.5)"),
    ("tabula_lung_endo",       "tabula_lung_endo (lymphatic)\nmid-high sep (~1.7-2.3)"),
    ("tabula_small_intestine", "tabula_small_intestine (tuft)\nhigh sep (~2.3-3.2)"),
    ("tabula_sapiens_stomach", "tabula_sapiens_stomach (mast)\nmid sep (~1.8)"),
    ("pancreas_integrated",    "pancreas_integrated (endothelial)\nintegrated 5-platform"),
]
# (name, color, marker, linewidth, markersize) — scRareRefine 最后画、置顶突出
# Colors match comparison plots (low-saturation muted palette)
METHOD_STYLE = [
    ("scANVI",       "#888888", "o",  1.5, 5),    # neutral gray
    ("kNN",          "#5B7FA6", "s",  1.5, 5),    # muted slate blue
    ("CellTypist",   "#C97A50", "^",  1.5, 5),    # muted terracotta
    ("scBalance",    "#7D6A9E", "D",  1.5, 5),    # muted indigo
    ("ProtoCloud",   "#C47BAB", "v",  1.5, 5),    # muted dusty mauve
    ("HiCat",        "#6BADB5", "P",  1.5, 5),    # muted steel teal
    ("scCAD",        "#B55D5A", "X",  1.5, 5),    # muted brick rose
    ("TOSICA",       "#906B5A", "*",  1.5, 6),    # warm sand brown
    ("scRareRefine", "#1A7A4A", "o",  3.0, 8.5),  # deep emerald (ours)
]


def main():
    df = pd.read_csv(AGG, dtype={"rare_train_size": str})
    x = list(range(len(RTS_ORDER)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharey=True)
    axes = axes.ravel()

    for ax, (ds, title) in zip(axes, DATASETS):
        sub = df[df["dataset"] == ds]
        for mname, color, marker, lw, ms in METHOD_STYLE:
            ys = []
            for rts in RTS_ORDER:
                r = sub[(sub["method"] == mname) & (sub["rare_train_size"] == rts)]
                ys.append(float(r["f1_mean"].iloc[0]) if len(r) else float("nan"))
            if all(pd.isna(y) for y in ys):
                continue  # 该数据集无此方法（如 immune_dc 无 TOSICA）
            z = 6 if mname == "scRareRefine" else 2
            mew = 0.6 if mname == "scRareRefine" else 0.0
            ax.plot(x, ys, marker=marker, color=color, lw=lw, ms=ms, label=mname,
                    zorder=z, markeredgecolor="k", markeredgewidth=mew)

        ax.set_xticks(x)
        ax.set_xticklabels(RTS_ORDER)
        ax.set_xlabel("rare_train_size (labeled rare fraction)")
        ax.set_title(title)
        ax.set_xlim(-0.25, len(RTS_ORDER) - 0.75)
        ax.set_ylim(-0.02, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)

    axes[0].set_ylabel("Rare-cell F1 (seed=42)")
    axes[3].set_ylabel("Rare-cell F1 (seed=42)")

    # 隐藏多余空子图（数据集数 < 网格数时）
    for ax in axes[len(DATASETS):]:
        ax.axis("off")

    # 顶部 figure 图例（用出现方法最全的子图收集 handles）
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[:len(DATASETS)]:
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labels):
            handles, labels = h, l
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=True, bbox_to_anchor=(0.5, 1.005))

    fig.suptitle("rare_train_size robustness: scRareRefine vs. 8 baselines (6 datasets, seed=42)",
                 fontsize=14, y=1.045)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
