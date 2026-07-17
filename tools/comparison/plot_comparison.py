"""绘制对比实验柱状图（scRareRefine vs 8 baselines，6 数据集）。

读取 results/comparison/comparison_summary.csv，绘制 2×N 子图：
  上排：rare-cell F1（当前仅 seed=42；若补 seed 43/44 会自动按 mean±SD 聚合）
  下排：rare-cell Recall（同上）

注：
  - HiCat 为 transductive 方法（PCA/UMAP/DBSCAN 在 train+test 合并特征上 fit，
    阈值取自 test 簇统计），图中以 † 标记，仅作 transductive 上界参照，非公平 inductive 基线。
  - rare_fp_rate 是所有方法可比的 total target-class FPR；只有 scRareRefine 的 rescue
    决策以 incremental FPR 参考预算 α=0.01 校准。

输出：results/comparison/comparison_bars.png / .pdf
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.9,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
    }
)

DETAIL = Path("results/comparison/comparison_summary.csv")
OUT_PNG = Path("results/comparison/comparison_bars.png")

# 图表只展示单一比例，避免把不同标注规模混入均值
# 可选值：None（使用全部数据）、"0.01"、"0.05"、"0.10"、"all"
SHOW_PROPORTION: str | None = "all"

# Low-saturation science palette (colorblind-aware, muted/Morandi style)
METHODS = [
    ("scANVI", "#888888"),  # neutral gray (backbone reference)
    ("kNN", "#5B7FA6"),  # muted slate blue
    ("CellTypist", "#C97A50"),  # muted terracotta/amber
    ("scBalance", "#7D6A9E"),  # muted indigo; scBalance 2023
    ("ProtoCloud", "#C47BAB"),  # muted dusty mauve; Cell Genomics 2026
    ("HiCat", "#6BADB5"),  # muted steel teal; Briefings Bioinf 2025 (transductive†)
    ("scCAD", "#B55D5A"),  # muted brick rose; Nature Commun 2024
    ("TOSICA", "#906B5A"),  # warm sand brown; Nature Commun 2023
    ("scRareRefine", "#1A7A4A"),  # deep emerald (our method)
]

DATASETS = [
    ("immune_dc", "immune_dc\n(ASDC)"),
    ("pancreas_baron", "pancreas_baron\n(gamma)"),
    ("tabula_lung_endo", "tabula_lung_endo\n(lymph EC)"),
    ("tabula_small_intestine", "tabula_small_intestine\n(tuft cell)"),
    ("tabula_sapiens_stomach", "tabula_sapiens_stomach\n(mast cell)"),
    ("pancreas_integrated", "pancreas_integrated\n(endothelial)"),
    ("mouse_lung_tms_10x", "mouse_lung_tms\n(vein EC)"),
    ("mouse_pancreas_tms_10x", "mouse_pancreas_tms\n(D cell)"),
]


def main():
    raw = pd.read_csv(DETAIL, dtype={"rare_train_size": str})
    raw = raw[raw["status"] == "ok"]

    # 过滤至指定比例，保证聚合语义正确（同一比例下按 seed 取均值，非混合比例）
    if SHOW_PROPORTION is not None:
        raw = raw[raw["rare_train_size"].astype(str) == SHOW_PROPORTION]

    # 动态 seed 标签：避免硬编码「3 seeds」与实际数据不符（当前仅 seed=42）
    seeds_present = sorted(raw["seed"].dropna().astype(int).unique().tolist())
    n_seeds = len(seeds_present)
    seed_label = (
        f"seed {seeds_present[0]}" if n_seeds == 1 else f"mean ± SD, {n_seeds} seeds"
    )

    # 按 dataset × method 聚合（同一比例下 3 seeds 的均值）
    agg = (
        raw.groupby(["dataset", "method"])
        .agg(
            f1_mean=("rare_f1", "mean"),
            f1_std=("rare_f1", "std"),
            rec_mean=("rare_recall", "mean"),
            rec_std=("rare_recall", "std"),
        )
        .reset_index()
    )

    # 只绘制数据中实际存在的方法
    present = set(agg["method"].unique())
    active_methods = [(m, c) for m, c in METHODS if m in present]

    x = np.arange(len(active_methods))
    colors = [c for _, c in active_methods]
    # HiCat 是 transductive 方法，x 轴加 † 标记
    xlabels = [m + "†" if m == "HiCat" else m for m, _ in active_methods]

    n_ds = len(DATASETS)
    fig, axes = plt.subplots(2, n_ds, figsize=(4.5 * n_ds, 8.6), sharex=True)

    for col, (ds, title) in enumerate(DATASETS):
        sub = agg[agg["dataset"] == ds].set_index("method")

        sub = sub.reindex([m for m, _ in active_methods])  # 缺失方法保留 NaN
        f1s = sub["f1_mean"].tolist()  # NaN → 不画 bar
        f1_sds = [v if pd.notna(v) else 0.0 for v in sub["f1_std"].tolist()]
        recs = sub["rec_mean"].tolist()
        rec_sds = [v if pd.notna(v) else 0.0 for v in sub["rec_std"].tolist()]

        # ── 上排：F1 ─────────────────────────────────────────────────
        ax = axes[0, col]
        # 缺失方法画灰色斜线占位，有值方法画彩色 bar
        f1s_plot = [v if pd.notna(v) else 0.0 for v in f1s]
        bars = ax.bar(
            x,
            f1s_plot,
            yerr=f1_sds,
            color=colors,
            edgecolor="k",
            linewidth=0.7,
            capsize=4,
            error_kw={"elinewidth": 1.0},
        )
        for i, v in enumerate(f1s):
            if pd.isna(v):
                bars[i].set_hatch("//")
                bars[i].set_facecolor("#eeeeee")
        ours_idx = next(
            (i for i, (m, _) in enumerate(active_methods) if m == "scRareRefine"),
            len(active_methods) - 1,
        )
        bars[ours_idx].set_linewidth(2.0)
        for xi, f1, sd in zip(x, f1s, f1_sds):
            if pd.isna(f1):
                ax.text(
                    xi,
                    0.015,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#888888",
                )
                continue
            is_ours = xi == x[ours_idx]
            ax.text(
                xi,
                f1 + sd + 0.022,
                f"{f1:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_ours else "normal",
                color="#1A7A4A" if is_ours else "black",
            )
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, fontsize=11, pad=6)
        ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)
        if col == 0:
            prop_label = SHOW_PROPORTION if SHOW_PROPORTION else "all proportions"
            ax.set_ylabel(f"Rare-cell F1\n({seed_label}, rts={prop_label})")

        # ── 下排：Recall ──────────────────────────────────────────────
        ax = axes[1, col]
        recs_plot = [v if pd.notna(v) else 0.0 for v in recs]
        bars2 = ax.bar(
            x,
            recs_plot,
            yerr=rec_sds,
            color=colors,
            edgecolor="k",
            linewidth=0.7,
            capsize=4,
            error_kw={"elinewidth": 1.0},
        )
        for i, v in enumerate(recs):
            if pd.isna(v):
                bars2[i].set_hatch("//")
                bars2[i].set_facecolor("#eeeeee")
        bars2[ours_idx].set_linewidth(2.0)
        for xi, r, sd in zip(x, recs, rec_sds):
            if pd.isna(r):
                ax.text(
                    xi,
                    0.015,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#888888",
                )
                continue
            is_ours = xi == x[ours_idx]
            ax.text(
                xi,
                r + sd + 0.022,
                f"{r:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_ours else "normal",
                color="#1A7A4A" if is_ours else "black",
            )
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, axis="y", ls="--", alpha=0.4, linewidth=0.6)
        ax.tick_params(length=4, width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9, rotation=15, ha="right")
        if col == 0:
            ax.set_ylabel(f"Rare-cell Recall\n({seed_label}, rts={prop_label})")

    # 图例
    legend_handles = [
        Patch(
            facecolor=c,
            edgecolor="k",
            label=("scRareRefine (ours)" if m == "scRareRefine" else m),
        )
        for m, c in active_methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(active_methods),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
    )
    fig.suptitle(
        f"Comparison of rare-cell identification methods across {len(DATASETS)} datasets ({seed_label}, rts={SHOW_PROPORTION})",
        fontsize=12,
        y=1.06,
    )

    # 脚注：解释 † 标记 + 操作点差异
    fig.text(
        0.5,
        -0.02,
        "† HiCat is transductive (PCA/UMAP/DBSCAN fit on combined train+test; threshold from test clusters) "
        "— shown as a transductive upper-bound reference, not an inductive baseline.\n"
        "rare_fp_rate reports total target-class FPR for every method; only scRareRefine calibrates "
        "its incremental rescue-induced FPR against α=0.01.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
