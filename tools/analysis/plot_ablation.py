"""Ablation 绘图（重构版 2026-06-21）—— 对应新两表结构，3-seed 误差棒。

产出 2 张图（与 ablation.py 的两张表一一对应）：
  ablation_table1_components.{png,pdf}  表1·组件留一法：6 数据集 × {A0..A5}，上 F1(±SD) 下 FFR_max，高亮 A5_full
  ablation_table2_rank.{png,pdf}        表2·rank 敏感性：6 数据集 × {k=1,2,3,adaptive}，上 F1 下 FFR，高亮 adaptive

输入：results/ablation/ablation_table1_components.csv、ablation_table2_rank.csv
（由 tools/analysis/ablation.py 生成；含 f1_mean/f1_std/recall_mean/ffr_max，3 seed × 4 rts）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.9,
    "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "savefig.dpi": 300,
})

OUT_DIR = Path("results/ablation")
FFR_LIMIT = 0.01
DATASETS = [
    ("immune_dc", "immune_dc\n(ASDC)"),
    ("pancreas_baron", "pancreas_baron\n(gamma)"),
    ("pancreas_integrated", "pancreas_int.\n(endothelial)"),
    ("tabula_lung_endo", "lung_endo\n(lymph endo)"),
    ("tabula_sapiens_stomach", "stomach\n(mast cell)"),
    ("tabula_small_intestine", "small_intestine\n(tuft cell)"),
]

# (variant_key, xtick, legend, color, highlight)
# Low-saturation muted palette consistent with comparison plots
TABLE1 = [
    ("A0_baseline",            "base",  "A0 baseline scANVI",          "#AAAAAA", False),  # light gray
    ("A1_minus_sep",           "-sep",  "A1 - sep gate",               "#6A9EC2", False),  # muted sky blue
    ("A2_minus_necessity",     "-nec",  "A2 - necessity guard",        "#D4A55A", False),  # muted amber
    ("A3_minus_adaptive_rank", "-adpt", "A3 - adaptive rank (=> k=1)", "#82B090", False),  # muted sage green
    ("A4_minus_tau",           "-tau",  "A4 - conformal tau",          "#C47BAB", False),  # muted mauve
    ("A5_full",                "full",  "A5 full (ours)",              "#1A7A4A", True),   # deep emerald
]
TABLE2 = [
    ("R1_rank1",    "k=1",     "fixed rank=1",     "#8FA8C0", False),  # light muted blue
    ("R2_rank2",    "k=2",     "fixed rank=2",     "#9E8FC0", False),  # muted lavender
    ("R3_rank3",    "k=3",     "fixed rank=3",     "#B8B8CC", False),  # blue-gray
    ("R_adaptive",  "adaptive","adaptive (ours)",  "#1A7A4A", True),   # deep emerald
]


def _series(df, ds, variants, col):
    sub = df[df["dataset"] == ds]
    out = []
    for key, *_ in variants:
        m = sub[sub["variant"] == key]
        out.append(float(m[col].iloc[0]) if len(m) and col in m else np.nan)
    return out


def plot_facets(df, variants, out_name, title, has_std=True):
    n_ds, n_v = len(DATASETS), len(variants)
    colors = [c for *_, c, _ in variants]
    tags = [t for _, t, *_ in variants]
    hi = [h for *_, h in variants]
    x = np.arange(n_v)

    fig = plt.figure(figsize=(2.55 * n_ds + 1.0, 6.3))
    gs = fig.add_gridspec(2, n_ds, left=0.06, right=0.99, top=0.78, bottom=0.10, hspace=0.33, wspace=0.18)
    all_ffr = np.array([_series(df, ds, variants, "ffr_max") for ds, _ in DATASETS])
    ffr_ymax = max(0.018, float(np.nanmax(all_ffr)) * 1.18)

    for col, (ds, label) in enumerate(DATASETS):
        f1 = _series(df, ds, variants, "f1_mean")
        sd = _series(df, ds, variants, "f1_std") if has_std else [0] * n_v
        ffr = _series(df, ds, variants, "ffr_max")

        ax = fig.add_subplot(gs[0, col])
        bars = ax.bar(x, f1, yerr=sd, color=colors, edgecolor="black", linewidth=0.5,
                      width=0.78, capsize=2.5, error_kw={"elinewidth": 0.8})
        for i, h in enumerate(hi):
            if h:
                bars[i].set_edgecolor("#0D4A2A"); bars[i].set_linewidth(1.6)
        # 数据标签：柱顶显示 F1 值
        for i, (v, s) in enumerate(zip(f1, sd if has_std else [0]*n_v)):
            if not np.isnan(v):
                y_top = v + (s or 0) + 0.015
                ax.text(x[i], y_top, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold" if hi[i] else "normal",
                        color="#0D4A2A" if hi[i] else "#444444")
        ax.set_title(label, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", alpha=0.3, linewidth=0.5); ax.set_axisbelow(True)
        ax.set_ylabel("rare F1\n(mean±SD, 3 seed×4 rts)" if col == 0 else "", fontsize=9)
        if col: ax.set_yticklabels([])

        ax = fig.add_subplot(gs[1, col])
        bars = ax.bar(x, ffr, color=colors, edgecolor="black", linewidth=0.5, width=0.78)
        for i, h in enumerate(hi):
            if h:
                bars[i].set_edgecolor("#0D4A2A"); bars[i].set_linewidth(1.6)
        # 标红越界 FFR
        for i, v in enumerate(ffr):
            if not np.isnan(v) and v > FFR_LIMIT:
                bars[i].set_edgecolor("#B04040"); bars[i].set_linewidth(1.6)
        ax.axhline(FFR_LIMIT, color="#B04040", ls="--", lw=1.0, zorder=0)
        ax.text(n_v - 0.3, FFR_LIMIT + ffr_ymax * 0.02, f"α={FFR_LIMIT}", color="#B04040",
                fontsize=8, ha="right", va="bottom")
        # 数据标签：FFR 柱顶显示值
        def _fmt_ffr(v):
            if v == 0 or v < 5e-5: return "0"
            elif v < 0.001: return f"{v:.4f}"
            elif v < 0.01:  return f"{v:.4f}"
            else:           return f"{v:.3f}"
        for i, v in enumerate(ffr):
            if not np.isnan(v):
                y_top = v + ffr_ymax * 0.03
                ax.text(x[i], y_top, _fmt_ffr(v), ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold" if (hi[i] or v > FFR_LIMIT) else "normal",
                        color="#B04040" if v > FFR_LIMIT else ("#0D4A2A" if hi[i] else "#444444"))
        ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, ffr_ymax * 1.18); ax.grid(axis="y", alpha=0.3, linewidth=0.5); ax.set_axisbelow(True)
        ax.set_ylabel("FFR (max over rts)" if col == 0 else "", fontsize=9)
        if col: ax.set_yticklabels([])

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=lbl) for _, _, lbl, c, _ in variants]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
                     ncol=min(len(variants), 6), fontsize=9, frameon=False, columnspacing=1.6, handletextpad=0.5)
    for txt, (*_, h) in zip(leg.get_texts(), variants):
        if h: txt.set_fontweight("bold")
    fig.suptitle(title, fontsize=11, y=0.995)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR/(out_name+'.png')}")


def main():
    t1 = pd.read_csv(OUT_DIR / "ablation_table1_components.csv")
    t2 = pd.read_csv(OUT_DIR / "ablation_table2_rank.csv")
    plot_facets(
        t1, TABLE1, "ablation_table1_components",
        "Ablation Table 1 · component leave-one-out (6 datasets, 3 seeds × 4 rts).  "
        "Top: rare F1 (mean±SD).  Bottom: FFR max (red dashed = α=0.01).  A5 full = ours.",
    )
    plot_facets(
        t2, TABLE2, "ablation_table2_rank",
        "Ablation Table 2 · candidate-rank sensitivity (6 datasets, 3 seeds × 4 rts).  "
        "adaptive selects per-val rank under FFR≤α; fixed rank=3 over-fires FFR.  adaptive = ours.",
    )


if __name__ == "__main__":
    main()
