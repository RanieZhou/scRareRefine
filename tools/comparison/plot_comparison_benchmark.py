"""Benchmark 版式对比图（仿 benchmark 论文 Figure 5 风格，PNG only）。

两张图，均聚焦标注稀缺区 (rts ∈ {0.01,0.05,0.10}, 3 seed)：

  comparison_heatmap.png  —— 三联热图（method × dataset），cell=稀缺区均值
    (a) rare F1     Greens，星号标每数据集最高
    (b) rare recall Blues， 星号标每数据集最高
    (c) FFR         Reds（反向语义，高=坏），星号标每数据集最低
    HiCat 标 † (transductive，上界参考)。

  comparison_boxplot.png  —— 三行指标 × 9 方法箱线，每方法跨稀缺区分布（6 ds × 3 rts × 3 seed=54 点）
    标中位数；scRareRefine 深绿高亮；FFR 行加 α=0.01 红线。

数据源：results/comparison/comparison_summary.csv（status==ok）。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _light_cmap(name, top=0.70, n=256):
    """截断标准 colormap 到上端 top，使最深值更柔和（参考浅原色 benchmark 配色）。"""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(f"{name}_light", base(np.linspace(0.0, top, n)))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "savefig.dpi": 300,
})

SRC = Path("results/comparison/comparison_summary.csv")
OUT_DIR = Path("results/comparison")
SCARCE = ["0.01", "0.05", "0.10"]
ALPHA = 0.01

# 方法顺序与配色（与 main_summary / sweep_rts 一致）；scRareRefine 置末高亮
METHODS = [
    ("scANVI",       "#888888"),
    ("kNN",          "#5B7FA6"),
    ("CellTypist",   "#C97A50"),
    ("scBalance",    "#7D6A9E"),
    ("ProtoCloud",   "#C47BAB"),
    ("HiCat",        "#6BADB5"),
    ("scCAD",        "#B55D5A"),
    ("TOSICA",       "#906B5A"),
    ("scRareRefine", "#1A7A4A"),
]
OURS = "scRareRefine"
EMERALD = "#1A7A4A"
DEEP_EMERALD = "#0D4A2A"
WARN_COLOR = "#B04040"

# 数据集顺序（与 ablation 一致）
DATASETS = [
    "immune_dc", "pancreas_baron", "pancreas_integrated",
    "tabula_lung_endo", "tabula_sapiens_stomach", "tabula_small_intestine",
]

# (列名, 标题, colormap, 越大越好?)
METRICS = [
    ("rare_f1",     "(a) rare F1",     "Greens", True),
    ("rare_recall", "(b) rare recall", "Blues",  True),
    ("rare_fp_rate", "(c) FFR (false-rescue rate)", "Reds", False),
]


def _scarce(df):
    return df[(df["status"] == "ok") & (df["rare_train_size"].isin(SCARCE))]


def _ds_label(df, ds):
    rc = df[df.dataset == ds]["rare_class"].iloc[0]
    return f"{ds}\n({rc})"


def _xlabels(names):
    out = []
    for m in names:
        if m == OURS:
            out.append("scRareRefine\n(ours)")
        elif m == "HiCat":
            out.append("HiCat †")
        else:
            out.append(m)
    return out


def plot_heatmap(df):
    names = [m for m, _ in METHODS]
    sc = _scarce(df)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    row_labels = [_ds_label(df, ds) for ds in DATASETS]
    xlabels = _xlabels(names)

    for ax, (col, title, cmap, higher_better) in zip(axes, METRICS):
        # 构造 dataset × method 均值矩阵
        M = np.full((len(DATASETS), len(names)), np.nan)
        for r, ds in enumerate(DATASETS):
            for c, m in enumerate(names):
                v = sc[(sc.dataset == ds) & (sc.method == m)][col]
                if len(v):
                    M[r, c] = float(v.mean())

        vmax = np.nanmax(M) if np.isfinite(np.nanmax(M)) else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(M, cmap=_light_cmap(cmap), vmin=0, vmax=vmax, aspect="auto")

        # 仅 F1/recall 联打星号标每行（数据集）最优；不画数值标签。
        # FFR 联越低越好且多数方法并列 0，满屏星号无信息，故不打星，纯颜色 + colorbar 表达。
        if higher_better:
            for r in range(len(DATASETS)):
                row = M[r]
                if not np.any(np.isfinite(row)):
                    continue
                best = np.nanmax(row)
                for c in range(len(names)):
                    if np.isfinite(M[r, c]) and np.isclose(M[r, c], best, atol=1e-9):
                        ax.text(c, r, "*", ha="center", va="center",
                                fontsize=14, color="black", fontweight="bold")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(row_labels if ax is axes[0] else [""] * len(DATASETS), fontsize=8.5)
        ax.set_title(title, fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Per-method × per-dataset comparison in the label-scarce regime (rts ≤ 0.10, mean over 3 fractions × 3 seeds).  "
        "* = best method per dataset in (a) F1 and (b) recall.  In (c) FFR, darker = worse (more false rescues).  "
        "† HiCat is transductive (upper-bound reference).",
        fontsize=10, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "comparison_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def plot_boxplot(df):
    names = [m for m, _ in METHODS]
    colors = [c for _, c in METHODS]
    sc = _scarce(df)
    xlabels = _xlabels(names)
    x = np.arange(len(names))
    ours_i = names.index(OURS)

    box_metrics = [
        ("rare_f1",      "rare F1",      False),
        ("rare_recall",  "rare recall",  False),
        ("rare_fp_rate", "FFR",          True),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for ax, (col, ylabel, is_ffr) in zip(axes, box_metrics):
        data = [sc[sc.method == m][col].dropna().to_numpy() for m in names]
        bp = ax.boxplot(data, positions=x, widths=0.6, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.3},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[i])
            box.set_alpha(0.85)
            box.set_edgecolor(DEEP_EMERALD if i == ours_i else "black")
            box.set_linewidth(2.0 if i == ours_i else 0.7)
        # 中位数标注
        for i, d in enumerate(data):
            if len(d) == 0:
                continue
            med = float(np.median(d))
            label = f"{med:.3f}" if is_ffr else f"{med:.2f}"
            ax.text(x[i] + 0.34, med, label, va="center", ha="left", fontsize=7.5,
                    fontweight="bold" if i == ours_i else "normal",
                    color=EMERALD if i == ours_i else "#333333")
        if is_ffr:
            ax.axhline(ALPHA, color=WARN_COLOR, ls="--", lw=1.2)
            ax.text(len(names) - 0.6, ALPHA, " α=0.01", color=WARN_COLOR,
                    fontsize=9, va="bottom", ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xlabels, rotation=30, ha="right")
    fig.suptitle(
        "Distribution across datasets in the label-scarce regime (rts ≤ 0.10; each box = 6 datasets × 3 fractions × 3 seeds).  "
        "Median labelled beside each box.  † HiCat is transductive.",
        fontsize=10, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "comparison_boxplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    df = pd.read_csv(SRC, dtype={"rare_train_size": str})
    plot_heatmap(df)
    plot_boxplot(df)


if __name__ == "__main__":
    main()
