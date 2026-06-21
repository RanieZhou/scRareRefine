"""Round 12 ablation 绘图 — 论文级版式（无 overlap）。

产出 2 套图：
  ablation_bars.{png,pdf}   — 2×6 分面柱状（F1 上、FFR_max 下），顶部公共图例，xtick 仅 V0..V7
  ablation_heatmap.{png,pdf} — 单图热力图（6 数据集 × 8 变体），格内显示 F1 数值

输入：results/ablation/ablation_summary_agg.csv
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
    "font.size":        10,
    "axes.linewidth":   0.9,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "savefig.dpi":      300,
})

AGG = Path("results/ablation/ablation_summary_agg.csv")
OUT_DIR = Path("results/ablation")
FFR_LIMIT = 0.01

# (key, short tag for xtick, long legend label, color, font weight for legend)
VARIANTS = [
    ("V0_baseline_scanvi", "V0", "V0 · baseline scANVI",          "#9e9e9e", "normal"),
    ("V1_no_sep_gate",     "V1", "V1 · no sep gate (LOW_SEP=0)",  "#6baed6", "normal"),
    ("V2_no_necessity",    "V2", "V2 · no necessity guard",       "#fd8d3c", "normal"),
    ("V3_rank1_fixed",     "V3", "V3 · rank=1 fixed",             "#74c476", "normal"),
    ("V4_rank2_fixed",     "V4", "V4 · rank=2 fixed",             "#9e9ac8", "normal"),
    ("V5_no_conformal_tau","V5", "V5 · no conformal τ",           "#e7298a", "normal"),
    ("V6_full",            "V6", "V6 · full (ours)",              "#2ca02c", "bold"),
    ("V7_rank3_fixed",     "V7", "V7 · rank=3 fixed (sensitivity)","#bcbddc", "normal"),
]
DATASETS = [
    ("immune_dc",              "immune_dc\n(ASDC)"),
    ("pancreas_baron",         "pancreas_baron\n(gamma)"),
    ("pancreas_integrated",    "pancreas_int.\n(endothelial)"),
    ("tabula_lung_endo",       "lung_endo\n(lymph endo)"),
    ("tabula_sapiens_stomach", "stomach\n(mast cell)"),
    ("tabula_small_intestine", "small_intestine\n(tuft cell)"),
]


def _values(df, ds, col):
    sub = df[df["dataset"] == ds]
    out = []
    for key, *_ in VARIANTS:
        m = sub[sub["variant"] == key]
        out.append(float(m[col].iloc[0]) if len(m) else np.nan)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — 2×6 bar facets，无柱顶数值，顶部统一图例
# ──────────────────────────────────────────────────────────────────────────────
def plot_bars(df):
    n_ds = len(DATASETS)
    n_v = len(VARIANTS)
    # 给每个 panel 留 2.6"，加一些边距；顶部留 1.2" 给图例
    fig_w = 2.6 * n_ds + 1.2
    fig_h = 6.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=2, ncols=n_ds,
        left=0.055, right=0.99, top=0.78, bottom=0.10,
        hspace=0.35, wspace=0.18,
    )
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(n_ds)]
    axes_bot = [fig.add_subplot(gs[1, c]) for c in range(n_ds)]

    colors = [c for *_, c, _ in VARIANTS]
    tags = [t for _, t, *_ in VARIANTS]
    x = np.arange(n_v)

    # 计算共用 ylim
    all_ffrs = np.array([_values(df, ds, "ffr_max") for ds, _ in DATASETS])
    ffr_ymax = max(0.018, float(np.nanmax(all_ffrs)) * 1.18)

    for col, (ds, ds_label) in enumerate(DATASETS):
        f1s = _values(df, ds, "f1_mean")
        ffrs = _values(df, ds, "ffr_max")

        ax = axes_top[col]
        bars = ax.bar(x, f1s, color=colors, edgecolor="black", linewidth=0.5, width=0.78)
        # 高亮 V6 边框
        bars[6].set_edgecolor("#1b5e20"); bars[6].set_linewidth(1.4)
        ax.set_title(ds_label, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("rare F1\n(mean over 4 rts)", fontsize=10)
        else:
            ax.set_yticklabels([])

        ax = axes_bot[col]
        bars = ax.bar(x, ffrs, color=colors, edgecolor="black", linewidth=0.5, width=0.78)
        bars[6].set_edgecolor("#1b5e20"); bars[6].set_linewidth(1.4)
        ax.axhline(FFR_LIMIT, color="#C0392B", linestyle="--", linewidth=1.0, zorder=0)
        ax.text(n_v - 0.3, FFR_LIMIT + ffr_ymax * 0.02, f"α={FFR_LIMIT}",
                color="#C0392B", fontsize=8, ha="right", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=9)
        ax.set_ylim(0, ffr_ymax)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("FFR (max over 4 rts)", fontsize=10)
        else:
            ax.set_yticklabels([])

    # 顶部统一图例（4 列 × 2 行）
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=lbl)
               for _, _, lbl, c, _ in VARIANTS]
    leg = fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
        ncol=4, fontsize=9, frameon=False, columnspacing=2.0, handletextpad=0.5,
    )
    for txt, (_, _, _, _, wt) in zip(leg.get_texts(), VARIANTS):
        txt.set_fontweight(wt)

    fig.suptitle(
        "Ablation across 6 datasets × 4 rts (seed=42).  Top: rare F1.  Bottom: FFR (max across rts).",
        fontsize=11, y=0.995,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "ablation_bars.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "ablation_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR/'ablation_bars.png'}")
    print(f"[saved] {OUT_DIR/'ablation_bars.pdf'}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — 热力图（datasets × variants），格内显示 F1 数值
# ──────────────────────────────────────────────────────────────────────────────
def plot_heatmap(df):
    n_ds = len(DATASETS)
    n_v = len(VARIANTS)
    M_f1 = np.full((n_ds, n_v), np.nan)
    M_ffr = np.full((n_ds, n_v), np.nan)
    for i, (ds, _) in enumerate(DATASETS):
        sub = df[df["dataset"] == ds]
        for j, (key, *_) in enumerate(VARIANTS):
            m = sub[sub["variant"] == key]
            if len(m):
                M_f1[i, j] = m["f1_mean"].iloc[0]
                M_ffr[i, j] = m["ffr_max"].iloc[0]

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(
        nrows=1, ncols=2, left=0.10, right=0.99, top=0.88, bottom=0.18, wspace=0.18,
        width_ratios=[1, 1],
    )

    # left: F1 heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M_f1, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_v))
    ax.set_xticklabels([t for _, t, *_ in VARIANTS], fontsize=10)
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels([lbl.replace("\n", " ") for _, lbl in DATASETS], fontsize=10)
    ax.set_title("rare F1 (mean over 4 rts)", fontsize=11, pad=8)
    for i in range(n_ds):
        for j in range(n_v):
            v = M_f1[i, j]
            if not np.isnan(v):
                color = "white" if v >= 0.65 else "#222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5, color=color)
    # highlight V6 column
    ax.add_patch(plt.Rectangle((5.5, -0.5), 1.0, n_ds, fill=False,
                                edgecolor="#1b5e20", linewidth=2, zorder=10))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("F1", rotation=270, labelpad=12, fontsize=9)

    # right: FFR heatmap
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(M_ffr, cmap="Reds", vmin=0.0,
                   vmax=max(0.02, float(np.nanmax(M_ffr)) * 1.05), aspect="auto")
    ax.set_xticks(range(n_v))
    ax.set_xticklabels([t for _, t, *_ in VARIANTS], fontsize=10)
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels([])
    ax.set_title("FFR (max over 4 rts)  ·  red dashed = α=0.01 contour", fontsize=11, pad=8)
    for i in range(n_ds):
        for j in range(n_v):
            v = M_ffr[i, j]
            if not np.isnan(v):
                over = v > FFR_LIMIT
                color = "white" if v > FFR_LIMIT * 1.5 else "#222"
                txt = f"{v*1000:.1f}‰" if v >= 0.0005 else "0"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight=("bold" if over else "normal"))
    # 在格上标 α violation
    for i in range(n_ds):
        for j in range(n_v):
            if M_ffr[i, j] > FFR_LIMIT:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                            fill=False, edgecolor="#C0392B",
                                            linewidth=1.3, zorder=8))
    ax.add_patch(plt.Rectangle((5.5, -0.5), 1.0, n_ds, fill=False,
                                edgecolor="#1b5e20", linewidth=2, zorder=10))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("FFR", rotation=270, labelpad=12, fontsize=9)

    # bottom legend explaining variant tags
    legend_text = "  ".join([f"{t}={lbl.split('·')[1].strip()}"
                              for _, t, lbl, *_ in VARIANTS])
    fig.text(0.5, 0.04, legend_text, ha="center", va="center",
             fontsize=8.5, color="#333")
    fig.text(0.5, 0.005,
             "Green outline = full method (V6, ours).  Red outline = FFR exceeds α=0.01.",
             ha="center", va="center", fontsize=8.5, color="#555")

    fig.suptitle("Ablation matrix · 6 datasets × 8 variants · seed=42", fontsize=12, y=0.97)
    fig.savefig(OUT_DIR / "ablation_heatmap.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "ablation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR/'ablation_heatmap.png'}")
    print(f"[saved] {OUT_DIR/'ablation_heatmap.pdf'}")


def main():
    df = pd.read_csv(AGG)
    plot_bars(df)
    plot_heatmap(df)


if __name__ == "__main__":
    main()
