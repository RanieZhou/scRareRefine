"""UMAP 对照图：高可分(immune) vs 边界可分(pancreas)，解释为何低 sep 时 rescue 优势收窄。

读取 plot_umap_rescue.py 导出的 npz（每数据集 1 个），绘制 2×2：
  上行 immune_dc (sep~2.4)：稀有簇独立 → scANVI 全漏判 → rescue 大幅救回
  下行 pancreas (sep~1.4)：稀有簇与多数类接壤 → scANVI 已部分识别 → rescue 收益有限

输出：results/umap/umap_contrast_sep.png / .pdf
前置：先跑 plot_umap_rescue.py 生成两个 npz。
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 0.9, "savefig.dpi": 300,
})

NPZ = {
    "immune":   Path("results/umap/umap_rescue_immune_dc.npz"),
    "pancreas": Path("results/umap/umap_rescue_pancreas_baron.npz"),
}
OUT_PNG = Path("results/umap/umap_contrast_sep.png")
OUT_PDF = Path("results/umap/umap_contrast_sep.pdf")

GRAY = "#d9d9d9"; RED = "#d62728"
BLUE = "#1f77b4"; GREEN = "#2ca02c"; ORANGE = "#ff7f0e"


def panel_truth(ax, d, tag):
    xy = d["xy"]; tr = d["true_rare"]
    rare = str(d["rare"]); ds = str(d["dataset"]); sep = float(d["sep"])
    ax.scatter(xy[~tr, 0], xy[~tr, 1], s=4, c=GRAY, alpha=0.45, linewidths=0, rasterized=True)
    ax.scatter(xy[tr, 0], xy[tr, 1], s=20, c=RED, edgecolors="k", linewidths=0.3,
               label=f"{rare} (true, n={int(tr.sum())})")
    ax.set_title(f"{tag} {ds}  —  ground truth  (sep={sep:.2f})", fontsize=12, loc="left")
    ax.legend(loc="best", fontsize=9, markerscale=1.3)


def panel_outcome(ax, d, tag):
    xy = d["xy"]
    tr = d["true_rare"]; fp = d["fp_rescue"]; tp = d["tp_rescue"]
    ok = d["already_ok"]; miss = d["missed"]
    rec0 = float(d["rec_scanvi"]); rec1 = float(d["rec_srr"]); prec = float(d["prec_srr"])
    ax.scatter(xy[~tr & ~fp, 0], xy[~tr & ~fp, 1], s=4, c=GRAY, alpha=0.4,
               linewidths=0, rasterized=True)
    ax.scatter(xy[ok, 0], xy[ok, 1], s=22, c=BLUE, edgecolors="k", linewidths=0.3,
               label=f"already correct (n={int(ok.sum())})")
    ax.scatter(xy[tp, 0], xy[tp, 1], s=30, c=GREEN, marker="*", edgecolors="k",
               linewidths=0.3, label=f"rescued ✓ (n={int(tp.sum())})")
    ax.scatter(xy[miss, 0], xy[miss, 1], s=22, c=ORANGE, marker="v", edgecolors="k",
               linewidths=0.3, label=f"still missed (n={int(miss.sum())})")
    ax.scatter(xy[fp, 0], xy[fp, 1], s=40, c=RED, marker="x", linewidths=1.4,
               label=f"false rescue (n={int(fp.sum())})")
    ax.set_title(f"{tag} rescue outcome  —  recall {rec0:.2f}→{rec1:.2f}, prec {prec:.2f}",
                 fontsize=12, loc="left")
    ax.legend(loc="best", fontsize=8.5, markerscale=1.2)


def main():
    d_im = np.load(NPZ["immune"], allow_pickle=True)
    d_pa = np.load(NPZ["pancreas"], allow_pickle=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    panel_truth(axes[0, 0], d_im, "(a)")
    panel_outcome(axes[0, 1], d_im, "(b)")
    panel_truth(axes[1, 0], d_pa, "(c)")
    panel_outcome(axes[1, 1], d_pa, "(d)")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1", fontsize=10); ax.set_ylabel("UMAP-2", fontsize=10)

    fig.suptitle(
        "Why scRareRefine's gain shrinks at low separability\n"
        "high-sep (immune_dc, sep=2.4): isolated rare cluster → scANVI misses all → large rescue (TP=120, FP=3)\n"
        "low-sep (pancreas_baron, sep=1.4): rare cluster abuts majority → scANVI already finds 67% → limited rescue (TP=6, missed=22)",
        fontsize=12.5, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
