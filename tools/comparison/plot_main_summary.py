"""Fig 2 主结果精简汇总图（PNG only）。

两面板（per-method 跨 6 数据集汇总；全 grid 明细见 Supp）：
  (a) 稀缺区(rts≤0.10) rare-cell F1：各方法均值±SD（over 6 数据集 × 3 scarce rts × 3 seed）
  (b) worst-case total target-class FPR：各方法 rare_fp_rate 的全 benchmark 最大值。

数据源：results/comparison/comparison_summary.csv（status==ok，3 seed）。
输出：results/comparison/main_summary.png（仅 PNG，按用户要求不出 PDF）。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
    }
)

DETAIL = Path("results/comparison/comparison_summary.csv")
OUT = Path("results/comparison/main_summary.png")
SCARCE = ["0.01", "0.05", "0.10"]
ALPHA = 0.01
# Low-saturation science palette (colorblind-aware, muted/Morandi style)
METHODS = [
    ("scANVI", "#888888"),  # neutral gray (backbone reference)
    ("kNN", "#5B7FA6"),  # muted slate blue
    ("CellTypist", "#C97A50"),  # muted terracotta/amber
    ("scBalance", "#7D6A9E"),  # muted indigo
    ("ProtoCloud", "#C47BAB"),  # muted dusty mauve
    ("HiCat", "#6BADB5"),  # muted steel teal (transductive)
    ("scCAD", "#B55D5A"),  # muted brick rose
    ("TOSICA", "#906B5A"),  # warm sand brown
    ("scRareRefine", "#1A7A4A"),  # deep emerald (our method)
]
WARN_COLOR = "#B04040"


def main():
    df = pd.read_csv(DETAIL, dtype={"rare_train_size": str})
    df = df[df["status"] == "ok"]
    sc = df[df["rare_train_size"].isin(SCARCE)]

    names = [m for m, _ in METHODS]
    colors = [c for _, c in METHODS]
    xlabels = [
        m + "†"
        if m == "HiCat"
        else ("scRareRefine\n(ours)" if m == "scRareRefine" else m)
        for m in names
    ]
    x = np.arange(len(names))
    ours_i = names.index("scRareRefine")

    # (a) 稀缺区 per-method F1 均值±SD
    f1_mean = [sc[sc.method == m]["rare_f1"].mean() for m in names]
    f1_sd = [sc[sc.method == m]["rare_f1"].std(ddof=0) for m in names]
    # (b) per-method worst-case total target-class FPR
    fpr_max = [df[df.method == m]["rare_fp_rate"].max() for m in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    b1 = ax1.bar(
        x,
        f1_mean,
        yerr=f1_sd,
        color=colors,
        edgecolor="k",
        linewidth=0.6,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )
    b1[ours_i].set_edgecolor("#0D4A2A")
    b1[ours_i].set_linewidth(2.2)
    for xi, v in zip(x, f1_mean):
        ax1.text(
            xi,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if xi == ours_i else "normal",
            color="#1A7A4A" if xi == ours_i else "black",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, rotation=30, ha="right")
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("rare-cell F1")
    ax1.set_title(
        "(a) Label-scarce regime (rts ≤ 0.10)\nmean ± SD over 6 datasets × 3 fractions × 3 seeds"
    )
    ax1.grid(axis="y", ls="--", alpha=0.35)
    ax1.set_axisbelow(True)

    b2 = ax2.bar(x, fpr_max, color=colors, edgecolor="k", linewidth=0.6)
    b2[ours_i].set_edgecolor("#0D4A2A")
    b2[ours_i].set_linewidth(2.2)
    for i, v in enumerate(fpr_max):
        if v > ALPHA:
            b2[i].set_edgecolor(WARN_COLOR)
            b2[i].set_linewidth(1.8)
    ax2.axhline(ALPHA, color=WARN_COLOR, ls="--", lw=1.2)
    ax2.text(
        len(names) - 0.4,
        ALPHA + 0.0008,
        "α = 0.01",
        color=WARN_COLOR,
        fontsize=9,
        ha="right",
        va="bottom",
    )
    for xi, v in zip(x, fpr_max):
        ax2.text(
            xi,
            v + max(fpr_max) * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold" if (xi == ours_i or v > ALPHA) else "normal",
            color=WARN_COLOR if v > ALPHA else ("#1A7A4A" if xi == ours_i else "black"),
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=30, ha="right")
    ax2.set_ylabel("worst-case total target-class FPR")
    ax2.set_title(
        "(b) Worst-case total target-class FPR\n(maximum across all configurations)"
    )
    ax2.grid(axis="y", ls="--", alpha=0.35)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Rare-cell F1 and empirical total target-class false-positive rates.  "
        "† HiCat is transductive (upper-bound reference).",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"[saved] {OUT}")
    # 控制台核对
    print("\nper-method 稀缺区 F1 均值 / worst-case total target-class FPR:")
    for m, fm, ff in zip(names, f1_mean, fpr_max):
        flag = "  <-- >α" if ff > ALPHA else ""
        print(f"  {m:14s} F1={fm:.3f}  total_FPR_max={ff:.4f}{flag}")


if __name__ == "__main__":
    main()
