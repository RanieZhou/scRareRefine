"""separability vs rescue 收益散点（G21 — 论证 CONFORMAL_LOW_SEP=1.3 非 cherry-pick）。

读 results/ablation/ablation_summary.csv 的 V6_full 行（= 当前主方法），
画 separability_ratio（x）对 rescue F1 增益 f1_gain（y）的散点：
  - 竖线 = CONFORMAL_LOW_SEP=1.3（弃权下限，跨数据集固定先验）
  - 弃权配置（abstain=True）单独标记（gain 必为 0）
  - 不重训、不碰 test 标签，纯读现有缓存派生

目的：让审稿人直观看到「sep 低 → 收益不稳 / 需弃权；sep 高 → 收益稳定为正」，
说明 1.3 这个阈值落在收益由不稳转稳的过渡带，而非为某数据集挑出来的魔法值。

用法：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/plot_sep_vs_gain.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
ABL = ROOT / "results" / "ablation" / "ablation_summary.csv"
OUT_PNG = ROOT / "results" / "ablation" / "sep_vs_gain.png"
OUT_PDF = ROOT / "results" / "ablation" / "sep_vs_gain.pdf"
LOW_SEP = 1.3

DS_COLORS = {
    "immune_dc": "#1f77b4",
    "pancreas_baron": "#ff7f0e",
    "tabula_lung_endo": "#2ca02c",
    "tabula_small_intestine": "#9467bd",
    "tabula_sapiens_stomach": "#d62728",
    "pancreas_integrated": "#8c564b",
}


def main():
    df = pd.read_csv(ABL)
    v6 = df[df["variant"] == "V6_full"].copy()
    if v6.empty:
        raise SystemExit("ablation_summary.csv 无 V6_full 行，先跑 tools/analysis/ablation.py")

    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for ds, g in v6.groupby("dataset"):
        c = DS_COLORS.get(ds, "#444")
        abst = g["abstain"].astype(bool)
        # 非弃权：实心圆
        ax.scatter(g.loc[~abst, "sep"], g.loc[~abst, "f1_gain"],
                   s=70, c=c, edgecolors="k", linewidths=0.6, label=ds, zorder=3)
        # 弃权：空心方块（gain=0）
        ax.scatter(g.loc[abst, "sep"], g.loc[abst, "f1_gain"],
                   s=80, facecolors="none", edgecolors=c, linewidths=1.6,
                   marker="s", zorder=3)

    ax.axvline(LOW_SEP, color="k", ls="--", lw=1.2)
    ax.text(LOW_SEP + 0.02, ax.get_ylim()[1] * 0.92,
            f"CONFORMAL_LOW_SEP={LOW_SEP}\n(abstain to the left)", fontsize=8, va="top")
    ax.axhline(0.0, color="#999", ls=":", lw=0.9, zorder=1)

    ax.set_xlabel("separability_ratio (train-only)")
    ax.set_ylabel("rescue F1 gain over scANVI (V6_full − baseline)")
    ax.set_title("Separability vs rescue gain across 6 datasets × 4 rts (seed=42)\n"
                 "● applied   □ abstained (gain=0)", fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")

    # 简短文字证据
    below = v6[v6["sep"] < LOW_SEP]
    above = v6[v6["sep"] >= LOW_SEP]
    print(f"\nsep < {LOW_SEP}: {len(below)} 配置，弃权 {int(below['abstain'].sum())}，"
          f"平均 gain {below['f1_gain'].mean():.4f}")
    print(f"sep >= {LOW_SEP}: {len(above)} 配置，弃权 {int(above['abstain'].sum())}，"
          f"平均 gain {above['f1_gain'].mean():.4f}")


if __name__ == "__main__":
    main()
