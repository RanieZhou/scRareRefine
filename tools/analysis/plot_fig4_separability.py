"""Fig 4 separability-gate 分析（合成 stress test + 真实 benchmark sensitivity）。

三联面板（PNG only）：
  (a) 合成 sep 扫描 — F1 vs sep：baseline / full(带闸门) / nogate(关闸门)；灰带=full 弃权(sep<1.3)
  (b) 合成 sep 扫描 — FFR vs sep：full / nogate + α 线；nogate 在最低 sep 才破 α
  (c) 真实 benchmark — low_sep 阈值 vs mean F1(左) + worst-case FFR(右)：1.3 是 FFR≤α 的最小阈值

数据源：results/sep_sweep/sep_sweep_summary.csv、lowsep_sensitivity_agg.csv
输出：results/sep_sweep/fig4_separability.png（仅 PNG）
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 8, "savefig.dpi": 300})
OUT = Path("results/sep_sweep")
LOW_SEP, ALPHA = 1.3, 0.01
GREEN, ORANGE, GREY, RED = "#1A7A4A", "#C97A50", "#888888", "#B04040"

sw = pd.read_csv(OUT / "sep_sweep_summary.csv").sort_values("sep")
ls = pd.read_csv(OUT / "lowsep_sensitivity_agg.csv")
lsa = ls[ls.region == "ALL"].sort_values("low_sep")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) 合成：F1 vs sep
ax = axes[0]; x = sw.sep.to_numpy()
ax.plot(x, sw.baseline_f1, "o-", color=GREY, label="scANVI baseline")
ax.plot(x, sw.full_f1, "o-", color=GREEN, lw=2, label="scRareRefine (gate on)")
ax.plot(x, sw.nogate_f1, "s--", color=ORANGE, label="rescue (gate off)")
ax.axvspan(x.min() - 0.05, LOW_SEP, color="#cccccc", alpha=0.18)
ax.axvline(LOW_SEP, color="k", ls="--", lw=1.1)
ax.text(LOW_SEP + 0.03, 0.04, "gate=1.3\n(abstain ←)", fontsize=8, va="bottom")
ax.set_xlabel("separability_ratio"); ax.set_ylabel("rare-cell F1"); ax.set_ylim(-0.02, 1.02)
ax.set_title("(a) Synthetic sweep: F1"); ax.grid(True, ls="--", alpha=0.35); ax.legend(loc="upper left")

# (b) 合成：FFR vs sep
ax = axes[1]
ax.plot(x, sw.full_ffr, "o-", color=GREEN, lw=2, label="gate on")
ax.plot(x, sw.nogate_ffr, "s--", color=ORANGE, label="gate off")
ax.axhline(ALPHA, color=RED, ls="--", lw=1.1); ax.text(x.max(), ALPHA + 0.0004, "α=0.01", color=RED, fontsize=8, ha="right")
ax.axvspan(x.min() - 0.05, LOW_SEP, color="#cccccc", alpha=0.18); ax.axvline(LOW_SEP, color="k", ls="--", lw=1.1)
broke = sw[sw.nogate_ffr > ALPHA]
if len(broke):
    bs = float(broke.sep.max()); ax.axvline(bs, color=RED, ls=":", lw=1.1)
    ax.text(bs + 0.03, ALPHA * 1.4, f"FFR breaks\nonly at sep~{bs:.2f}", color=RED, fontsize=8, va="bottom")
ax.set_xlabel("separability_ratio"); ax.set_ylabel("FFR"); ax.set_title("(b) Synthetic sweep: FFR")
ax.grid(True, ls="--", alpha=0.35); ax.legend(loc="upper right")

# (c) 真实 benchmark：low_sep sensitivity 双轴
ax = axes[2]; ax2 = ax.twinx(); xl = lsa.low_sep.to_numpy()
l1 = ax.plot(xl, lsa.f1_mean, "o-", color=GREEN, lw=2, label="mean rare F1 (L)")
l2 = ax2.plot(xl, lsa.ffr_max, "s--", color=ORANGE, lw=2, label="worst-case FFR (R)")
ax2.axhline(ALPHA, color=RED, ls="--", lw=1.1); ax2.text(1.62, ALPHA + 0.0004, "α=0.01", color=RED, fontsize=8, ha="right")
ax.axvline(LOW_SEP, color="k", ls="--", lw=1.0); ax.text(LOW_SEP + 0.02, lsa.f1_mean.min(), "default 1.3", fontsize=8)
ax.set_xlabel("CONFORMAL_LOW_SEP (gate threshold)"); ax.set_ylabel("mean rare F1", color=GREEN)
ax2.set_ylabel("worst-case FFR", color=ORANGE)
ax.set_title("(c) Real benchmark: low_sep sensitivity")
ax.grid(True, ls="--", alpha=0.3)
ll = l1 + l2; ax.legend(ll, [a.get_label() for a in ll], loc="center right")

fig.suptitle("Separability gate (CONFORMAL_LOW_SEP=1.3): conservative, with a dataset-dependent risk axis.  "
             "(a,b) synthetic stress test on lung_endo: rescue stays FFR-safe well below 1.3.  "
             "(c) real benchmark: 1.3 is the smallest gate keeping worst-case FFR ≤ α.",
             fontsize=9, y=1.04)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig4_separability.png", bbox_inches="tight")
print(f"[saved] {OUT/'fig4_separability.png'}")
