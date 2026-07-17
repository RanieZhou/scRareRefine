"""绘制可控 sep 扫描曲线（第十四轮，G21）。

两面板（x=实际 sep，因 t→sep 非单调，按 sep 排序）：
  左：F1 —— baseline / full(带 sep 闸门) / nogate(关 sep 闸门)
  右：incremental FPR —— full / nogate，标 α=0.01 与 CONFORMAL_LOW_SEP=1.3
标出经验崩塌边界（nogate incremental FPR 首次破 α）。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/sep_sweep")
LOW_SEP = 1.3
ALPHA = 0.01
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})

d = pd.read_csv(OUT / "sep_sweep_summary.csv").sort_values("sep").reset_index(drop=True)
x = d["sep"].to_numpy()

# 经验崩塌点：nogate incremental FPR 首次超过 α（按 sep 升序找最大的越界 sep）
broke = d[d["nogate_ffr"] > ALPHA]
break_sep = float(broke["sep"].max()) if len(broke) else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 左：F1
ax1.plot(x, d["baseline_f1"], "o-", color="#7f7f7f", label="scANVI baseline")
ax1.plot(
    x,
    d["full_f1"],
    "o-",
    color="#2ca02c",
    lw=2,
    label="scRareRefine full (with sep gate)",
)
ax1.plot(x, d["nogate_f1"], "s--", color="#ff7f0e", label="rescue without sep gate")
ax1.axvline(LOW_SEP, color="k", ls="--", lw=1.2)
ax1.text(
    LOW_SEP + 0.02,
    0.05,
    f"CONFORMAL_LOW_SEP={LOW_SEP}\n(full abstains to the left)",
    fontsize=8,
    va="bottom",
)
ax1.axvspan(x.min() - 0.05, LOW_SEP, color="#cccccc", alpha=0.18)
if break_sep:
    ax1.axvline(break_sep, color="#C0392B", ls=":", lw=1.2)
ax1.set_xlabel("separability_ratio (train-only)")
ax1.set_ylabel("rare-cell F1")
ax1.set_title("F1 vs separability")
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(fontsize=8, loc="upper left")
ax1.set_ylim(-0.02, 1.02)

# 右：incremental FPR
ax2.plot(x, d["full_ffr"], "o-", color="#2ca02c", lw=2, label="full (with sep gate)")
ax2.plot(x, d["nogate_ffr"], "s--", color="#ff7f0e", label="without sep gate")
ax2.axhline(ALPHA, color="#C0392B", ls="--", lw=1.2)
ax2.text(
    x.max(),
    ALPHA + 0.0004,
    "alpha = 0.01",
    color="#C0392B",
    fontsize=9,
    ha="right",
    va="bottom",
)
ax2.axvline(LOW_SEP, color="k", ls="--", lw=1.2)
ax2.axvspan(x.min() - 0.05, LOW_SEP, color="#cccccc", alpha=0.18)
if break_sep:
    ax2.axvline(break_sep, color="#C0392B", ls=":", lw=1.2)
    ax2.text(
        break_sep + 0.02,
        ALPHA * 1.3,
        f"empirical incremental-FPR break\n~ sep {break_sep:.2f}",
        color="#C0392B",
        fontsize=8,
        va="bottom",
    )
ax2.set_xlabel("separability_ratio (train-only)")
ax2.set_ylabel("Incremental rescue-induced FPR")
ax2.set_title("Incremental FPR vs separability")
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Controlled separability sweep (lung_endo, semi-synthetic entanglement, seed=42, rts=0.05).\n"
    "Grey band = full method abstains (sep < 1.3).  Without the gate, incremental FPR stays within budget down to ~0.76 and exceeds it at ~0.69.",
    fontsize=10,
    y=1.02,
)
fig.tight_layout()
fig.savefig(OUT / "sep_sweep.png", bbox_inches="tight")
print(
    f"[saved] {OUT / 'sep_sweep.png'}  | empirical incremental-FPR break sep ~ {break_sep}"
)
