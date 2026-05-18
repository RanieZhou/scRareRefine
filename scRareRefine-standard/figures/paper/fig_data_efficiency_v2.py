"""
Data efficiency: rare_train_size vs rare F1, baseline vs scRareRefine.
Nature single-column style, 88 x 90 mm.
Shows the advantage at low-label regime.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

ROOT = Path(__file__).parent
OUT  = ROOT / "fig_data_efficiency_v2"

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "table_trainsize.csv")

# Keep only baseline and prototype_gate_marker; drop "all" size
df = df[df["method"].isin(["baseline", "prototype_gate_marker"])]
df = df[df["rare_train_size"] != "all"]
df["rts"] = df["rts_num"].astype(float)

# Panel definition: (dataset, rare_class, title, sep_ratio_str)
PANELS = [
    ("immune_dc",    "ASDC",                    "ASDC\n(sep = 1.50)",  "#3A6FA8"),
    ("immune_dc",    "cDC1",                    "cDC1\n(sep = 1.39)",  "#7B4DA8"),
    ("tabula_liver", "non-classical monocyte",  "NCM\n(sep = 2.01)",   "#2A8A78"),
    ("pancreas",     "epsilon",                 "ε-cell\n(sep = 1.07)","#D47A2A"),
]

METHOD_STYLE = {
    "baseline":              dict(color="#AAAAAA", ls="--", lw=1.1, label="scANVI baseline", zorder=3),
    "prototype_gate_marker": dict(color=None,      ls="-",  lw=1.3, label="scRareRefine",   zorder=4),
}
# Each panel gets its own accent color for scRareRefine line
PANEL_COLORS = [meta[3] for meta in PANELS]

# ── Figure layout: 2×2 ───────────────────────────────────────────────────────
FIGW, FIGH = 3.70, 3.70   # inches (94 × 94 mm) — slightly wider for 2x2
fig, axes = plt.subplots(2, 2, figsize=(FIGW, FIGH), sharey=False)
axes = axes.flatten()

X_TICKS = [5, 10, 20, 50]
X_LABELS = ["5", "10", "20", "50"]

for idx, (dataset, rare_class, title, accent) in enumerate(PANELS):
    ax = axes[idx]

    sub = df[(df["dataset"] == dataset) & (df["rare_class"] == rare_class)]

    for method, style in METHOD_STYLE.items():
        ms = sub[sub["method"] == method].sort_values("rts")
        if ms.empty:
            continue

        color = accent if style["color"] is None else style["color"]

        # Line
        ax.plot(ms["rts"], ms["mean"],
                color=color, ls=style["ls"], lw=style["lw"],
                zorder=style["zorder"], marker="o", markersize=3.0,
                markerfacecolor=color, markeredgewidth=0)

        # Shaded ± SD band
        ax.fill_between(ms["rts"],
                        ms["mean"] - ms["std"].fillna(0),
                        ms["mean"] + ms["std"].fillna(0),
                        alpha=0.12, color=color, linewidth=0,
                        zorder=style["zorder"] - 1)

    # Annotate gap at lowest available train size
    lo = sub[sub["method"] == "baseline"]["rts"].min()
    base_lo  = sub[(sub["method"] == "baseline")       & (sub["rts"] == lo)]["mean"].values
    ours_lo  = sub[(sub["method"] == "prototype_gate_marker") & (sub["rts"] == lo)]["mean"].values
    if len(base_lo) and len(ours_lo) and ours_lo[0] - base_lo[0] > 0.05:
        ax.annotate("",
                    xy=(lo, ours_lo[0]), xytext=(lo, base_lo[0]),
                    arrowprops=dict(arrowstyle="<->", lw=0.7,
                                    color=accent, mutation_scale=6),
                    zorder=6)
        mid_y = (base_lo[0] + ours_lo[0]) / 2
        ax.text(lo + 1.5, mid_y, f"+{ours_lo[0]-base_lo[0]:.2f}",
                fontsize=5.6, color=accent, va="center", ha="left")

    ax.set_title(title, fontsize=6.5, color=accent, pad=3, fontweight="bold")
    ax.set_xlim(3, 58)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_LABELS, fontsize=6.2)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=6.2)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1.0, color="#E0E0E0", lw=0.5, zorder=0)

# Shared axis labels
fig.text(0.52, 0.01, "Rare training size", ha="center", fontsize=7.5)
fig.text(0.01, 0.52, "Rare-class F1", va="center", rotation=90, fontsize=7.5)

# Legend (bottom center)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="#AAAAAA", ls="--", lw=1.2, label="scANVI baseline"),
    Line2D([0], [0], color="#555555", ls="-",  lw=1.2, label="scRareRefine"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=2, fontsize=6.2, frameon=False,
           bbox_to_anchor=(0.52, -0.04))

fig.tight_layout(rect=[0.04, 0.06, 1, 1], h_pad=1.4, w_pad=1.0)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("svg", "pdf"):
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}.*")
