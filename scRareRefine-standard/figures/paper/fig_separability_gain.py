"""
Separability ratio vs F1 gain scatter plot.
Nature single-column style, 88 x 82 mm.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
OUT  = ROOT / "fig_separability_gain"

# ── Data ──────────────────────────────────────────────────────────────────────
df   = pd.read_csv(ROOT / "table_separability.csv")
df20 = df[df["n_rare_train"] == 20].copy()

# Cell-type display config: (label, color, marker)
META = {
    ("immune_dc",       "ASDC"):                     ("ASDC",    "#3A6FA8", "o"),
    ("immune_dc",       "cDC1"):                     ("cDC1",    "#7B4DA8", "s"),
    ("pancreas",        "epsilon"):                  ("ε-cell",  "#D47A2A", "^"),
    ("pancreas",        "gamma"):                    ("γ-cell",  "#B83535", "v"),
    ("tabula_liver",    "non-classical monocyte"):   ("NCM",     "#2A8A78", "D"),
    ("tabula_pancreas", "type B pancreatic cell"):   ("β-cell",  "#888888", "P"),
    ("tabula_spleen",   "innate lymphoid cell"):     ("ILC",     "#C05A8A", "h"),
    ("tabula_kidney",   "endothelial cell"):         ("EC",      "#5A8AC0", "8"),
}

SEP_THR = 1.3

# Drop rows without f1_gain (pipeline still running for that dataset)
df20 = df20.dropna(subset=["f1_gain"])

# Aggregate per (dataset, rare_class)
agg = (
    df20
    .groupby(["dataset", "rare_class"])
    .agg(
        sep_mean=("separability_ratio", "mean"),
        sep_std =("separability_ratio", "std"),
        gain_mean=("f1_gain", "mean"),
        gain_std =("f1_gain", "std"),
    )
    .reset_index()
)
# Fill std=NaN (single seed) with 0
agg["sep_std"]  = agg["sep_std"].fillna(0)
agg["gain_std"] = agg["gain_std"].fillna(0)

# Spearman on aggregate means
rho, pval = spearmanr(agg["sep_mean"], agg["gain_mean"])
p_str = "p < 0.05" if pval < 0.05 else f"p = {pval:.2f}"

# ── Figure ────────────────────────────────────────────────────────────────────
FIGW, FIGH = 3.46, 3.22   # inches (88 × 82 mm)
fig, ax = plt.subplots(figsize=(FIGW, FIGH))

# Background shading
ax.axvspan(0.55, SEP_THR, alpha=0.055, color="#C94F2C", zorder=0, lw=0)
ax.axvspan(SEP_THR, 2.55, alpha=0.055, color="#4A7C59", zorder=0, lw=0)
ax.axvline(SEP_THR, color="#AAAAAA", lw=0.9, ls="--", zorder=1)

# ── Per-seed dots (jittered slightly to reduce overplotting) ──────────────────
rng = np.random.default_rng(0)
for _, row in df20.iterrows():
    key = (row["dataset"], row["rare_class"])
    _, color, marker = META[key]
    jx = rng.uniform(-0.012, 0.012)
    ax.scatter(row["separability_ratio"] + jx, row["f1_gain"],
               color=color, marker=marker, s=16, alpha=0.40,
               linewidths=0, zorder=3)

# ── Aggregate means with error bars ──────────────────────────────────────────
# Label x-offsets to avoid collisions
LABEL_OFFSET = {
    "ASDC":   ( 0.06,  0.00),
    "cDC1":   ( 0.06,  0.04),
    "ε-cell": ( 0.06, -0.04),
    "γ-cell": (-0.06, -0.04),
    "NCM":    ( 0.06,  0.00),
    "β-cell": (-0.06,  0.00),
    "ILC":    ( 0.06,  0.00),
    "EC":     (-0.06,  0.04),
}
LABEL_HA = {
    "ASDC": "left", "cDC1": "left", "ε-cell": "left",
    "γ-cell": "right", "NCM": "left", "β-cell": "right",
    "ILC": "left", "EC": "right",
}

for _, row in agg.iterrows():
    key   = (row["dataset"], row["rare_class"])
    label, color, marker = META[key]
    x, y  = row["sep_mean"], row["gain_mean"]
    xe, ye = row["sep_std"],  row["gain_std"]

    ax.errorbar(x, y, xerr=xe, yerr=ye,
                fmt="none", ecolor=color, elinewidth=0.75,
                capsize=2.2, capthick=0.75, zorder=4)
    ax.scatter(x, y, color=color, marker=marker, s=58, zorder=5,
               linewidths=0.6, edgecolors="white")

    dx, dy = LABEL_OFFSET[label]
    ha     = LABEL_HA[label]
    ax.text(x + dx, y + dy, label,
            fontsize=6.2, color=color, ha=ha, va="center", zorder=6)

# ── Zone labels ───────────────────────────────────────────────────────────────
trans = ax.get_xaxis_transform()   # x in data, y in axes [0,1]
ax.text(SEP_THR - 0.06, 0.96, "fallback zone",
        transform=trans, ha="right", va="top",
        fontsize=6, color="#C94F2C", style="italic")
ax.text(SEP_THR + 0.06, 0.96, "effective zone",
        transform=trans, ha="left",  va="top",
        fontsize=6, color="#4A7C59", style="italic")

# ── Spearman annotation ───────────────────────────────────────────────────────
ax.text(0.97, 0.05,
        f"Spearman ρ = {rho:.2f}  ({p_str})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.2, color="#555555")

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.55, 2.55)
ax.set_ylim(-0.12, 1.08)
ax.set_xlabel("Separability ratio", fontsize=7.5)
ax.set_ylabel("F1 gain  (scRareRefine − scANVI baseline)", fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=6.5, direction="out")
ax.axhline(0, color="#CCCCCC", lw=0.6, zorder=0)

fig.tight_layout(pad=0.4)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("svg", "pdf"):
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}.*")
print(f"Spearman ρ = {rho:.3f}, p = {pval:.4f}")
