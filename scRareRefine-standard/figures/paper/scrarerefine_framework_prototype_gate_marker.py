from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
})

OUT = Path(__file__).with_suffix("")

COLORS = {
    "ink": "#27313A",
    "muted": "#66717C",
    "line": "#B7C0C9",
    "panel": "#F7F9FB",
    "reference": "#BFD7EA",
    "query": "#E9D8A6",
    "scanvi": "#C9C3E6",
    "failure": "#F2C6B4",
    "module": "#D8E8D5",
    "rare": "#4A7C59",
    "major": "#8A95A5",
    "accent": "#3D6F8E",
}


def rounded_box(ax, xy, w, h, label, face, edge=None, lw=0.9, radius=0.018, fontsize=7, weight="normal", color=None):
    edge = edge or COLORS["line"]
    color = color or COLORS["ink"]
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=patches.BoxStyle("Round", pad=0.012, rounding_size=radius),
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center", fontsize=fontsize, color=color, weight=weight)
    return box


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.2):
    color = color or COLORS["muted"]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=8),
    )


def curved_arrow(ax, start, end, c1, c2, color=None, lw=1.0):
    color = color or COLORS["muted"]
    path = MplPath([start, c1, c2, end], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = patches.FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color)
    ax.add_patch(patch)


def cells(ax, center, scale=1.0, rare=False):
    x, y = center
    offsets = [(-0.018, 0.018), (0.0, 0.024), (0.018, 0.016), (-0.014, -0.006), (0.006, -0.006), (0.024, -0.012)]
    for i, (dx, dy) in enumerate(offsets):
        fc = COLORS["rare"] if rare and i in (1, 4) else COLORS["major"]
        ax.add_patch(patches.Circle((x + dx * scale, y + dy * scale), 0.0075 * scale, facecolor=fc, edgecolor="white", lw=0.4))


def latent_cloud(ax, x, y, w, h, show_failure=False):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", lw=0.6, edgecolor=COLORS["line"], facecolor="white"))
    majors = [(0.18, 0.32), (0.27, 0.44), (0.34, 0.30), (0.47, 0.42), (0.57, 0.30), (0.68, 0.40), (0.78, 0.32)]
    rares = [(0.25, 0.70), (0.34, 0.76), (0.43, 0.68)]
    for px, py in majors:
        ax.add_patch(patches.Circle((x + px * w, y + py * h), 0.010, facecolor=COLORS["major"], edgecolor="white", lw=0.35))
    for px, py in rares:
        ax.add_patch(patches.Circle((x + px * w, y + py * h), 0.010, facecolor=COLORS["rare"], edgecolor="white", lw=0.35))
    if show_failure:
        ax.add_patch(patches.Circle((x + 0.74 * w, y + 0.56 * h), 0.012, facecolor=COLORS["rare"], edgecolor="white", lw=0.4))
        ax.plot([x + 0.74 * w, x + 0.64 * w], [y + 0.56 * h, y + 0.38 * h], color=COLORS["failure"], lw=1.2)


def probability_bars(ax, x, y, w, h):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", lw=0.6, edgecolor=COLORS["line"], facecolor="white"))
    vals = [0.74, 0.42, 0.18]
    cols = [COLORS["major"], COLORS["accent"], COLORS["rare"]]
    for i, val in enumerate(vals):
        yy = y + h * (0.72 - i * 0.25)
        ax.add_patch(patches.Rectangle((x + 0.13 * w, yy), val * 0.68 * w, 0.045 * h, facecolor=cols[i], edgecolor="none"))


def mini_marker(ax, x, y, w, h):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", lw=0.6, edgecolor=COLORS["line"], facecolor="white"))
    for i, val in enumerate([0.76, 0.64, 0.51, 0.31]):
        ax.add_patch(patches.Rectangle((x + 0.18 * w + i * 0.16 * w, y + 0.18 * h), 0.075 * w, val * 0.58 * h, facecolor=COLORS["rare"], edgecolor="none", alpha=0.85))


fig = plt.figure(figsize=(7.2, 3.8), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.035, 0.94, "scRareRefine framework", ha="left", va="center", fontsize=10, weight="bold", color=COLORS["ink"])
ax.text(0.035, 0.895, "Known rare-cell rescue after scANVI", ha="left", va="center", fontsize=7.2, color=COLORS["muted"])

rounded_box(ax, (0.035, 0.55), 0.14, 0.18, "Partially labeled\nreference", COLORS["reference"], weight="bold")
rounded_box(ax, (0.035, 0.31), 0.14, 0.18, "Held-out\nquery", COLORS["query"], weight="bold")
cells(ax, (0.105, 0.61), 1.15, rare=True)
cells(ax, (0.105, 0.37), 1.15, rare=True)
ax.text(0.105, 0.23, "Input", ha="center", va="center", fontsize=7.5, weight="bold", color=COLORS["ink"])

arrow(ax, 0.19, 0.52, 0.235, 0.52)

rounded_box(ax, (0.245, 0.30), 0.17, 0.43, "scANVI\nbackbone", COLORS["scanvi"], weight="bold", fontsize=8)
latent_cloud(ax, 0.265, 0.49, 0.13, 0.16)
probability_bars(ax, 0.265, 0.335, 0.13, 0.10)
ax.text(0.33, 0.23, "Latent + probabilities", ha="center", va="center", fontsize=7.2, color=COLORS["muted"])

arrow(ax, 0.43, 0.52, 0.47, 0.52)

rounded_box(ax, (0.48, 0.30), 0.15, 0.43, "Baseline\nmiss", COLORS["failure"], weight="bold", fontsize=8)
latent_cloud(ax, 0.50, 0.43, 0.11, 0.18, show_failure=True)
ax.text(0.555, 0.35, "Rare → majority", ha="center", va="center", fontsize=7.1, color=COLORS["ink"])

arrow(ax, 0.645, 0.52, 0.69, 0.52)

module = patches.FancyBboxPatch((0.70, 0.25), 0.20, 0.54, boxstyle="round,pad=0.014,rounding_size=0.025", linewidth=1.0, edgecolor=COLORS["rare"], facecolor="#F3F8F1")
ax.add_patch(module)
ax.text(0.80, 0.745, "prototype_gate_marker", ha="center", va="center", fontsize=8.2, weight="bold", color=COLORS["ink"])
rounded_box(ax, (0.725, 0.61), 0.15, 0.08, "Rare prototype", COLORS["module"], fontsize=7.0)
rounded_box(ax, (0.725, 0.48), 0.15, 0.08, "Prototype gate", COLORS["module"], fontsize=7.0)
rounded_box(ax, (0.725, 0.35), 0.15, 0.08, "Marker verification", COLORS["module"], fontsize=7.0)
mini_marker(ax, 0.748, 0.285, 0.105, 0.045)
arrow(ax, 0.80, 0.60, 0.80, 0.565, color=COLORS["rare"], lw=1.0)
arrow(ax, 0.80, 0.47, 0.80, 0.435, color=COLORS["rare"], lw=1.0)

arrow(ax, 0.915, 0.52, 0.945, 0.52)

rounded_box(ax, (0.945, 0.36), 0.04, 0.32, "Refined\nrare-cell\npredictions", "#E5F1E2", edge=COLORS["rare"], weight="bold", fontsize=6.1)
ax.text(0.965, 0.29, "Low false rescue", ha="center", va="center", fontsize=6.6, color=COLORS["rare"], weight="bold")

for x, label, face in [(0.19, "Train-only\nreference", COLORS["reference"]), (0.49, "Validation-selected\ngate", "#E6E1F2"), (0.76, "Test-only final\nevaluation", "#ECEFF3")]:
    rounded_box(ax, (x, 0.08), 0.17, 0.075, label, face, lw=0.6, radius=0.012, fontsize=6.6)
arrow(ax, 0.36, 0.118, 0.48, 0.118, color=COLORS["line"], lw=0.9)
arrow(ax, 0.66, 0.118, 0.75, 0.118, color=COLORS["line"], lw=0.9)

curved_arrow(ax, (0.275, 0.18), (0.752, 0.65), (0.36, 0.25), (0.59, 0.78), color=COLORS["line"], lw=0.9)
curved_arrow(ax, (0.57, 0.18), (0.80, 0.52), (0.64, 0.23), (0.76, 0.40), color=COLORS["line"], lw=0.9)

fig.savefig(f"{OUT}.svg", bbox_inches="tight")
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)
