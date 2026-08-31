"""Plot the held-out biological plausibility case-study figure.

The script reads all plotted values from ``results/biological_case_study/v1``.
By default it writes the primary Immune DC--ASDC case to ``figure4.*``; the
secondary Baron pancreas--gamma case can be written to a separate stem for a
 supplementary figure.  The primary Immune DC--ASDC panel uses an independent
expression UMAP with a local magnified inset; the scANVI latent space is not
used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "results" / "biological_case_study" / "v1"
OUT_DIR = ROOT / "paper" / "figures"

CASE_DISPLAY = {
    "immune_dc": "Immune DC - ASDC",
    "pancreas_baron": "Baron pancreas - gamma",
}
BASE_GROUPS = ["Baseline TP", "Rescued TP", "Unrescued FN"]
GROUP_COLORS = {
    "Baseline TP": "#536F91",
    "Rescued TP": "#C96F68",
    "Unrescued FN": "#C99552",
    "Rescue FP": "#A65350",
    "Competitor": "#9AA8B6",
}
GROUP_SHORT = {
    "Baseline TP": "Baseline",
    "Rescued TP": "Rescued",
    "Unrescued FN": "Unrescued",
    "Rescue FP": "Rescue FP",
    "Competitor": "Competitor",
}
EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "soft_expression",
    ["#F7FBFF", "#E8F0F8", "#CADBEF", "#9DB9DA", "#6E91B4"],
)

# Background true-cell-type colours and foreground rescue markers used in the
# primary Immune DC--ASDC expression UMAP.
CELL_TYPE_COLORS = {
    "pDC": "#6A3D9A",
    "HLA-DRhi cDC2": "#2878B5",
    "CD14+ cDC2": "#E07A1F",
    "ISG+ cDC2": "#2A9D55",
    "cDC1": "#A6761D",
    "ASDC": "#D14A9B",
}
RESCUE_COLORS = {
    "Baseline TP": "#253D73",
    "Rescued TP": "#D04E3E",
    "Unrescued FN": "#007C91",
    "Rescue FP": "#2A9D8F",
}


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#20262B",
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(SOURCE_DIR / "test_cell_groups.csv")
    scores = pd.read_csv(SOURCE_DIR / "group_scores.csv")
    dotplot = pd.read_csv(SOURCE_DIR / "marker_dotplot.csv")
    summary = pd.read_csv(SOURCE_DIR / "case_summary.csv")
    required = {"immune_dc", "pancreas_baron"}
    for table, name in ((groups, "test-cell groups"), (scores, "group scores"), (dotplot, "marker dotplot")):
        if not required.issubset(set(table["dataset"].unique())):
            raise AssertionError(f"{name} does not contain both frozen case-study datasets")
    return groups, scores, dotplot, summary


def _available_groups(scores: pd.DataFrame, dataset: str, min_fp_n: int = 5) -> list[str]:
    subset = scores[scores["dataset"] == dataset]
    groups = BASE_GROUPS.copy()
    fp_n = int((subset["group"] == "Rescue FP").sum())
    if fp_n >= min_fp_n:
        groups.append("Rescue FP")
    groups.append("Competitor")
    return [group for group in groups if group in set(subset["group"])]


def _counts(scores: pd.DataFrame, dataset: str, groups: list[str]) -> dict[str, int]:
    subset = scores[scores["dataset"] == dataset]
    return {group: int((subset["group"] == group).sum()) for group in groups}


def _tick_labels(groups: list[str]) -> list[str]:
    """Return concise group labels for the module-score panels."""

    return [GROUP_SHORT[group] for group in groups]


def draw_legacy_embedding(ax: plt.Axes, groups: pd.DataFrame, dataset: str, label: str) -> None:
    subset = groups[groups["dataset"] == dataset].copy()
    ax.scatter(
        subset["umap1"],
        subset["umap2"],
        s=2.0,
        c="#D9DEE3",
        alpha=0.22,
        linewidths=0,
        zorder=1,
    )

    competitor = subset[subset["is_competitor"].astype(bool)]
    ax.scatter(
        competitor["umap1"],
        competitor["umap2"],
        s=4.0,
        c=GROUP_COLORS["Competitor"],
        alpha=0.27,
        linewidths=0,
        zorder=2,
    )
    styles = {
        "Baseline TP": dict(marker="o", c=GROUP_COLORS["Baseline TP"], s=13, alpha=0.78, edgecolors="white", linewidths=0.25),
        "Rescued TP": dict(marker="o", c=GROUP_COLORS["Rescued TP"], s=20, alpha=0.92, edgecolors="white", linewidths=0.3),
        "Unrescued FN": dict(marker="o", facecolors="none", edgecolors=GROUP_COLORS["Unrescued FN"], s=24, alpha=0.95, linewidths=0.85),
        "Rescue FP": dict(marker="x", c=GROUP_COLORS["Rescue FP"], s=34, alpha=0.95, linewidths=1.2),
    }
    for group, style in styles.items():
        selected = subset[subset["primary_group"] == group]
        if selected.empty:
            continue
        ax.scatter(selected["umap1"], selected["umap2"], zorder=4, **style)

    ax.set_xlabel("Expression t-SNE 1", labelpad=2)
    ax.set_ylabel("Expression t-SNE 2", labelpad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#9AA3AC")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=GROUP_COLORS["Baseline TP"], markersize=3.3, label="Baseline TP"),
        plt.Line2D([], [], marker="o", linestyle="", color=GROUP_COLORS["Rescued TP"], markersize=3.8, label="Rescued TP"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=GROUP_COLORS["Unrescued FN"], markersize=3.8, label="Unrescued FN"),
        plt.Line2D([], [], marker="x", linestyle="", color=GROUP_COLORS["Rescue FP"], markersize=4.2, label="Rescue FP"),
        plt.Line2D([], [], marker="o", linestyle="", color=GROUP_COLORS["Competitor"], alpha=0.55, markersize=3.3, label="Competitor cells"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=5.2,
        frameon=True,
        framealpha=0.88,
        borderpad=0.3,
        handletextpad=0.3,
        labelspacing=0.18,
    )
    add_panel_label(ax, label)


def _load_primary_umap(groups: pd.DataFrame, dataset: str) -> pd.DataFrame:
    subset = groups[groups["dataset"] == dataset].copy().reset_index(drop=True)
    # Use the frozen coordinates exported with the case-study table so that
    # the manuscript figure exactly matches the reviewed UMAP preview.
    if {"umap1", "umap2"}.issubset(subset.columns):
        subset["_x"] = subset["umap1"].astype(float)
        subset["_y"] = subset["umap2"].astype(float)
        return subset

    # Backward-compatible fallback for older cached case-study tables.
    cache_path = SOURCE_DIR / "embedding_immune_dc_umap.npz"
    cache = np.load(cache_path, allow_pickle=True)
    cached_ids = np.asarray(cache["cell_id"]).astype(str)
    cell_ids = subset["cell_id"].astype(str).to_numpy()
    if not np.array_equal(cached_ids, cell_ids):
        raise RuntimeError("cached UMAP cell order does not match the Immune DC group table")
    subset["_x"] = np.asarray(cache["xy"][:, 0], dtype=float)
    subset["_y"] = np.asarray(cache["xy"][:, 1], dtype=float)
    return subset


def _draw_umap_background(ax: plt.Axes, subset: pd.DataFrame, point_size: float, alpha: float) -> None:
    for cell_type, colour in CELL_TYPE_COLORS.items():
        selected = subset[subset["true_label"] == cell_type]
        if selected.empty:
            continue
        ax.scatter(
            selected["_x"],
            selected["_y"],
            s=point_size * (1.8 if cell_type == "ASDC" else 1.0),
            c=colour,
            alpha=0.82 if cell_type == "ASDC" else alpha,
            linewidths=0,
            zorder=1,
        )


def _draw_umap_rescue_groups(ax: plt.Axes, subset: pd.DataFrame, scale: float = 1.0) -> None:
    styles = {
        "Baseline TP": dict(marker="o", c=RESCUE_COLORS["Baseline TP"], s=8.5 * scale, alpha=0.95, edgecolors="white", linewidths=0.25),
        "Rescued TP": dict(marker="o", c=RESCUE_COLORS["Rescued TP"], s=12.5 * scale, alpha=0.98, edgecolors="white", linewidths=0.3),
        "Unrescued FN": dict(marker="o", facecolors=CELL_TYPE_COLORS["ASDC"], edgecolors=RESCUE_COLORS["Unrescued FN"], s=16.0 * scale, alpha=0.98, linewidths=0.8),
        "Rescue FP": dict(marker="o", facecolors="white", edgecolors=RESCUE_COLORS["Rescue FP"], s=20.0 * scale, alpha=1.0, linewidths=1.15),
    }
    for group, style in styles.items():
        selected = subset[subset["primary_group"] == group]
        if selected.empty:
            continue
        ax.scatter(selected["_x"], selected["_y"], zorder=5, **style)


def _umap_rescue_legend() -> list[plt.Line2D]:
    return [
        plt.Line2D([], [], marker="o", linestyle="", color=RESCUE_COLORS["Baseline TP"], markersize=3.0, label="Baseline TP"),
        plt.Line2D([], [], marker="o", linestyle="", color=RESCUE_COLORS["Rescued TP"], markersize=3.3, label="Rescued TP"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=CELL_TYPE_COLORS["ASDC"], markeredgecolor=RESCUE_COLORS["Unrescued FN"], markersize=3.5, label="Unrescued FN"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=RESCUE_COLORS["Rescue FP"], markeredgewidth=1.0, markersize=3.8, label="Rescue FP"),
    ]


def _umap_cell_type_legend() -> list[plt.Line2D]:
    return [
        plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=2.8, label=cell_type)
        for cell_type, colour in CELL_TYPE_COLORS.items()
    ]


def _umap_focus(subset: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return the rescue-relevant focus box with visible clearance."""

    focus_cells = subset[subset["primary_group"].isin(["Baseline TP", "Rescued TP", "Rescue FP"])]
    margin_x = 1.05
    margin_y = 1.05
    return (
        float(focus_cells["_x"].min() - margin_x),
        float(focus_cells["_x"].max() + margin_x),
        float(focus_cells["_y"].min() - margin_y),
        float(focus_cells["_y"].max() + margin_y),
    )


def _format_tsne_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("t-SNE 1", labelpad=1.5, fontsize=6.8)
    ax.set_ylabel("t-SNE 2", labelpad=1.5, fontsize=6.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#8F99A3")


def draw_umap_truth(ax: plt.Axes, subset: pd.DataFrame, label: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Panel A: independent expression UMAP coloured by held-out truth."""

    _draw_umap_background(ax, subset, point_size=1.55, alpha=0.38)
    _format_tsne_axes(ax, xlim, ylim)
    ax.legend(
        handles=_umap_cell_type_legend(),
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        fontsize=4.2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.55,
        labelspacing=0.15,
    )
    add_panel_label(ax, label)


def draw_umap_rescue(
    ax: plt.Axes,
    subset: pd.DataFrame,
    label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    focus: tuple[float, float, float, float],
) -> None:
    """Panel B: rescue outcomes with a compact focus inset."""

    ax.scatter(
        subset["_x"],
        subset["_y"],
        s=1.45,
        color="#D7DDE2",
        alpha=0.38,
        linewidths=0,
        zorder=1,
    )
    _draw_umap_rescue_groups(ax, subset, scale=0.82)
    _format_tsne_axes(ax, xlim, ylim)

    focus_x0, focus_x1, focus_y0, focus_y1 = focus
    ax.add_patch(
        Rectangle(
            (focus_x0, focus_y0),
            focus_x1 - focus_x0,
            focus_y1 - focus_y0,
            fill=False,
            edgecolor="#263238",
            linewidth=0.75,
            linestyle="--",
            zorder=8,
        )
    )

    inset = ax.inset_axes([0.49, 0.06, 0.46, 0.38], zorder=10)
    inset.set_facecolor("white")
    zoom = subset[
        subset["_x"].between(focus_x0, focus_x1)
        & subset["_y"].between(focus_y0, focus_y1)
    ]
    _draw_umap_background(inset, zoom, point_size=4.5, alpha=0.45)
    _draw_umap_rescue_groups(inset, zoom, scale=1.05)
    inset.set_xlim(focus_x0, focus_x1)
    inset.set_ylim(focus_y0, focus_y1)
    inset.set_xlabel("t-SNE 1", labelpad=1, fontsize=5.2)
    inset.set_ylabel("t-SNE 2", labelpad=1, fontsize=5.2)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#5D6872")

    ax.figure.add_artist(
        ConnectionPatch(
            xyA=(focus_x1, focus_y0),
            coordsA=ax.transData,
            xyB=(0.0, 1.0),
            coordsB=inset.transAxes,
            color="#59636D",
            linewidth=0.55,
            linestyle="-",
        )
    )
    ax.legend(
        handles=_umap_rescue_legend(),
        loc="upper right",
        fontsize=4.6,
        frameon=True,
        framealpha=0.92,
        edgecolor="#C7CDD3",
        borderpad=0.28,
        handletextpad=0.28,
        labelspacing=0.16,
    )
    add_panel_label(ax, label)


def _display_markers(dotplot: pd.DataFrame, dataset: str) -> tuple[list[str], list[str]]:
    subset = dotplot[dotplot["dataset"] == dataset]
    target = subset.loc[subset["marker_family"] == "Target rare markers", "gene"].drop_duplicates().tolist()
    competitor = subset.loc[subset["marker_family"] == "Competing-type markers", "gene"].drop_duplicates().tolist()
    return target[:4], competitor[:2]


def draw_dotplot(
    ax: plt.Axes,
    dotplot: pd.DataFrame,
    dataset: str,
    groups: list[str],
    label: str,
    expression_norm: Normalize,
) -> ScalarMappable:
    target_markers, competitor_markers = _display_markers(dotplot, dataset)
    marker_rows = target_markers + competitor_markers
    subset = dotplot[
        (dotplot["dataset"] == dataset)
        & dotplot["group"].isin(groups)
        & dotplot["gene"].isin(marker_rows)
    ].copy()
    x_lookup = {group: index for index, group in enumerate(groups)}
    y_lookup = {gene: index for index, gene in enumerate(marker_rows)}
    for row in subset.itertuples(index=False):
        ax.scatter(
            x_lookup[row.group],
            y_lookup[row.gene],
            s=17.0 + 112.0 * float(row.pct_expressed),
            c=[float(row.mean_expression)],
            cmap=EXPRESSION_CMAP,
            norm=expression_norm,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    ax.axhline(len(target_markers) - 0.5, color="#C9D0D7", linewidth=0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_SHORT[group] for group in groups], rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(marker_rows)))
    ax.set_yticklabels(marker_rows)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(len(marker_rows) - 0.45, -0.45)
    # The group names on the x-axis are self-explanatory; omit the redundant
    # axis title to keep the biological case-study panel compact.
    ax.set_xlabel("")
    ax.set_ylabel("")
    target_mid = (len(target_markers) - 1) / 2.0
    competitor_mid = len(target_markers) + (len(competitor_markers) - 1) / 2.0
    for y, text in ((target_mid, "Target markers"), (competitor_mid, "Competitor markers")):
        ax.text(
            -0.27,
            y,
            text,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            rotation=90,
            fontsize=5.2,
            color="#7F8A94",
        )
    ax.grid(axis="x", color="#EEF1F4", linewidth=0.4)
    ax.set_axisbelow(True)
    add_panel_label(ax, label)
    return ScalarMappable(norm=expression_norm, cmap=EXPRESSION_CMAP)


def _boxplot_with_jitter(
    ax: plt.Axes,
    values: list[np.ndarray],
    groups: list[str],
    counts: dict[str, int],
    ylabel: str,
) -> None:
    positions = np.arange(len(groups), dtype=float)
    valid = [value if len(value) else np.asarray([np.nan]) for value in values]
    artists = ax.boxplot(
        valid,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "#263238", "linewidth": 0.85},
        whiskerprops={"color": "#59636C", "linewidth": 0.55},
        capprops={"color": "#59636C", "linewidth": 0.55},
        boxprops={"linewidth": 0.6},
    )
    for patch, group in zip(artists["boxes"], groups):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_edgecolor("#59636C")
        patch.set_alpha(0.64)

    ax.set_xticks(positions)
    ax.set_xticklabels(_tick_labels(groups), rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel, labelpad=2)
    ax.margins(x=0.03)


def draw_module_scores(
    ax_target: plt.Axes,
    ax_competitor: plt.Axes,
    scores: pd.DataFrame,
    dataset: str,
    groups: list[str],
    counts: dict[str, int],
) -> None:
    subset = scores[scores["dataset"] == dataset]
    target_values = [subset.loc[subset.group == group, "rare_marker_score"].to_numpy(float) for group in groups]
    competitor_values = [subset.loc[subset.group == group, "competitor_marker_score"].to_numpy(float) for group in groups]
    _boxplot_with_jitter(ax_target, target_values, groups, counts, "Target-marker score")
    _boxplot_with_jitter(ax_competitor, competitor_values, groups, counts, "Competitor-marker score")
    ax_target.set_title("Target markers", fontsize=7.0, pad=2)
    ax_competitor.set_title("Competitor markers", fontsize=7.0, pad=2)
    ax_target.set_xlabel("")
    ax_competitor.set_xlabel("")
    ax_target.tick_params(axis="x", labelsize=5.4)
    ax_competitor.tick_params(axis="x", labelsize=5.4)
    add_panel_label(ax_target, "D")


def draw_similarity(ax: plt.Axes, scores: pd.DataFrame, dataset: str, groups: list[str], counts: dict[str, int]) -> None:
    subset = scores[scores["dataset"] == dataset]
    values = [subset.loc[subset.group == group, "delta_similarity"].to_numpy(float) for group in groups]
    _boxplot_with_jitter(ax, values, groups, counts, r"$\Delta$ similarity (rare - competitor)")
    ax.axhline(0, color="#68737D", linewidth=0.55, linestyle="--")
    ax.set_xlabel("")
    add_panel_label(ax, "E")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CASE_DISPLAY), default="immune_dc")
    parser.add_argument("--figure-stem", default="figure4", help="Output filename stem in paper/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_publication_style()
    groups_table, scores, dotplot, summary = load_inputs()
    if args.dataset not in set(summary["dataset"]):
        raise ValueError(f"dataset {args.dataset!r} is not present in the frozen case-study outputs")

    display_groups = _available_groups(scores, args.dataset)
    counts = _counts(scores, args.dataset, display_groups)
    display_dotplot_groups = display_groups
    dot_values = dotplot[
        (dotplot["dataset"] == args.dataset)
        & dotplot["group"].isin(display_dotplot_groups)
    ]
    expression_norm = Normalize(vmin=0.0, vmax=max(float(dot_values["mean_expression"].max()), 1e-6))

    if args.dataset != "immune_dc":
        raise ValueError("The five-panel primary Figure 4 layout is defined for the Immune DC--ASDC case")

    umap_subset = _load_primary_umap(groups_table, args.dataset)
    x_span = float(umap_subset["_x"].max() - umap_subset["_x"].min())
    y_span = float(umap_subset["_y"].max() - umap_subset["_y"].min())
    x_pad = 0.035 * x_span
    y_pad = 0.035 * y_span
    umap_xlim = (float(umap_subset["_x"].min() - x_pad), float(umap_subset["_x"].max() + x_pad))
    umap_ylim = (float(umap_subset["_y"].min() - y_pad), float(umap_subset["_y"].max() + y_pad))
    focus = _umap_focus(umap_subset)

    # Five logical parts: two complementary UMAP views (A--B), the marker
    # dotplot (C), the two module-score summaries (D), and similarity (E).
    fig = plt.figure(figsize=(7.35, 8.15), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.18, 0.92, 0.78],
        width_ratios=[1.0, 1.0],
        hspace=0.32,
        wspace=0.34,
        left=0.085,
        right=0.945,
        bottom=0.075,
        top=0.965,
    )
    umap_grid = grid[0, :].subgridspec(1, 2, wspace=0.16)
    ax_a = fig.add_subplot(umap_grid[0, 0])
    ax_b = fig.add_subplot(umap_grid[0, 1], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(grid[1, 0])
    module_grid = grid[1, 1].subgridspec(1, 2, wspace=0.25)
    ax_d1 = fig.add_subplot(module_grid[0, 0])
    ax_d2 = fig.add_subplot(module_grid[0, 1])
    ax_e = fig.add_subplot(grid[2, :])

    draw_umap_truth(ax_a, umap_subset, "A", umap_xlim, umap_ylim)
    draw_umap_rescue(ax_b, umap_subset, "B", umap_xlim, umap_ylim, focus)
    color_mappable = draw_dotplot(ax_c, dotplot, args.dataset, display_dotplot_groups, "C", expression_norm)
    draw_module_scores(ax_d1, ax_d2, scores, args.dataset, display_groups, counts)
    draw_similarity(ax_e, scores, args.dataset, display_groups, counts)

    colorbar = fig.colorbar(color_mappable, ax=ax_c, fraction=0.045, pad=0.035, aspect=26)
    colorbar.set_label("Mean log-normalized expression", fontsize=6.7, labelpad=3)
    colorbar.ax.tick_params(labelsize=6.0, width=0.4, length=1.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("pdf", {"bbox_inches": "tight"}),
        ("svg", {"bbox_inches": "tight"}),
        ("png", {"dpi": 600, "bbox_inches": "tight"}),
    ):
        path = OUT_DIR / f"{args.figure_stem}.{suffix}"
        fig.savefig(path, **kwargs)
        print(f"[saved] {path}")
    print(f"[case] {CASE_DISPLAY[args.dataset]}; groups={display_groups}; counts={counts}")
    plt.close(fig)


if __name__ == "__main__":
    main()
