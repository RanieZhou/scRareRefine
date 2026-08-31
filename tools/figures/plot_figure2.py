"""Generate the final Figure 2 for the scRareRefine manuscript.

The figure is deliberately assembled from the paper-ready benchmark files:

* ``comparison_summary.csv`` supplies the completed baseline runs.
* ``adaptive_separability_gate/v1/{human,mouse}_run_level.csv`` supplies the
  frozen adaptive scRareRefine runs and the paired scANVI predictions.

The old scRareRefine row in ``comparison_summary.csv`` is excluded because it
predates the adaptive mainline.  No metric values are entered by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
OUTPUT_STEM = "figure2"

DATASETS = [
    "immune_dc",
    "mouse_lung_tms_10x",
    "mouse_pancreas_tms_10x",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
]
DISPLAY_NAMES = {
    "immune_dc": "Immune DC",
    "mouse_lung_tms_10x": "Mouse lung",
    "mouse_pancreas_tms_10x": "Mouse pancreas",
    "pancreas_baron": "Baron pancreas",
    "pancreas_integrated": "Integrated pancreas",
    "tabula_lung_endo": "Lung endothelium",
    "tabula_sapiens_stomach": "Stomach",
    "tabula_small_intestine": "Small intestine",
}
BUDGETS = ["0.01", "0.05", "0.10"]
BUDGET_LABELS = {"0.01": "1% rare-label budget", "0.05": "5% rare-label budget", "0.10": "10% rare-label budget"}
METHODS = [
    "scRareRefine",
    "scANVI",
    "kNN",
    "scBalance",
    "ProtoCloud",
    "CellTypist",
    "scCAD",
    "TOSICA",
]
METHOD_DISPLAY_NAMES = {method: method for method in METHODS}

# Muted, publication-oriented palettes.  Panel A uses one shared, softened
# YlGnBu sequential map and one nonlinear normalization for all three budgets.
# The upper end is deliberately truncated before the darkest navy colors so
# high-F1 cells retain visible tonal differences rather than collapsing into a
# nearly black block.  This changes only the display palette, not the values or
# the shared 0--1 colorbar limits.
_YLGNBU = mpl.colormaps["YlGnBu"]
F1_CMAP = LinearSegmentedColormap.from_list(
    "soft_ylgnbu_f1",
    _YLGNBU(np.linspace(0.02, 0.84, 256)),
)
F1_NORM = PowerNorm(gamma=2.0, vmin=0.0, vmax=1.0)
F1_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "soft_delta",
    ["#5D82A8", "#A9C8DC", "#F8F7F2", "#F1C1B4", "#C9675F"],
)
BOX_COLORS = [
    "#D8736B",
    "#E4A04D",
    "#D6B361",
    "#AFC274",
    "#6EB77D",
    "#55B0A5",
    "#5CA5C3",
    "#858ABC",
    "#AD7EAA",
]
BOX_EDGE_COLORS = [
    "#A9514B",
    "#B5772D",
    "#A48435",
    "#758D40",
    "#46845A",
    "#337F77",
    "#3E7892",
    "#62669A",
    "#805C7E",
]


def normalize_budget(value: object) -> str:
    """Use one budget key for CSV values read as strings or floats."""

    text = str(value).strip()
    if text.lower() == "all":
        return "all"
    return f"{float(text):.2f}"


def load_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one long F1 table and the paired scRareRefine/scANVI table."""

    comparison_path = ROOT / "results" / "comparison" / "comparison_summary.csv"
    adaptive_paths = [
        ROOT / "results" / "adaptive_separability_gate" / "v1" / "human_run_level.csv",
        ROOT / "results" / "adaptive_separability_gate" / "v1" / "mouse_run_level.csv",
    ]

    comparison = pd.read_csv(comparison_path)
    adaptive = pd.concat((pd.read_csv(path) for path in adaptive_paths), ignore_index=True)
    comparison["budget"] = comparison["rare_train_size"].map(normalize_budget)
    adaptive["budget"] = adaptive["rare_train_size"].map(normalize_budget)

    scarce = comparison[comparison["budget"].isin(BUDGETS)].copy()
    adaptive = adaptive[
        adaptive["budget"].isin(BUDGETS)
        & (adaptive["variant"] == "adaptive_sep_gate")
        & (adaptive["status"] == "success")
    ].copy()

    # Use the adaptive file for both scRareRefine and scANVI so Panel C is
    # exactly paired at the dataset/seed/budget level.  The old scRareRefine
    # rows in comparison_summary.csv are intentionally not used.
    ours = adaptive[["dataset", "seed", "budget", "rare_f1"]].copy()
    ours["method"] = "scRareRefine"
    ours = ours.rename(columns={"rare_f1": "f1"})

    scanvi = adaptive[["dataset", "seed", "budget", "baseline_rare_f1"]].copy()
    scanvi["method"] = "scANVI"
    scanvi = scanvi.rename(columns={"baseline_rare_f1": "f1"})

    baseline_methods = [method for method in METHODS if method not in {"scRareRefine", "scANVI"}]
    baseline = scarce[scarce["method"].isin(baseline_methods)][
        ["dataset", "seed", "budget", "method", "rare_f1"]
    ].rename(columns={"rare_f1": "f1"})

    f1 = pd.concat([ours, scanvi, baseline], ignore_index=True)
    f1["f1"] = pd.to_numeric(f1["f1"], errors="raise")

    expected_keys = {"dataset", "seed", "budget", "method"}
    if set(f1.columns).intersection(expected_keys) != expected_keys:
        raise AssertionError("Figure 2 data is missing a required key column")
    counts = f1.groupby(["dataset", "budget", "method"])["seed"].nunique()
    if not (counts == 3).all():
        bad = counts[counts != 3]
        raise AssertionError(f"Expected three seeds per cell; bad cells:\n{bad}")
    method_counts = f1.groupby("method").size()
    if not (method_counts == 72).all():
        raise AssertionError(f"Expected 72 observations per method; got:\n{method_counts}")
    if set(f1["method"].unique()) != set(METHODS):
        raise AssertionError("The plotted method set does not match the requested order")

    paired = ours.merge(
        scanvi,
        on=["dataset", "seed", "budget"],
        suffixes=("_ours", "_scanvi"),
    )
    paired["delta"] = paired["f1_ours"] - paired["f1_scanvi"]
    if len(paired) != 72:
        raise AssertionError(f"Expected 72 paired scarce-budget units, got {len(paired)}")

    # Confirm that the scANVI baseline used for pairing agrees with the
    # completed baseline table.  A mismatch would indicate a protocol drift.
    comparison_scanvi = scarce[scarce["method"] == "scANVI"][
        ["dataset", "seed", "budget", "rare_f1"]
    ].rename(columns={"rare_f1": "f1_comparison"})
    check = scanvi.merge(comparison_scanvi, on=["dataset", "seed", "budget"])
    max_diff = float(np.abs(check["f1"] - check["f1_comparison"]).max())
    if max_diff > 1e-4:
        raise AssertionError(f"scANVI baseline differs between sources by {max_diff}")
    if max_diff > 1e-9:
        print(
            "note: comparison_summary.csv rounds some scANVI F1 values; "
            f"paired adaptive values differ by at most {max_diff:.2e}"
        )

    return f1, paired


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#222222",
    )


def draw_panel_a(axes: list[plt.Axes], f1: pd.DataFrame, cax: plt.Axes) -> None:
    matrix_by_budget = {}
    for budget in BUDGETS:
        summary = (
            f1[f1["budget"] == budget]
            .groupby(["dataset", "method"], as_index=False)["f1"]
            .mean()
        )
        matrix = (
            summary.pivot(index="dataset", columns="method", values="f1")
            .reindex(index=DATASETS, columns=METHODS)
        )
        if matrix.isna().any().any():
            raise AssertionError(f"Panel A has missing values at budget {budget}")
        matrix_by_budget[budget] = matrix

    add_panel_label(axes[0], "A")
    image = None
    for index, budget in enumerate(BUDGETS):
        ax = axes[index]
        matrix = matrix_by_budget[budget]
        image = ax.imshow(
            matrix.to_numpy(),
            cmap=F1_CMAP,
            norm=F1_NORM,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_xticklabels(
            [METHOD_DISPLAY_NAMES[method] for method in METHODS],
            rotation=42,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(len(DATASETS)))
        ax.set_yticklabels([DISPLAY_NAMES[name] for name in DATASETS])
        if index > 0:
            ax.tick_params(axis="y", labelleft=False, left=False)
        ax.tick_params(axis="x", pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(METHODS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
        ax.tick_params(which="minor", bottom=False, left=False)

        row_max = matrix.max(axis=1).to_numpy()
        for row in range(len(DATASETS)):
            for col in range(len(METHODS)):
                value = float(matrix.iloc[row, col])
                is_best = np.isclose(value, row_max[row], atol=5e-9)
                if is_best:
                    ax.text(
                        col,
                        row,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=8.6,
                        fontweight="bold",
                        color="#111111",
                        zorder=4,
                    )
        ax.text(
            0.5,
            1.075,
            BUDGET_LABELS[budget],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#333333",
        )

    if image is None:
        raise AssertionError("Panel A did not render an image")
    colorbar = plt.colorbar(image, cax=cax)
    colorbar.set_ticks(F1_TICKS)
    colorbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "0.9", "1.0"])
    colorbar.set_label("Rare-cell F1", fontsize=7.8, labelpad=5)
    colorbar.ax.tick_params(labelsize=6.8, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.5)


def draw_panel_b(ax: plt.Axes, f1: pd.DataFrame) -> None:
    values = [f1.loc[f1["method"] == method, "f1"].to_numpy() for method in METHODS]
    positions = 1.2 * np.arange(1, len(METHODS) + 1)
    box_width = 0.46
    box = ax.boxplot(
        values,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.0},
        whiskerprops={"color": "#5c6670", "linewidth": 0.7},
        capprops={"color": "#5c6670", "linewidth": 0.7},
        boxprops={"linewidth": 0.8},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(BOX_COLORS[index])
        patch.set_edgecolor(BOX_EDGE_COLORS[index])
        patch.set_linewidth(0.85)

    for index, vals in enumerate(values):
        median_value = float(np.median(vals))
        label_offset = (2, 5) if median_value < 0.05 else (2, 0)
        label_va = "bottom" if median_value < 0.05 else "center"
        label_x = positions[index] + box_width / 2 + 0.05
        ax.annotate(
            f"{median_value:.2f}",
            xy=(label_x, median_value),
            xytext=label_offset,
            textcoords="offset points",
            ha="left",
            va=label_va,
            fontsize=6.25,
            color=BOX_EDGE_COLORS[index],
            zorder=4,
        )

    ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.78)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rare-cell F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [METHOD_DISPLAY_NAMES[method] for method in METHODS],
        rotation=42,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "B")


def draw_panel_c(ax: plt.Axes, cax: plt.Axes, paired: pd.DataFrame) -> tuple[int, int, int, float]:
    summary = (
        paired.groupby(["dataset", "budget"], as_index=False)["delta"]
        .mean()
        .pivot(index="dataset", columns="budget", values="delta")
        .reindex(index=DATASETS, columns=BUDGETS)
    )
    if summary.isna().any().any():
        raise AssertionError("Panel C has missing paired differences")
    limit = max(0.05, float(np.nanmax(np.abs(summary.to_numpy()))))
    limit = float(np.ceil(limit * 10.0) / 10.0)
    image = ax.imshow(
        summary.to_numpy(),
        cmap=DELTA_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BUDGETS)))
    ax.set_xticklabels(["1%", "5%", "10%"])
    ax.set_xlabel("Rare-label budget")
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels([DISPLAY_NAMES[name] for name in DATASETS])
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6.25, pad=1.5)
    ax.set_xticks(np.arange(-0.5, len(BUDGETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(len(DATASETS)):
        for col in range(len(BUDGETS)):
            value = float(summary.iloc[row, col])
            text = "0.00" if abs(value) < 0.005 else f"{value:+.2f}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="normal",
                color="#202020",
            )
    add_panel_label(ax, "C")
    colorbar = plt.colorbar(image, cax=cax)
    colorbar.set_label("Δ Rare-cell F1\n(scRareRefine − scANVI)", fontsize=7.2, labelpad=5)
    colorbar.ax.tick_params(labelsize=6.7, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.5)

    eps = 1e-12
    wins = int((paired["delta"] > eps).sum())
    ties = int(np.isclose(paired["delta"], 0.0, atol=eps).sum())
    losses = int((paired["delta"] < -eps).sum())
    return wins, ties, losses, float(paired["delta"].mean())


def main() -> None:
    set_publication_style()
    f1, paired = load_benchmark()

    figure = plt.figure(figsize=(7.8, 7.5), facecolor="white")
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.35, 1.0],
        left=0.075,
        right=0.965,
        bottom=0.085,
        top=0.965,
        hspace=0.42,
    )
    top_grid = outer_grid[0].subgridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.14],
        wspace=0.52,
    )
    bottom_grid = outer_grid[1].subgridspec(
        1,
        4,
        width_ratios=[1.70, 0.55, 1.0, 0.14],
        wspace=0.18,
    )
    axes_a = [
        figure.add_subplot(top_grid[0, 0]),
        figure.add_subplot(top_grid[0, 1]),
        figure.add_subplot(top_grid[0, 2]),
    ]
    cax_a = figure.add_subplot(top_grid[0, 3])
    ax_b = figure.add_subplot(bottom_grid[0, 0])
    ax_c = figure.add_subplot(bottom_grid[0, 2])
    cax_c = figure.add_subplot(bottom_grid[0, 3])

    draw_panel_a(axes_a, f1, cax_a)
    draw_panel_b(ax_b, f1)
    wins, ties, losses, mean_delta = draw_panel_c(ax_c, cax_c, paired)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in {
        "pdf": {"format": "pdf"},
        "svg": {"format": "svg"},
        "png": {"format": "png", "dpi": 300},
    }.items():
        path = OUT_DIR / f"{OUTPUT_STEM}.{suffix}"
        figure.savefig(path, bbox_inches="tight", pad_inches=0.04, **options)
        print(f"saved {path} ({path.stat().st_size} bytes)")

    print(
        f"paired scarce-budget result: {wins} wins / {ties} ties / {losses} losses; "
        f"mean delta F1 = {mean_delta:+.4f}"
    )
    print(f"methods={len(METHODS)}, datasets={len(DATASETS)}, budgets={len(BUDGETS)}, seeds=3")
    plt.close(figure)


if __name__ == "__main__":
    main()
