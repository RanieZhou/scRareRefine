"""Plot the seed-42 batch-vs-cell split sensitivity summary.

The figure is descriptive: it reads the frozen 48-row summary and visualizes
the label-scarce region (rare_train_size in 0.01, 0.05, 0.10). It does not
recompute predictions or alter any experiment artifact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "split_sensitivity" / "cell_stratified_seed42_summary.csv"
OUTPUT_DIR = ROOT / "results" / "split_sensitivity"
OUTPUT_STEM = "split_sensitivity_seed42_summary"

SPLITS = ("batch_heldout", "cell_stratified")
BUDGETS = ("0.01", "0.05", "0.10")
DATASETS = (
    "immune_dc",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
)
DATASET_LABELS = (
    "Immune DC",
    "Baron pancreas",
    "Integrated pancreas",
    "Lung endothelium",
    "Stomach",
    "Small intestine",
)
BASELINE_COLOR = "#6F6F6F"
REFINED_COLOR = "#1B7F4B"


def _load_data(path: Path = INPUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path, dtype={"rare_train_size": str})
    required = {
        "dataset",
        "split_mode",
        "seed",
        "rare_train_size",
        "status",
        "scANVI_f1",
        "scRareRefine_f1",
        "delta_f1",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(data) != 48 or not data["status"].eq("ok").all():
        raise ValueError("Expected a closed 48-row, all-ok split-sensitivity ledger")
    if set(data["dataset"]) != set(DATASETS) or set(data["split_mode"]) != set(SPLITS):
        raise ValueError("Dataset or split grid differs from the prespecified sensitivity analysis")
    if set(data["seed"].astype(int)) != {42}:
        raise ValueError("This figure is restricted to the prespecified seed-42 analysis")
    if data.duplicated(["dataset", "split_mode", "rare_train_size", "seed"]).any():
        raise ValueError("Duplicate split-sensitivity ledger keys")

    scarce = data[data["rare_train_size"].isin(BUDGETS)].copy()
    if len(scarce) != 36:
        raise ValueError("Expected 6 datasets x 2 splits x 3 scarce budgets = 36 rows")
    metric_cols = ["scANVI_f1", "scRareRefine_f1", "delta_f1"]
    if not np.isfinite(scarce[metric_cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite rare-F1 value in scarce-region data")
    return data, scarce


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _paired_f1_panel(ax: plt.Axes, scarce: pd.DataFrame) -> None:
    positions = {
        "batch_heldout": (0.0, 0.75),
        "cell_stratified": (2.0, 2.75),
    }
    rng = np.random.default_rng(42)
    for split in SPLITS:
        frame = scarce[scarce["split_mode"].eq(split)].sort_values(
            ["dataset", "rare_train_size"]
        )
        x_base, x_refined = positions[split]
        jitter = rng.uniform(-0.045, 0.045, size=len(frame))
        for offset, (_, row) in zip(jitter, frame.iterrows()):
            ax.plot(
                [x_base + offset, x_refined + offset],
                [row["scANVI_f1"], row["scRareRefine_f1"]],
                color="#C7C7C7",
                linewidth=0.65,
                alpha=0.8,
                zorder=1,
            )
        ax.scatter(
            np.full(len(frame), x_base) + jitter,
            frame["scANVI_f1"],
            s=12,
            facecolor="white",
            edgecolor=BASELINE_COLOR,
            linewidth=0.7,
            zorder=2,
        )
        ax.scatter(
            np.full(len(frame), x_refined) + jitter,
            frame["scRareRefine_f1"],
            s=12,
            facecolor="white",
            edgecolor=REFINED_COLOR,
            linewidth=0.7,
            zorder=2,
        )
        means = [frame["scANVI_f1"].mean(), frame["scRareRefine_f1"].mean()]
        ax.plot(
            [x_base, x_refined],
            means,
            color="#222222",
            linewidth=1.4,
            zorder=3,
        )
        ax.scatter(
            [x_base, x_refined],
            means,
            s=48,
            color=[BASELINE_COLOR, REFINED_COLOR],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        for x, value in zip((x_base, x_refined), means):
            ax.text(x, min(value + 0.045, 1.025), f"{value:.3f}", ha="center", va="bottom")

    ax.set_xticks(
        [0.0, 0.75, 2.0, 2.75],
        ["scANVI", "scRareRefine", "scANVI", "scRareRefine"],
        rotation=12,
    )
    ax.text(0.375, -0.20, "Batch-heldout", transform=ax.get_xaxis_transform(), ha="center")
    ax.text(2.375, -0.20, "Cell-stratified", transform=ax.get_xaxis_transform(), ha="center")
    ax.set_ylabel("Rare-cell F1")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(-0.35, 3.1)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")


def _delta_matrix(scarce: pd.DataFrame, split: str) -> np.ndarray:
    frame = scarce[scarce["split_mode"].eq(split)]
    pivot = frame.pivot(index="dataset", columns="rare_train_size", values="delta_f1")
    return pivot.reindex(index=DATASETS, columns=BUDGETS).to_numpy(dtype=float)


def _heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    split_label: str,
    norm: TwoSlopeNorm,
    *,
    show_ylabels: bool,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(BUDGETS)), BUDGETS)
    ax.set_xlabel(f"Rare-label fraction\n{split_label}")
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASET_LABELS if show_ylabels else [""] * len(DATASETS))
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value > 0.55 or value < -0.045 else "#222222"
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", color=text_color, fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.6)
    return image


def make_figure(scarce: pd.DataFrame) -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(8.1, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.55, wspace=0.13)
    ax_top = fig.add_subplot(grid[0, :])
    ax_batch = fig.add_subplot(grid[1, 0])
    ax_cell = fig.add_subplot(grid[1, 1])

    _paired_f1_panel(ax_top, scarce)
    norm = TwoSlopeNorm(vmin=-0.07, vcenter=0.0, vmax=1.0)
    batch_matrix = _delta_matrix(scarce, "batch_heldout")
    cell_matrix = _delta_matrix(scarce, "cell_stratified")
    image = _heatmap(
        ax_batch, batch_matrix, "Batch-heldout", norm, show_ylabels=True
    )
    _heatmap(ax_cell, cell_matrix, "Cell-stratified", norm, show_ylabels=False)
    _panel_label(ax_batch, "b")
    _panel_label(ax_cell, "c")
    colorbar = fig.colorbar(image, ax=[ax_batch, ax_cell], fraction=0.032, pad=0.025)
    colorbar.set_label(r"$\Delta$ rare-cell F1 (scRareRefine $-$ scANVI)")
    colorbar.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.16, right=0.91, top=0.98, bottom=0.10)
    return fig


def main() -> None:
    _, scarce = _load_data()
    figure = make_figure(scarce)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=300)
    plt.close(figure)
    print(f"saved {pdf.relative_to(ROOT)}")
    print(f"saved {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
