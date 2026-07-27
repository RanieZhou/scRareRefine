"""Plot the completed TOSICA-backbone rescue portability results.

Only datasets with all four prespecified rare-label budgets are included. This
allows the same script to create the seven-dataset interim figure after mouse
lung finishes and overwrite it with the final eight-dataset figure after mouse
pancreas finishes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tosica_backbone_rescue" / "v1" / "run_level.csv"
FIG_DIR = ROOT / "results" / "tosica_backbone_rescue" / "v1" / "figures"
STEM = "tosica_backbone_rescue_summary"
BUDGETS = ("0.01", "0.05", "0.10", "all")
DATASET_ORDER = (
    "immune_dc",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
    "mouse_lung_tms_10x",
    "mouse_pancreas_tms_10x",
)
DATASET_LABELS = {
    "immune_dc": "Immune DC",
    "pancreas_baron": "Baron pancreas",
    "pancreas_integrated": "Integrated pancreas",
    "tabula_lung_endo": "Lung endothelium",
    "tabula_sapiens_stomach": "Stomach",
    "tabula_small_intestine": "Small intestine",
    "mouse_lung_tms_10x": "Mouse lung",
    "mouse_pancreas_tms_10x": "Mouse pancreas",
}
ALPHA = 0.01
BASELINE_COLOR = "#6F6F6F"
REFINED_COLOR = "#1B7F4B"
NEGATIVE_COLOR = "#B85C4A"


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


def load_completed(path: Path = INPUT, *, require_runs: int = 0) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"rare_train_size": str})
    required = {
        "dataset",
        "seed",
        "rare_train_size",
        "status",
        "baseline_rare_f1",
        "refined_rare_f1",
        "delta_rare_f1",
        "incremental_fpr",
        "alpha_violation",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if require_runs and len(data) < require_runs:
        raise ValueError(f"Ledger has {len(data)} rows; at least {require_runs} required")
    if data.duplicated(["dataset", "seed", "rare_train_size"]).any():
        raise ValueError("Duplicate TOSICA portability ledger keys")
    if not set(data["seed"].astype(int)).issubset({42}):
        raise ValueError("This figure is the prespecified seed-42 screen")

    complete_datasets = []
    for dataset in DATASET_ORDER:
        frame = data[data["dataset"].astype(str).eq(dataset)]
        successful = frame[frame["status"].eq("success")]
        if len(successful) == len(BUDGETS) and set(successful["rare_train_size"]) == set(BUDGETS):
            complete_datasets.append(dataset)
    if not complete_datasets:
        raise ValueError("No dataset has a complete four-budget result grid")
    completed = data[
        data["dataset"].isin(complete_datasets) & data["status"].eq("success")
    ].copy()
    if len(completed) != 4 * len(complete_datasets):
        raise ValueError("Completed dataset grid is not rectangular")
    metric_cols = [
        "baseline_rare_f1",
        "refined_rare_f1",
        "delta_rare_f1",
        "incremental_fpr",
    ]
    if not np.isfinite(completed[metric_cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite metric in completed TOSICA result grid")
    return completed


def _paired_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    ordered = data.copy()
    ordered["_dataset_order"] = ordered["dataset"].map(
        {value: index for index, value in enumerate(DATASET_ORDER)}
    )
    ordered["_budget_order"] = ordered["rare_train_size"].map(
        {value: index for index, value in enumerate(BUDGETS)}
    )
    ordered = ordered.sort_values(["_dataset_order", "_budget_order"])
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.055, 0.055, len(ordered))
    for offset, (_, row) in zip(jitter, ordered.iterrows()):
        delta = float(row["delta_rare_f1"])
        color = REFINED_COLOR if delta > 1e-12 else NEGATIVE_COLOR if delta < -1e-12 else "#BDBDBD"
        ax.plot(
            [0 + offset, 1 + offset],
            [row["baseline_rare_f1"], row["refined_rare_f1"]],
            color=color,
            linewidth=0.75,
            alpha=0.55,
            zorder=1,
        )
    ax.scatter(
        jitter,
        ordered["baseline_rare_f1"],
        s=15,
        facecolor="white",
        edgecolor=BASELINE_COLOR,
        linewidth=0.7,
        zorder=2,
    )
    ax.scatter(
        1 + jitter,
        ordered["refined_rare_f1"],
        s=15,
        facecolor="white",
        edgecolor=REFINED_COLOR,
        linewidth=0.7,
        zorder=2,
    )
    means = [ordered["baseline_rare_f1"].mean(), ordered["refined_rare_f1"].mean()]
    ax.plot([0, 1], means, color="#222222", linewidth=1.5, zorder=3)
    ax.scatter(
        [0, 1],
        means,
        s=52,
        color=[BASELINE_COLOR, REFINED_COLOR],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    for x, value in zip((0, 1), means):
        ax.text(x, min(value + 0.045, 1.03), f"{value:.3f}", ha="center", va="bottom")
    delta = ordered["delta_rare_f1"].to_numpy(dtype=float)
    wins = int((delta > 1e-12).sum())
    ties = int(np.isclose(delta, 0.0, atol=1e-12).sum())
    losses = int((delta < -1e-12).sum())
    ax.text(
        0.5,
        0.04,
        f"{wins} wins / {ties} ties / {losses} losses",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.set_xticks([0, 1], ["TOSICA", "TOSICA + rescue"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("Rare-cell F1")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    _panel_label(ax, "a")


def _matrix(data: pd.DataFrame, metric: str, datasets: list[str]) -> np.ndarray:
    pivot = data.pivot(index="dataset", columns="rare_train_size", values=metric)
    return pivot.reindex(index=datasets, columns=BUDGETS).to_numpy(dtype=float)


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    datasets: list[str],
    *,
    cmap: str,
    norm,
    ffr: bool,
    show_ylabels: bool = True,
) -> mpl.image.AxesImage:
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(BUDGETS)), BUDGETS)
    ax.set_yticks(
        range(len(datasets)),
        [DATASET_LABELS[value] for value in datasets]
        if show_ylabels
        else [""] * len(datasets),
    )
    ax.set_xlabel("Rare-label fraction")
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if ffr:
                label = f"{value:.3f}"
                violation = value > ALPHA
                text_color = "white" if value > max(ALPHA * 2.0, matrix.max() * 0.55) else "#222222"
                if violation:
                    ax.add_patch(
                        mpl.patches.Rectangle(
                            (col - 0.49, row - 0.49),
                            0.98,
                            0.98,
                            fill=False,
                            edgecolor="#8B0000",
                            linewidth=1.6,
                        )
                    )
            else:
                label = f"{value:+.3f}"
                limit = max(abs(norm.vmin), abs(norm.vmax))
                text_color = "white" if abs(value) > 0.58 * limit else "#222222"
            ax.text(col, row, label, ha="center", va="center", color=text_color, fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.6)
    return image


def make_figure(data: pd.DataFrame) -> plt.Figure:
    _style()
    datasets = [value for value in DATASET_ORDER if value in set(data["dataset"])]
    fig_height = 6.8 if len(datasets) <= 7 else 7.3
    fig = plt.figure(figsize=(8.4, fig_height))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.35], hspace=0.42, wspace=0.42)
    paired_ax = fig.add_subplot(grid[0, :])
    delta_ax = fig.add_subplot(grid[1, 0])
    ffr_ax = fig.add_subplot(grid[1, 1])
    _paired_panel(paired_ax, data)

    delta_matrix = _matrix(data, "delta_rare_f1", datasets)
    delta_abs = max(float(np.abs(delta_matrix).max()), 0.01)
    delta_norm = TwoSlopeNorm(vmin=-delta_abs, vcenter=0.0, vmax=delta_abs)
    delta_image = _annotated_heatmap(
        delta_ax,
        delta_matrix,
        datasets,
        cmap="RdBu",
        norm=delta_norm,
        ffr=False,
        show_ylabels=True,
    )
    delta_bar = fig.colorbar(delta_image, ax=delta_ax, fraction=0.046, pad=0.03)
    delta_bar.set_label(r"$\Delta$ rare-cell F1")
    delta_bar.outline.set_linewidth(0.6)
    _panel_label(delta_ax, "b")

    ffr_matrix = _matrix(data, "incremental_fpr", datasets)
    ffr_max = max(float(ffr_matrix.max()), ALPHA * 1.01)
    ffr_norm = TwoSlopeNorm(vmin=0.0, vcenter=ALPHA, vmax=ffr_max)
    ffr_image = _annotated_heatmap(
        ffr_ax,
        ffr_matrix,
        datasets,
        cmap="YlOrRd",
        norm=ffr_norm,
        ffr=True,
        show_ylabels=False,
    )
    ffr_bar = fig.colorbar(ffr_image, ax=ffr_ax, fraction=0.046, pad=0.03)
    ffr_bar.set_label("Incremental FPR")
    ffr_bar.outline.set_linewidth(0.6)
    ffr_bar.ax.axhline(ALPHA, color="#8B0000", linewidth=1.0)
    _panel_label(ffr_ax, "c")
    fig.subplots_adjust(left=0.17, right=0.94, top=0.98, bottom=0.08)
    return fig


def _write_latex(n_datasets: int, n_runs: int) -> None:
    path = FIG_DIR / f"{STEM}_figure.tex"
    path.write_text(
        "% Generated from results/tosica_backbone_rescue/v1/run_level.csv.\n"
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.95\\textwidth]{{results/tosica_backbone_rescue/v1/figures/{STEM}.pdf}}\n"
        "  \\caption{Single-seed portability of the fixed rescue procedure to native TOSICA embeddings. "
        f"The figure contains {n_runs} completed dataset--budget runs from {n_datasets} datasets. "
        "(a) Paired rare-cell F1 before and after rescue; large markers denote descriptive means. "
        "(b) Per-run F1 changes. (c) Empirical incremental false-positive rates; red borders mark "
        "values above the fixed $\\alpha=0.01$ budget. TOSICA predictions and 48-dimensional CLS "
        "embeddings were frozen before validation-only rescue calibration and final test evaluation.}\n"
        "  \\label{fig:tosica-backbone-rescue}\n"
        "\\end{figure*}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-runs", type=int, default=0)
    args = parser.parse_args()
    data = load_completed(require_runs=args.require_runs)
    figure = make_figure(data)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{STEM}.pdf"
    png = FIG_DIR / f"{STEM}.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=300)
    plt.close(figure)
    _write_latex(data["dataset"].nunique(), len(data))
    print(f"saved {pdf.relative_to(ROOT)}")
    print(f"saved {png.relative_to(ROOT)}")
    print(f"datasets={data['dataset'].nunique()} runs={len(data)}")


if __name__ == "__main__":
    main()
